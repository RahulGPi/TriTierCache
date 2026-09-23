"""
src/tri_tier/integration/patch_model.py
Universal Multi-Model Integration Layer for TriTierCache.

Supports:
- LLaMA family: LLaMA 3, LLaMA 3.1, LLaMA 3.2, SmolLM, SmolLM2, TinyLlama
- Mistral family: Mistral-7B, Mistral-Instruct
- Qwen family: Qwen 2, Qwen 2.5, Qwen 3

Dynamically handles MHA and GQA, head_dim (64, 128), and long context scaling up to 32k+.
"""

from typing import Dict, List, Any, Optional, Tuple, Type
import torch
from ..cache import TriTierCache

try:
    import tri_tier._C as _C
    HAS_CPP_EXT = True
except ImportError:
    _C = None
    HAS_CPP_EXT = False


MAX_SEQ_LEN = 32768
R_SIZE = 256
H_RATIO = 0.05
SCORE_DECAY = 0.999
K_GROUP_SIZE = 16
PBS_METADATA_DTYPE = "fp16"


def _reference_attention_path(self, cache: TriTierCache, Q: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int):
    K_full, V_full, full_ids = cache.reconstruct_full_cache()

    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        K_full = K_full.repeat_interleave(repeat_factor, dim=1)
        V_full = V_full.repeat_interleave(repeat_factor, dim=1)

    K_full = K_full.transpose(0, 1).unsqueeze(0)   # [total_kv, heads, dim] -> [1, heads, total_kv, dim]
    V_full = V_full.transpose(0, 1).unsqueeze(0)

    scaling = getattr(self, "scaling", None) or (head_dim ** -0.5)
    attn_scores = torch.matmul(Q, K_full.transpose(-2, -1)) * scaling
    attn_weights = torch.softmax(attn_scores, dim=-1)

    cache.accumulate_attn_scrs(attn_weights, full_ids)

    context_layer = torch.matmul(attn_weights, V_full)          # [1, q_heads, 1, head_dim]
    return context_layer, attn_weights


def patched_forward(self,
                    hidden_states: torch.Tensor,
                    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    past_key_values: Optional[Any] = None,
                    **kwargs):
    """
    Universal forward wrapper supporting LlamaAttention, MistralAttention,
    Qwen2Attention, and Qwen3Attention.
    """
    bsz, q_len, _ = hidden_states.shape
    assert bsz == 1, "TriTierCache only supports batch_size=1"

    num_q_heads = self.config.num_attention_heads
    num_kv_heads = getattr(self.config, "num_key_value_heads", num_q_heads)
    head_dim = getattr(self, "head_dim", None) or (self.config.hidden_size // num_q_heads)

    # Context capacity: at least 32768 or model max_position_embeddings
    max_pos = getattr(self.config, "max_position_embeddings", MAX_SEQ_LEN)
    seq_len_cap = max(MAX_SEQ_LEN, max_pos)

    # ---- Part 1: lazy init ----
    if not hasattr(self, "tri_tier_cache") or self.tri_tier_cache is None:
        self.tri_tier_cache = TriTierCache(
            max_seq_len=seq_len_cap,
            head_dim=head_dim,
            num_heads=num_kv_heads,
            R_size=R_SIZE,
            H_ratio=H_RATIO,
            num_q_heads=num_q_heads,
            score_decay=SCORE_DECAY,
            k_group_size=K_GROUP_SIZE,
            pbs_metadata_dtype=PBS_METADATA_DTYPE,
        )
    cache = self.tri_tier_cache

    # ---- Part 2: project and rotate ----
    Q = self.q_proj(hidden_states).view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    K_new = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    V_new = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Rotary position embedding application
    cos, sin = position_embeddings
    # Use standard rotary embedding calculation:
    # Rotate half
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    Q = (Q * cos) + (_rotate_half(Q) * sin)
    K_new = (K_new * cos) + (_rotate_half(K_new) * sin)

    need_full_weights = kwargs.get("output_attentions", False)

    # ---- Case A: Batched Prefill (q_len > 1) ----
    if q_len > 1:
        repeat_factor = num_q_heads // num_kv_heads
        if repeat_factor > 1:
            K_exp = K_new.repeat_interleave(repeat_factor, dim=1)
            V_exp = V_new.repeat_interleave(repeat_factor, dim=1)
        else:
            K_exp = K_new
            V_exp = V_new

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            Q, K_exp, V_exp, is_causal=True
        )

        K_tokens = K_new.squeeze(0).transpose(0, 1).contiguous()
        V_tokens = V_new.squeeze(0).transpose(0, 1).contiguous()
        if hasattr(self, "capture_needle_pos") and self.capture_needle_pos is not None:
            pos = self.capture_needle_pos
            self.captured_needle_k = K_tokens[pos].clone()
            self.captured_needle_v = V_tokens[pos].clone()
        cache.prefill(K_tokens, V_tokens)

        context_layer = context_layer.transpose(1, 2).reshape(bsz, q_len, -1)
        final_output = self.o_proj(context_layer)
        return final_output, None

    # ---- Case B: Single-token Decode Step (q_len == 1) ----
    if cache._engine is not None and not need_full_weights:
        Q_flat = Q.reshape(num_q_heads * head_dim).contiguous()
        K_flat = K_new.reshape(num_kv_heads * head_dim).contiguous()
        V_flat = V_new.reshape(num_kv_heads * head_dim).contiguous()

        attn_output = torch.empty((num_q_heads * head_dim,), dtype=torch.float32, device="cpu")

        cache.step(Q_flat, K_flat, V_flat, attn_output)

        context_layer = attn_output.view(bsz, q_len, -1)
        final_output = self.o_proj(context_layer)
        return final_output, None
    else:
        K_new_flat = K_new.squeeze(0).squeeze(1)
        V_new_flat = V_new.squeeze(0).squeeze(1)
        cache.ingest_token(K_new_flat, V_new_flat)
        context_layer, attn_weights = _reference_attention_path(self, cache, Q, num_q_heads, num_kv_heads, head_dim)
        context_layer = context_layer.transpose(1, 2).reshape(bsz, q_len, -1)
        final_output = self.o_proj(context_layer)
        return final_output, attn_weights


# ---------------------------------------------------------------------------
# Attention Classes Discovery & Patch Management
# ---------------------------------------------------------------------------

_SUPPORTED_ATTENTION_CLASSES: List[Type] = []
_ORIGINAL_FORWARDS: Dict[Type, Any] = {}
_IS_PATCHED: bool = False

# 1. LLaMA attention (LLaMA 3/3.1/3.2, SmolLM, SmolLM2, TinyLlama)
try:
    from transformers.models.llama.modeling_llama import LlamaAttention
    _SUPPORTED_ATTENTION_CLASSES.append(LlamaAttention)
except ImportError:
    pass

# 2. Mistral attention (Mistral 7B)
try:
    from transformers.models.mistral.modeling_mistral import MistralAttention
    _SUPPORTED_ATTENTION_CLASSES.append(MistralAttention)
except ImportError:
    pass

# 3. Qwen2 attention (Qwen 2, Qwen 2.5)
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
    _SUPPORTED_ATTENTION_CLASSES.append(Qwen2Attention)
except ImportError:
    pass

# 4. Qwen3 attention (Qwen 3)
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    _SUPPORTED_ATTENTION_CLASSES.append(Qwen3Attention)
except ImportError:
    pass


def apply_patch():
    """Patches all supported attention classes across LLaMA, Mistral, and Qwen families."""
    global _IS_PATCHED
    for cls in _SUPPORTED_ATTENTION_CLASSES:
        if cls not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[cls] = cls.forward
        cls.forward = patched_forward
    _IS_PATCHED = True


def remove_patch():
    """Restores all patched attention classes to original Hugging Face implementation."""
    global _IS_PATCHED
    for cls, orig_fwd in _ORIGINAL_FORWARDS.items():
        cls.forward = orig_fwd
    _IS_PATCHED = False


def reset_caches(model):
    """Resets any lazily created TriTierCache instances on any model layers."""
    for module in model.modules():
        if hasattr(module, "tri_tier_cache"):
            delattr(module, "tri_tier_cache")


def is_patched(model_or_module=None) -> bool:
    """Checks whether TriTierCache attention patching is currently active."""
    if _IS_PATCHED:
        return True
    for cls in _SUPPORTED_ATTENTION_CLASSES:
        if getattr(cls, "forward", None) == patched_forward:
            return True
    return False
