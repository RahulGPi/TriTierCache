import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from src.tri_tier.cache import TriTierCache

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


def _reference_attention_path(self, cache: TriTierCache, Q: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int):
    K_full, V_full, full_ids = cache.reconstruct_full_cache()

    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        K_full = K_full.repeat_interleave(repeat_factor, dim=1)
        V_full = V_full.repeat_interleave(repeat_factor, dim=1)

    K_full = K_full.transpose(0, 1).unsqueeze(0)   # [total_kv,heads,dim] -> [1,heads,total_kv,dim]
    V_full = V_full.transpose(0, 1).unsqueeze(0)

    attn_scores = torch.matmul(Q, K_full.transpose(-2, -1)) * self.scaling
    attn_weights = torch.softmax(attn_scores, dim=-1)

    cache.accumulate_attn_scrs(attn_weights, full_ids)

    context_layer = torch.matmul(attn_weights, V_full)          # [1, q_heads, 1, head_dim]
    return context_layer, attn_weights


def patched_forward(self, 
                    hidden_states, 
                    position_embeddings=None, 
                    attention_mask=None, 
                    past_key_values=None, 
                    **kwargs):
    """
    Wrapper for transformers.models.llama.modeling_llama.LlamaAttention
    uses the exact same input as the original function
    """

    bsz, q_len, _ = hidden_states.shape
    assert bsz == 1, "TriTierCache only supports batch_size=1"
 
    num_q_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
 
    # ---- Part 1: lazy init ----
    if not hasattr(self, "tri_tier_cache") or self.tri_tier_cache is None:
        self.tri_tier_cache = TriTierCache(
            max_seq_len=MAX_SEQ_LEN,
            head_dim=head_dim,
            num_heads=num_kv_heads,
            R_size=R_SIZE,
            H_ratio=H_RATIO,
            num_q_heads=num_q_heads,
            score_decay=SCORE_DECAY,
        )
    cache = self.tri_tier_cache

    # ---- Part 2: project and rotate ----
    Q = self.q_proj(hidden_states).view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    K_new = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    V_new = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
 
    cos, sin = position_embeddings
    Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

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


_ORIGINAL_FORWARD = LlamaAttention.forward


def apply_patch():
    LlamaAttention.forward = patched_forward


def remove_patch():
    LlamaAttention.forward = _ORIGINAL_FORWARD


def reset_caches(model):
    """Resets any lazily created TriTierCache instances on the model layers."""
    for module in model.modules():
        if hasattr(module, "tri_tier_cache"):
            delattr(module, "tri_tier_cache")
