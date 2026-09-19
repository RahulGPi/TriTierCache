import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from tri_tier.cache import TriTierCache

try:
    import tri_tier._C as _C
    HAS_CPP_EXT = True
except ImportError:
    _C = None
    HAS_CPP_EXT = False


MAX_SEQ_LEN = 32768
R_SIZE = 256
H_RATIO = 0.05


def _check_tensor(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    assert t.is_cpu, f"{name} must be on CPU"
    assert t.dtype == dtype, f"{name} must be {dtype}, got {t.dtype}"
    assert t.is_contiguous(), f"{name} must be contiguous"


def _fused_attention_path(self, cache: TriTierCache, Q: torch.Tensor, num_q_heads: int, num_kv_heads: int, head_dim: int):
    sink_K = cache.S_K_Buffer[:cache.S_count]
    sink_V = cache.S_V_Buffer[:cache.S_count]

    hh_K = cache.HH_K_Buffer[:cache.HH_count]
    hh_V = cache.HH_V_Buffer[:cache.HH_count]

    if cache.RW_count < cache.R_size:
        rw_K = cache.RW_K_Buffer[:cache.RW_count]
        rw_V = cache.RW_V_Buffer[:cache.RW_count]
    else:
        rw_K = torch.roll(cache.RW_K_Buffer, shifts=-cache.RW_head_index, dims=0)
        rw_V = torch.roll(cache.RW_V_Buffer, shifts=-cache.RW_head_index, dims=0)

    dense_K = torch.cat([sink_K, hh_K, rw_K], dim=0).contiguous()
    dense_V = torch.cat([sink_V, hh_V, rw_V], dim=0).contiguous()
    dense_count = dense_K.shape[0]

    Q_flat = Q.squeeze(0).squeeze(1).contiguous()

    _check_tensor(Q_flat, "Q_flat", torch.float32)
    _check_tensor(dense_K, "dense_K", torch.float32)
    _check_tensor(dense_V, "dense_V", torch.float32)
    _check_tensor(cache.PBS_K_Packed, "PBS_K_Packed", torch.int32)
    _check_tensor(cache.PBS_K_Scales, "PBS_K_Scales", torch.float32)
    _check_tensor(cache.PBS_K_Zeroes, "PBS_K_Zeroes", torch.float32)
    _check_tensor(cache.PBS_V_Packed, "PBS_V_Packed", torch.int32)
    _check_tensor(cache.PBS_V_Scales, "PBS_V_Scales", torch.float32)
    _check_tensor(cache.PBS_V_Zeroes, "PBS_V_Zeroes", torch.float32)
    _check_tensor(cache.PBS_token_ids, "PBS_token_ids", torch.int64)
    assert head_dim % 16 == 0, f"head_dim={head_dim} must be a multiple of 16"

    attn_output = torch.empty((num_q_heads, head_dim), dtype=torch.float32, device="cpu")
    total_tokens = dense_count + cache.num_blocks * 16
    mean_attn_weights = torch.empty((total_tokens,), dtype=torch.float32, device="cpu")

    _C.fused_attention_decode(
        Q_flat.data_ptr(),
        dense_K.data_ptr(),
        dense_V.data_ptr(),
        dense_count,
        cache.PBS_K_Packed.data_ptr(),
        cache.PBS_K_Scales.data_ptr(),
        cache.PBS_K_Zeroes.data_ptr(),
        cache.PBS_V_Packed.data_ptr(),
        cache.PBS_V_Scales.data_ptr(),
        cache.PBS_V_Zeroes.data_ptr(),
        cache.PBS_token_ids.data_ptr(),
        cache.num_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        attn_output.data_ptr(),
        mean_attn_weights.data_ptr(),
    )

    # Accumulate global attention scores
    sink_ids = torch.arange(0, cache.S_count, dtype=torch.int64)
    hh_ids = cache.HH_token_ids[:cache.HH_count]
    if cache.RW_count < cache.R_size:
        rw_ids = torch.arange(cache.S_count, cache.S_count + cache.RW_count, dtype=torch.int64)
    else:
        rw_ids = torch.arange(cache.Total_Processed_Tokens - cache.R_size, cache.Total_Processed_Tokens, dtype=torch.int64)

    all_ids = torch.cat([sink_ids, hh_ids, rw_ids, cache.PBS_token_ids], dim=0)
    valid_mask = all_ids != -1
    cache.Global_Attn_Scr.index_add_(0, all_ids[valid_mask], mean_attn_weights[valid_mask])

    context_layer = attn_output.unsqueeze(0).unsqueeze(2)
    return context_layer, None


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
    assert q_len == 1, "TriTierCache only supports one new token per forward call"
 
    num_q_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
 
    # ---- Part 1: one cache per layer, created lazily on first use ----
    if not hasattr(self, "tri_tier_cache"):
        self.tri_tier_cache = TriTierCache(
            max_seq_len=MAX_SEQ_LEN,
            head_dim=head_dim,
            num_heads=num_kv_heads,   # cache stores K/V -- size by KV heads, not query heads
            R_size=R_SIZE,
            H_ratio=H_RATIO,
        )
    cache = self.tri_tier_cache
 
    # ---- Part 2: project and rotate ----
    Q = self.q_proj(hidden_states).view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    K_new = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    V_new = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
 
    cos, sin = position_embeddings
    Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)
 
    # ---- Part 3: strip batch + seq dims before touching the cache ----
    K_new_flat = K_new.squeeze(0).squeeze(1)   # [1,kv_heads,1,head_dim] -> [kv_heads, head_dim]
    V_new_flat = V_new.squeeze(0).squeeze(1)
 
    # ---- Part 4: ingest BEFORE reconstructing ----
    cache.ingest_token(K_new_flat, V_new_flat)
 
    # ---- Part 5: dispatch attention path ----
    need_full_weights = kwargs.get("output_attentions", False)

    if HAS_CPP_EXT and not need_full_weights:
        context_layer, attn_weights = _fused_attention_path(self, cache, Q, num_q_heads, num_kv_heads, head_dim)
    else:
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
