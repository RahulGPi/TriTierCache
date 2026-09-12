import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from tri_tier.cache import TriTierCache


MAX_SEQ_LEN = 32768
R_SIZE = 256
H_RATIO = 0.05

def patched_forward(self, 
                    hidden_states, 
                    position_embeddings=None, 
                    attention_mask=None, 
                    past_key_values=None, 
                    **kwargs):
    """
    Wrapper for transformers.models.llama.modelling_llama.LLamaAttention
    uses the exact same input as the orgiinal function
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
 
    # ---- Part 5: reconstruct ----
    K_full, V_full, full_ids = cache.reconstruct_full_cache()   # [total_kv, kv_heads, head_dim]
 
    # ---- Part 5b: GQA -- expand KV heads to match query head count ----
    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        K_full = K_full.repeat_interleave(repeat_factor, dim=1)
        V_full = V_full.repeat_interleave(repeat_factor, dim=1)
 
    K_full = K_full.transpose(0, 1).unsqueeze(0)   # [total_kv,heads,dim] -> [1,heads,total_kv,dim]
    V_full = V_full.transpose(0, 1).unsqueeze(0)
 
    # ---- Part 6: attention math ----
    attn_scores = torch.matmul(Q, K_full.transpose(-2, -1)) * self.scaling
    attn_weights = torch.softmax(attn_scores, dim=-1)
 
    # ---- Part 7: accumulate, using the ids reconstruction handed back ----
    cache.accumulate_attn_scrs(attn_weights, full_ids)
 
    # ---- Part 8: output ----
    context_layer = torch.matmul(attn_weights, V_full)          # [1, q_heads, 1, head_dim]
    context_layer = context_layer.transpose(1, 2).reshape(bsz, q_len, -1)
    final_output = self.o_proj(context_layer)
 
    return final_output, attn_weights
 
 
def apply_patch():
    LlamaAttention.forward = patched_forward


