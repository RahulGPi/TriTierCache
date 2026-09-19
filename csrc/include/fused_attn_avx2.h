#pragma once
#include <cstdint>


void fused_attention_decode_avx2(
    const float* Q,                 // [num_q_heads, head_dim]
    const float* dense_K,           // [dense_count, num_kv_heads, head_dim]
    const float* dense_V,           // [dense_count, num_kv_heads, head_dim]
    int dense_count,
    const int32_t* PBS_K_Packed,    // [num_blocks, num_kv_heads, head_dim]
    const float* PBS_K_Scales,
    const float* PBS_K_Zeroes,
    const int32_t* PBS_V_Packed,    // [num_blocks*16, num_kv_heads, quant_head_dim]
    const float* PBS_V_Scales,      // [num_blocks*16, num_kv_heads]
    const float* PBS_V_Zeroes,
    const int64_t* PBS_token_ids,   // [num_blocks*16], -1 marks an unused slot
    int num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float* attn_output,             // [num_q_heads, head_dim]
    float* mean_attn_weights = nullptr); // [dense_count + num_blocks * 16] (optional)