// csrc/cpu/quantize_k_avx2.h
#pragma once
#include <cstdint>

void quantize_k_block_avx2(
    const float* K,       // [k_group_size, num_heads, head_dim], contiguous, token-major
    uint32_t* packed,     // out: [(k_group_size / 16) * num_heads, head_dim]
    float* scale_out,     // out: [num_heads, head_dim] (FP32)
    float* zero_out,      // out: [num_heads, head_dim]  (Min_hc, FP32)
    int num_heads,
    int head_dim,
    int k_group_size = 16);

void quantize_k_block_avx2(
    const float* K,       // [k_group_size, num_heads, head_dim], contiguous, token-major
    uint32_t* packed,     // out: [(k_group_size / 16) * num_heads, head_dim]
    uint16_t* scale_out,  // out: [num_heads, head_dim] (FP16 / half)
    uint16_t* zero_out,   // out: [num_heads, head_dim]  (Min_hc, FP16 / half)
    int num_heads,
    int head_dim,
    int k_group_size = 16);