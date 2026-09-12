// csrc/cpu/quantize_k_avx2.h
#pragma once
#include <cstdint>

void quantize_k_block_avx2(
    const float* K,       // [16, num_heads, head_dim], contiguous, token-major
    uint32_t* packed,     // out: [num_heads, head_dim]
    float* scale_out,     // out: [num_heads, head_dim]
    float* zero_out,      // out: [num_heads, head_dim]  (this is Min_hc)
    int num_heads,
    int head_dim);