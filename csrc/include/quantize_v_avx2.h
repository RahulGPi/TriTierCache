#pragma once
#include <cstdint>

void quantize_v_block_avx2(
    const float* V,
    int32_t* packed,
    float* scale_out,
    float* zero_out,
    int num_heads,
    int head_dim,
    int chunk_size = 16);

void quantize_v_block_avx2(
    const float* V,
    int32_t* packed,
    uint16_t* scale_out,
    uint16_t* zero_out,
    int num_heads,
    int head_dim,
    int chunk_size = 16);