#pragma once
#include <cstdint>

void dequantize_k_avx2(
    const int32_t* packed,
    const float* scale,
    const float* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size = 16);

void dequantize_k_avx2(
    const int32_t* packed,
    const uint16_t* scale,
    const uint16_t* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size = 16);

void dequantize_v_avx2(
    const int32_t* packed,
    const float* scale,
    const float* zero,
    float* out,
    int total_tokens,
    int num_heads,
    int head_dim);

void dequantize_v_avx2(
    const int32_t* packed,
    const uint16_t* scale,
    const uint16_t* zero,
    float* out,
    int total_tokens,
    int num_heads,
    int head_dim);