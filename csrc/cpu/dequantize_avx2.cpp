#include "../include/dequant_avx2.h"
#include <immintrin.h>
#include <cstdint>

// Loading helpers for float (FP32) vs uint16_t (FP16 via F16C)
static inline __m256 load_meta_vec(const float* src) {
    return _mm256_loadu_ps(src);
}

static inline __m256 load_meta_vec(const uint16_t* src) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    return _mm256_cvtph_ps(h);
}

static inline float load_meta_scalar(const float* src) {
    return *src;
}

static inline float load_meta_scalar(const uint16_t* src) {
    __m128i h = _mm_cvtsi32_si128(static_cast<int>(*src));
    return _mm_cvtss_f32(_mm_cvtph_ps(h));
}

template <typename MetaT>
static void dequantize_k_impl(
    const int32_t* packed,
    const MetaT* scale,
    const MetaT* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    const int words_per_k_block = k_group_size / 16;
    const int block_packed_stride = words_per_k_block * num_heads * head_dim;
    const int block_meta_stride = num_heads * head_dim;
    const int out_row_stride = num_heads * head_dim;
    const __m256i mask3 = _mm256_set1_epi32(0b11);

    for (int b = 0; b < num_blocks; ++b) {
        const int32_t* block_packed = packed + b * block_packed_stride;
        const MetaT* block_scale = scale + b * block_meta_stride;
        const MetaT* block_zero = zero + b * block_meta_stride;

        for (int h = 0; h < num_heads; ++h) {
            const MetaT* head_scale = block_scale + h * head_dim;
            const MetaT* head_zero = block_zero + h * head_dim;

            int c = 0;
            for (; c + 16 <= head_dim; c += 16) {
                __m256 svec0 = load_meta_vec(head_scale + c);
                __m256 svec1 = load_meta_vec(head_scale + c + 8);
                __m256 zvec0 = load_meta_vec(head_zero + c);
                __m256 zvec1 = load_meta_vec(head_zero + c + 8);

                for (int w = 0; w < words_per_k_block; ++w) {
                    const int32_t* word_packed = block_packed + (w * num_heads + h) * head_dim;
                    __m256i pvec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(word_packed + c));
                    __m256i pvec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(word_packed + c + 8));

                    for (int sub_t = 0; sub_t < 16; ++sub_t) {
                        __m256i q0 = _mm256_and_si256(_mm256_srli_epi32(pvec0, 2 * sub_t), mask3);
                        __m256i q1 = _mm256_and_si256(_mm256_srli_epi32(pvec1, 2 * sub_t), mask3);
                        __m256 dq0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q0), svec0, zvec0);
                        __m256 dq1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1), svec1, zvec1);

                        int global_token = b * k_group_size + w * 16 + sub_t;
                        float* out_ptr = out + global_token * out_row_stride + h * head_dim + c;
                        _mm256_storeu_ps(out_ptr, dq0);
                        _mm256_storeu_ps(out_ptr + 8, dq1);
                    }
                }
            }
            for (; c + 8 <= head_dim; c += 8) {
                __m256 svec = load_meta_vec(head_scale + c);
                __m256 zvec = load_meta_vec(head_zero + c);

                for (int w = 0; w < words_per_k_block; ++w) {
                    const int32_t* word_packed = block_packed + (w * num_heads + h) * head_dim;
                    __m256i pvec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(word_packed + c));

                    for (int sub_t = 0; sub_t < 16; ++sub_t) {
                        __m256i q = _mm256_and_si256(_mm256_srli_epi32(pvec, 2 * sub_t), mask3);
                        __m256 dq = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), svec, zvec);

                        int global_token = b * k_group_size + w * 16 + sub_t;
                        float* out_ptr = out + global_token * out_row_stride + h * head_dim + c;
                        _mm256_storeu_ps(out_ptr, dq);
                    }
                }
            }
            for (; c < head_dim; ++c) {
                float s = load_meta_scalar(head_scale + c);
                float z = load_meta_scalar(head_zero + c);

                for (int w = 0; w < words_per_k_block; ++w) {
                    const int32_t* word_packed = block_packed + (w * num_heads + h) * head_dim;
                    int32_t p = word_packed[c];

                    for (int sub_t = 0; sub_t < 16; ++sub_t) {
                        int q = (p >> (2 * sub_t)) & 0b11;
                        int global_token = b * k_group_size + w * 16 + sub_t;
                        out[global_token * out_row_stride + h * head_dim + c] = q * s + z;
                    }
                }
            }
        }
    }
}

void dequantize_k_avx2(
    const int32_t* packed,
    const float* scale,
    const float* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    dequantize_k_impl<float>(packed, scale, zero, out, num_blocks, num_heads, head_dim, k_group_size);
}

void dequantize_k_avx2(
    const int32_t* packed,
    const uint16_t* scale,
    const uint16_t* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    dequantize_k_impl<uint16_t>(packed, scale, zero, out, num_blocks, num_heads, head_dim, k_group_size);
}

template <typename MetaT>
static void dequantize_v_impl(
    const int32_t* packed,
    const MetaT* scale,
    const MetaT* zero,
    float* out,
    int total_tokens,
    int num_heads,
    int head_dim)
{
    const int quant_head_dim = (head_dim + 15) / 16;
    const int packed_row_stride = num_heads * quant_head_dim;
    const int out_row_stride = num_heads * head_dim;

    const __m256i mask3 = _mm256_set1_epi32(0b11);
    const __m256i shifts_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i shifts_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);

    for (int tok = 0; tok < total_tokens; ++tok) {
        const int32_t* tok_packed = packed + tok * packed_row_stride;

        for (int h = 0; h < num_heads; ++h) {
            const int32_t* head_packed = tok_packed + h * quant_head_dim;
            float s = load_meta_scalar(scale + tok * num_heads + h);
            float z = load_meta_scalar(zero + tok * num_heads + h);
            __m256 svec = _mm256_set1_ps(s);
            __m256 zvec = _mm256_set1_ps(z);
            float* out_head = out + tok * out_row_stride + h * head_dim;

            int g = 0;
            for (; g + 2 <= quant_head_dim; g += 2) {
                int32_t p0 = head_packed[g];
                int32_t p1 = head_packed[g + 1];

                __m256i pvec0 = _mm256_set1_epi32(p0);
                __m256i pvec1 = _mm256_set1_epi32(p1);

                __m256i q0_0 = _mm256_and_si256(_mm256_srlv_epi32(pvec0, shifts_lo), mask3);
                __m256i q0_1 = _mm256_and_si256(_mm256_srlv_epi32(pvec0, shifts_hi), mask3);
                __m256i q1_0 = _mm256_and_si256(_mm256_srlv_epi32(pvec1, shifts_lo), mask3);
                __m256i q1_1 = _mm256_and_si256(_mm256_srlv_epi32(pvec1, shifts_hi), mask3);

                __m256 dq0_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q0_0), svec, zvec);
                __m256 dq0_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q0_1), svec, zvec);
                __m256 dq1_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1_0), svec, zvec);
                __m256 dq1_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1_1), svec, zvec);

                _mm256_storeu_ps(out_head + g * 16, dq0_0);
                _mm256_storeu_ps(out_head + g * 16 + 8, dq0_1);
                _mm256_storeu_ps(out_head + (g + 1) * 16, dq1_0);
                _mm256_storeu_ps(out_head + (g + 1) * 16 + 8, dq1_1);
            }
            for (; g < quant_head_dim; ++g) {
                int32_t p = head_packed[g];
                __m256i pvec = _mm256_set1_epi32(p);
                __m256i q0 = _mm256_and_si256(_mm256_srlv_epi32(pvec, shifts_lo), mask3);
                __m256i q1 = _mm256_and_si256(_mm256_srlv_epi32(pvec, shifts_hi), mask3);

                __m256 dq0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q0), svec, zvec);
                __m256 dq1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q1), svec, zvec);

                _mm256_storeu_ps(out_head + g * 16, dq0);
                _mm256_storeu_ps(out_head + g * 16 + 8, dq1);
            }
        }
    }
}

void dequantize_v_avx2(
    const int32_t* packed,
    const float* scale,
    const float* zero,
    float* out,
    int total_tokens,
    int num_heads,
    int head_dim)
{
    dequantize_v_impl<float>(packed, scale, zero, out, total_tokens, num_heads, head_dim);
}

void dequantize_v_avx2(
    const int32_t* packed,
    const uint16_t* scale,
    const uint16_t* zero,
    float* out,
    int total_tokens,
    int num_heads,
    int head_dim)
{
    dequantize_v_impl<uint16_t>(packed, scale, zero, out, total_tokens, num_heads, head_dim);
}