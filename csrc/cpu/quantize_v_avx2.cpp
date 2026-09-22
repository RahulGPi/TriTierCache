#include "../include/quantize_v_avx2.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

// Horizontal reduction helpers
static inline float hmin8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_min_ps(lo, hi);
    m = _mm_min_ps(m, _mm_movehl_ps(m, m));
    m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

static inline float hmax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

// Horizontal OR
static inline uint32_t hor32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_or_si128(lo, hi);
    m = _mm_or_si128(m, _mm_srli_si128(m, 8));
    m = _mm_or_si128(m, _mm_srli_si128(m, 4));
    return static_cast<uint32_t>(_mm_cvtsi128_si32(m));
}

static inline void store_meta_scalar(float* dest, float val) {
    *dest = val;
}

static inline void store_meta_scalar(uint16_t* dest, float val) {
    __m128 v = _mm_set_ss(val);
    *dest = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0)));
}

template <typename MetaT>
static void quantize_v_block_impl(
    const float* V,
    int32_t* packed,
    MetaT* scale_out,
    MetaT* zero_out,
    int num_heads,
    int head_dim,
    int chunk_size)
{
    const int quant_head_dim = (head_dim + 15) / 16;
    const int row_stride = num_heads * head_dim;

    const __m256i shifts_lo = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i shifts_hi = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i vzero_i = _mm256_setzero_si256();
    const __m256i vthree = _mm256_set1_epi32(3);

    for (int t = 0; t < chunk_size; ++t) {
        const float* token_base = V + t * row_stride;

        for (int h = 0; h < num_heads; ++h) {
            const float* v = token_base + h * head_dim;

            float mn = INFINITY, mx = -INFINITY;
            int c = 0;
            for (; c + 8 <= head_dim; c += 8) {
                __m256 x = _mm256_loadu_ps(v + c);
                mn = std::min(mn, hmin8(x));
                mx = std::max(mx, hmax8(x));
            }

            for (; c < head_dim; ++c) {
                mn = std::min(mn, v[c]);
                mx = std::max(mx, v[c]);
            }

            float scale = std::max((mx - mn) / 3.0f, 1e-9f);
            store_meta_scalar(scale_out + t * num_heads + h, scale);
            store_meta_scalar(zero_out + t * num_heads + h, mn);

            const __m256 vmn = _mm256_set1_ps(mn);
            const __m256 vscale = _mm256_set1_ps(scale);

            for (int g = 0; g < quant_head_dim; ++g) {
                int base_c = g * 16;

                __m256 x0 = _mm256_loadu_ps(v + base_c);
                __m256 x1 = _mm256_loadu_ps(v + base_c + 8);

                __m256i q0 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_sub_ps(x0, vmn), vscale));
                __m256i q1 = _mm256_cvtps_epi32(_mm256_div_ps(_mm256_sub_ps(x1, vmn), vscale));

                q0 = _mm256_max_epi32(_mm256_min_epi32(q0, vthree), vzero_i);
                q1 = _mm256_max_epi32(_mm256_min_epi32(q1, vthree), vzero_i);

                __m256i shifted0 = _mm256_sllv_epi32(q0, shifts_lo);
                __m256i shifted1 = _mm256_sllv_epi32(q1, shifts_hi);

                uint32_t packed_val = hor32(shifted0) | hor32(shifted1);
                packed[(t * num_heads + h) * quant_head_dim + g] = static_cast<int32_t>(packed_val);
            }
        }
    }
}

void quantize_v_block_avx2(
    const float* V,
    int32_t* packed,
    float* scale_out,
    float* zero_out,
    int num_heads,
    int head_dim,
    int chunk_size)
{
    quantize_v_block_impl<float>(V, packed, scale_out, zero_out, num_heads, head_dim, chunk_size);
}

void quantize_v_block_avx2(
    const float* V,
    int32_t* packed,
    uint16_t* scale_out,
    uint16_t* zero_out,
    int num_heads,
    int head_dim,
    int chunk_size)
{
    quantize_v_block_impl<uint16_t>(V, packed, scale_out, zero_out, num_heads, head_dim, chunk_size);
}