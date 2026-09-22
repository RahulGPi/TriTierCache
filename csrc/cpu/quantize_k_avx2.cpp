#include "../include/quantize_k_avx2.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Storage helpers for float (FP32) vs uint16_t (FP16 via F16C)
static inline void store_meta_vec(float* dest, __m256 v) {
    _mm256_storeu_ps(dest, v);
}

static inline void store_meta_vec(uint16_t* dest, __m256 v) {
    __m128i h = _mm256_cvtps_ph(v, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dest), h);
}

static inline void store_meta_scalar(float* dest, float val) {
    *dest = val;
}

static inline void store_meta_scalar(uint16_t* dest, float val) {
    __m128 v = _mm_set_ss(val);
    *dest = static_cast<uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0)));
}

template <typename MetaT>
static void quantize_k_block_impl(
    const float* K,
    uint32_t* packed,
    MetaT* scale_out,
    MetaT* zero_out,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    const int row_stride = num_heads * head_dim;
    const int vec_width = 8;
    const int c_vec_end = (head_dim / vec_width) * vec_width;
    const int num_words = k_group_size / 16;

    for (int h = 0; h < num_heads; ++h) {
        const float* head_index = K + h * head_dim;

        for (int c = 0; c < c_vec_end; c += vec_width) {
            // Find min and max across all k_group_size tokens
            __m256 vmin = _mm256_set1_ps(INFINITY);
            __m256 vmax = _mm256_set1_ps(-INFINITY);
            for (int t = 0; t < k_group_size; ++t) {
                __m256 x = _mm256_loadu_ps(head_index + t * row_stride + c);
                vmin = _mm256_min_ps(vmin, x);
                vmax = _mm256_max_ps(vmax, x);
            }

            __m256 diff = _mm256_sub_ps(vmax, vmin);
            __m256 vscale = _mm256_div_ps(diff, _mm256_set1_ps(3.0f));
            vscale = _mm256_max_ps(vscale, _mm256_set1_ps(1e-9f));

            store_meta_vec(scale_out + h * head_dim + c, vscale);
            store_meta_vec(zero_out + h * head_dim + c, vmin);

            // Quantize & pack into 16-token words
            __m256i vzero_i = _mm256_setzero_si256();
            __m256i vthree = _mm256_set1_epi32(3);

            for (int w = 0; w < num_words; ++w) {
                __m256i vpacked = _mm256_setzero_si256();
                for (int sub_t = 0; sub_t < 16; ++sub_t) {
                    int t = w * 16 + sub_t;
                    __m256 x = _mm256_loadu_ps(head_index + t * row_stride + c);
                    __m256 normalised = _mm256_div_ps(_mm256_sub_ps(x, vmin), vscale);
                    __m256i q = _mm256_cvtps_epi32(normalised);
                    q = _mm256_max_epi32(q, vzero_i);
                    q = _mm256_min_epi32(q, vthree);
                    vpacked = _mm256_or_si256(vpacked, _mm256_slli_epi32(q, 2 * sub_t));
                }
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(packed + (w * num_heads + h) * head_dim + c),
                    vpacked
                );
            }
        }

        // Scalar tail
        for (int c = c_vec_end; c < head_dim; ++c) {
            float mn = INFINITY, mx = -INFINITY;
            for (int t = 0; t < k_group_size; ++t) {
                float x = head_index[t * row_stride + c];
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }

            float scale = std::max((mx - mn) / 3.0f, 1e-9f);
            store_meta_scalar(scale_out + h * head_dim + c, scale);
            store_meta_scalar(zero_out + h * head_dim + c, mn);

            for (int w = 0; w < num_words; ++w) {
                uint32_t bits = 0;
                for (int sub_t = 0; sub_t < 16; ++sub_t) {
                    int t = w * 16 + sub_t;
                    float x = head_index[t * row_stride + c];
                    float normalised = (x - mn) / scale;
                    int q = static_cast<int>(std::nearbyint(normalised));
                    q = std::max(0, std::min(3, q));
                    bits |= (static_cast<uint32_t>(q) << (2 * sub_t));
                }
                packed[(w * num_heads + h) * head_dim + c] = bits;
            }
        }
    }
}

void quantize_k_block_avx2(
    const float* K,
    uint32_t* packed,
    float* scale_out,
    float* zero_out,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    quantize_k_block_impl<float>(K, packed, scale_out, zero_out, num_heads, head_dim, k_group_size);
}

void quantize_k_block_avx2(
    const float* K,
    uint32_t* packed,
    uint16_t* scale_out,
    uint16_t* zero_out,
    int num_heads,
    int head_dim,
    int k_group_size)
{
    quantize_k_block_impl<uint16_t>(K, packed, scale_out, zero_out, num_heads, head_dim, k_group_size);
}