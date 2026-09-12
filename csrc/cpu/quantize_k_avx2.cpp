#include "../include/quantize_k_avx2.h"
#include <immintrin.h>
#include <cstdint>
#include <cmath>


//Processes completed WR 16-token block

void quantize_k_block_avx2(
    const float* K,
    uint32_t* packed,
    float* scale_out,
    float* zero_out,
    int num_heads,
    int head_dim)
{
    const int row_stride = num_heads * head_dim;
    const int vec_width = 8;
    const int c_vec_end = (head_dim / vec_width) * vec_width;

    for(int h = 0; h < num_heads; ++h){
        const float* head_index = K + h * head_dim;

        for(int c = 0; c < c_vec_end; c += vec_width)
        {
            //min & max
            __m256 vmin = _mm256_set1_ps(INFINITY);
            __m256 vmax = _mm256_set1_ps(-INFINITY);
            for(int t = 0; t < 16; ++t)
            {
                __m256 x = _mm256_loadu_ps(head_index +t * row_stride + c);
                vmin = _mm256_min_ps(vmin, x);
                vmax = _mm256_max_ps(vmax, x);
            }

            __m256 diff = _mm256_sub_ps(vmax, vmin);
            __m256 vscale = _mm256_div_ps(diff, _mm256_set1_ps(3.0f));
            vscale = _mm256_max_ps(vscale, _mm256_set1_ps(1e-9f));

            _mm256_storeu_ps(scale_out + h * head_dim + c, vscale);
            _mm256_storeu_ps(zero_out + h * head_dim + c, vmin);

            //quantise & pack
            __m256i vpacked = _mm256_setzero_si256();
            __m256i vzero_i = _mm256_setzero_si256();
            __m256i vthree = _mm256_set1_epi32(3);

            for(int t = 0; t < 16; ++t)
            {
                __m256 x = _mm256_loadu_ps(head_index +t * row_stride + c);
                __m256 normalised = _mm256_div_ps(_mm256_sub_ps(x, vmin), vscale);

                __m256i q = _mm256_cvtps_epi32(normalised);

                q = _mm256_max_epi32(q, vzero_i);
                q = _mm256_min_epi32(q, vthree);

                vpacked = _mm256_or_si256(vpacked, _mm256_slli_epi32(q, 2*t));
            }

            _mm256_storeu_si256((__m256i*)(packed + h * head_dim + c), vpacked);
        }

        for(int c = c_vec_end; c < head_dim; ++c)
        {
            float mn = INFINITY, mx = -INFINITY; // min & max for scalar tail kept as mn and mx not to conflict with min & max fns
            for (int t = 0; t < 16; ++t)
            {
                float x = head_index[t * row_stride + c];
                mn = std::min(mn, x);
                mx = std::max(mx, x);
            }

            float scale = std::max((mx - mn) / 3.0f, 1e-9f);
            scale_out[h * head_dim + c] = scale;
            zero_out[h * head_dim + c] = mn;

            uint32_t bits = 0;
            for(int t = 0; t < 16; ++t)
            {
                float x = head_index[t * row_stride + c];
                float normalised = (x - mn) / scale;
                int q = static_cast<int>(std::nearbyint(normalised));

                q = std::max(0, std::min(3, q));
                bits |= (static_cast<uint32_t>(q) << (2 * t));
            }
            packed[h * head_dim + c] = bits;
        }


    }
}