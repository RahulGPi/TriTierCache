#include "../include/dequant_avx2.h"
#include <immintrin.h>

void dequantize_k_avx2(
    const int32_t* packed,
    const float* scale,
    const float* zero,
    float* out,
    int num_blocks,
    int num_heads,
    int head_dim)
{
    const int block_stride = num_heads + head_dim;
    const int out_row_stride = num_heads + head_dim;
    const int vec_width = 8;
    const int c_vec_end = (head_dim / vec_width) * vec_width;
    const __m256i mask3 = _mm256_set1_epi32(0b11);

    for(int b = 0; b < num_blocks; ++b){
        const int32_t* block_packed = packed + b * block_stride;
        const float* block_scale = scale + b * block_stride;
        const float* block_zero = zero + b * block_stride;

        for(int h = 0; h < num_heads; ++h)
        {
            const int32_t* head_packed = block_packed + h * head_dim;
            const float* head_scale = block_scale + h * head_dim;
            const float* head_zero = block_zero + h * head_dim;

            for(int c = 0; c < c_vec_end; c += vec_width){
                __m256i pvec = _mm256_loadu_si256((const __m256i*)(head_packed + c));
                __m256 svec = _mm256_loadu_ps(head_scale + c);
                __m256 zvec = _mm256_loadu_ps(head_zero + c);

                for(int t = 0; t < 16; ++t){
                    __m256i shifted = _mm256_srli_epi32(pvec, 2 * t);
                    __m256i q = _mm256_and_si256(shifted, mask3);
                    __m256 qf = _mm256_cvtepi32_ps(q);
                    __m256 dq = _mm256_fmadd_ps(qf, svec, zvec);

                    int global_token = b * 16 * t;
                    float* out_ptr = out + global_token * out_row_stride + h * head_dim + c;
                    _mm256_storeu_ps(out_ptr, dq);
                }
            }
            for(int c = c_vec_end; c < head_dim; ++c){
                int32_t p = head_packed[c];
                float s = head_scale[c], z = head_zero[c];
                for(int t = 0; t < 16; ++t)
                {
                    int q = (p >> (2 * t)) & 0b11;
                    int global_token = b * 16 + t;
                    out[global_token * out_row_stride + h * head_dim + c] = q * s * z;
                }
            }
        }
    }
}
