// csrc/cpu/fused_attn_avx2.cpp
#include "fused_attn_avx2.h"
#include "dequant_avx2.h"
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <limits>

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

static inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum8(acc);
    for (; i < n; ++i) sum += a[i] * b[i]; // tail, only if head_dim % 8 != 0
    return sum;
}

// out[0..n) += scale * x[0..n)
static inline void axpy_avx2(float* out, const float* x, float scale, int n) {
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        vo = _mm256_fmadd_ps(vx, vscale, vo);
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += scale * x[i];
}

void fused_attention_decode_avx2(
    const float* Q,
    const float* dense_K,
    const float* dense_V,
    int dense_count,
    const int32_t* PBS_K_Packed,
    const float* PBS_K_Scales,
    const float* PBS_K_Zeroes,
    const int32_t* PBS_V_Packed,
    const float* PBS_V_Scales,
    const float* PBS_V_Zeroes,
    const int64_t* PBS_token_ids,
    int num_blocks,
    int num_heads,
    int head_dim,
    float* attn_output)
{
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total_pbs     = num_blocks * 16;
    const int total_tokens  = dense_count + total_pbs;
    const float NEG_INF     = -std::numeric_limits<float>::infinity();

    // One score per (head, token) — NOT per (head, token, channel). This is
    // the whole reason this kernel is small: no full K or V ever exists.
    std::vector<float> scores(static_cast<size_t>(num_heads) * total_tokens);

    // Reused scratch for one dequantized PBS block: 16 tokens, all heads.
    std::vector<float> block_buf(16 * static_cast<size_t>(num_heads) * head_dim);

    // ---------------- Pass A: raw scores (pre-softmax) ----------------

    for (int i = 0; i < dense_count; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            const float* k = dense_K + (static_cast<size_t>(i) * num_heads + h) * head_dim;
            const float* q = Q + h * head_dim;
            scores[h * total_tokens + i] = dot_avx2(q, k, head_dim) * inv_sqrt_hd;
        }
    }

    for (int b = 0; b < num_blocks; ++b) {
        dequantize_k_avx2(
            PBS_K_Packed + static_cast<size_t>(b) * num_heads * head_dim,
            PBS_K_Scales + static_cast<size_t>(b) * num_heads * head_dim,
            PBS_K_Zeroes + static_cast<size_t>(b) * num_heads * head_dim,
            block_buf.data(),
            /*num_blocks=*/1, num_heads, head_dim);

        for (int t = 0; t < 16; ++t) {
            int global_idx = dense_count + b * 16 + t;
            bool valid = PBS_token_ids[b * 16 + t] != -1;
            for (int h = 0; h < num_heads; ++h) {
                if (!valid) {
                    scores[h * total_tokens + global_idx] = NEG_INF;
                    continue;
                }
                const float* k = block_buf.data() + (static_cast<size_t>(t) * num_heads + h) * head_dim;
                const float* q = Q + h * head_dim;
                scores[h * total_tokens + global_idx] = dot_avx2(q, k, head_dim) * inv_sqrt_hd;
            }
        }
    }

    // ---------------- Softmax stats per head ----------------
    std::vector<float> row_max(num_heads, NEG_INF);
    std::vector<float> row_sum_exp(num_heads, 0.0f);

    for (int h = 0; h < num_heads; ++h) {
        const float* row = &scores[static_cast<size_t>(h) * total_tokens];
        float mx = NEG_INF;
        for (int i = 0; i < total_tokens; ++i) mx = std::max(mx, row[i]);
        row_max[h] = mx;

        float sum = 0.0f;
        for (int i = 0; i < total_tokens; ++i) sum += std::exp(row[i] - mx); // exp(-inf)=0
        row_sum_exp[h] = sum;
    }

    // ---------------- Pass B: weighted sum of V ----------------
    for (size_t i = 0; i < static_cast<size_t>(num_heads) * head_dim; ++i) attn_output[i] = 0.0f;

    for (int i = 0; i < dense_count; ++i) {
        for (int h = 0; h < num_heads; ++h) {
            float w = std::exp(scores[h * total_tokens + i] - row_max[h]);
            const float* v = dense_V + (static_cast<size_t>(i) * num_heads + h) * head_dim;
            axpy_avx2(attn_output + h * head_dim, v, w, head_dim);
        }
    }

    for (int b = 0; b < num_blocks; ++b) {
        dequantize_v_avx2(
            PBS_V_Packed + static_cast<size_t>(b) * 16 * num_heads * ((head_dim + 15) / 16),
            PBS_V_Scales + static_cast<size_t>(b) * 16 * num_heads,
            PBS_V_Zeroes + static_cast<size_t>(b) * 16 * num_heads,
            block_buf.data(),
            /*total_tokens=*/16, num_heads, head_dim);

        for (int t = 0; t < 16; ++t) {
            if (PBS_token_ids[b * 16 + t] == -1) continue;
            int global_idx = dense_count + b * 16 + t;
            for (int h = 0; h < num_heads; ++h) {
                float w = std::exp(scores[h * total_tokens + global_idx] - row_max[h]);
                const float* v = block_buf.data() + (static_cast<size_t>(t) * num_heads + h) * head_dim;
                axpy_avx2(attn_output + h * head_dim, v, w, head_dim);
            }
        }
    }

    for (int h = 0; h < num_heads; ++h) {
        float inv_sum = 1.0f / row_sum_exp[h];
        for (int c = 0; c < head_dim; ++c) attn_output[h * head_dim + c] *= inv_sum;
    }
}