// csrc/cpu/fused_attn_avx2.cpp
#include "fused_attn_avx2.h"
#include "dequant_avx2.h"
#include <immintrin.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

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
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float* attn_output,
    float* mean_attn_weights)
{
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total_pbs     = num_blocks * 16;
    const int total_tokens  = dense_count + total_pbs;
    const float NEG_INF     = -std::numeric_limits<float>::infinity();
    const int gqa_ratio     = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;

    // Allocate score matrix: [num_q_heads, total_tokens]
    std::vector<float> scores(static_cast<size_t>(num_q_heads) * total_tokens);

    // Dequantize PBS K and V blocks in parallel if num_blocks > 0
    std::vector<float> pbs_k_dequant;
    std::vector<float> pbs_v_dequant;

    if (num_blocks > 0) {
        pbs_k_dequant.resize(static_cast<size_t>(total_pbs) * num_kv_heads * head_dim);
        pbs_v_dequant.resize(static_cast<size_t>(total_pbs) * num_kv_heads * head_dim);

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < num_blocks; ++b) {
            dequantize_k_avx2(
                PBS_K_Packed + static_cast<size_t>(b) * num_kv_heads * head_dim,
                PBS_K_Scales + static_cast<size_t>(b) * num_kv_heads * head_dim,
                PBS_K_Zeroes + static_cast<size_t>(b) * num_kv_heads * head_dim,
                pbs_k_dequant.data() + static_cast<size_t>(b) * 16 * num_kv_heads * head_dim,
                /*num_blocks=*/1, num_kv_heads, head_dim);

            dequantize_v_avx2(
                PBS_V_Packed + static_cast<size_t>(b) * 16 * num_kv_heads * ((head_dim + 15) / 16),
                PBS_V_Scales + static_cast<size_t>(b) * 16 * num_kv_heads,
                PBS_V_Zeroes + static_cast<size_t>(b) * 16 * num_kv_heads,
                pbs_v_dequant.data() + static_cast<size_t>(b) * 16 * num_kv_heads * head_dim,
                /*total_tokens=*/16, num_kv_heads, head_dim);
        }
    }

    // Softmax stats per query head
    std::vector<float> row_max(num_q_heads, NEG_INF);
    std::vector<float> row_sum_exp(num_q_heads, 0.0f);

    // ---------------- Parallel execution across Query Heads ----------------
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < num_q_heads; ++h) {
        int kv_h = (gqa_ratio > 1) ? (h / gqa_ratio) : h;
        const float* q = Q + h * head_dim;
        float* head_scores = &scores[static_cast<size_t>(h) * total_tokens];

        // Pass A1: Dense tokens (Sinks, Heavy Hitters, Recent Window)
        for (int i = 0; i < dense_count; ++i) {
            const float* k = dense_K + (static_cast<size_t>(i) * num_kv_heads + kv_h) * head_dim;
            head_scores[i] = dot_avx2(q, k, head_dim) * inv_sqrt_hd;
        }

        // Pass A2: PBS background tokens
        for (int b = 0; b < num_blocks; ++b) {
            for (int t = 0; t < 16; ++t) {
                int global_idx = dense_count + b * 16 + t;
                if (PBS_token_ids[b * 16 + t] == -1) {
                    head_scores[global_idx] = NEG_INF;
                } else {
                    const float* k = pbs_k_dequant.data() + (static_cast<size_t>(b * 16 + t) * num_kv_heads + kv_h) * head_dim;
                    head_scores[global_idx] = dot_avx2(q, k, head_dim) * inv_sqrt_hd;
                }
            }
        }

        // Softmax reduction for head h
        float mx = NEG_INF;
        for (int i = 0; i < total_tokens; ++i) {
            mx = std::max(mx, head_scores[i]);
        }
        row_max[h] = mx;

        float sum_exp = 0.0f;
        for (int i = 0; i < total_tokens; ++i) {
            if (head_scores[i] != NEG_INF) {
                sum_exp += std::exp(head_scores[i] - mx);
            }
        }
        float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 1.0f;
        row_sum_exp[h] = sum_exp > 0.0f ? sum_exp : 1.0f;

        // Pass B: Weighted sum of V for head h
        float* out_h = attn_output + h * head_dim;
        for (int c = 0; c < head_dim; ++c) out_h[c] = 0.0f;

        // Dense V accumulation
        for (int i = 0; i < dense_count; ++i) {
            float w = std::exp(head_scores[i] - mx) * inv_sum;
            const float* v = dense_V + (static_cast<size_t>(i) * num_kv_heads + kv_h) * head_dim;
            axpy_avx2(out_h, v, w, head_dim);
        }

        // PBS V accumulation
        for (int b = 0; b < num_blocks; ++b) {
            for (int t = 0; t < 16; ++t) {
                if (PBS_token_ids[b * 16 + t] == -1) continue;
                int global_idx = dense_count + b * 16 + t;
                float w = std::exp(head_scores[global_idx] - mx) * inv_sum;
                const float* v = pbs_v_dequant.data() + (static_cast<size_t>(b * 16 + t) * num_kv_heads + kv_h) * head_dim;
                axpy_avx2(out_h, v, w, head_dim);
            }
        }
    }

    // Mean attention weights across heads for global scoring feedback
    if (mean_attn_weights) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total_tokens; ++i) {
            float sum_w = 0.0f;
            for (int h = 0; h < num_q_heads; ++h) {
                float sc = scores[static_cast<size_t>(h) * total_tokens + i];
                if (sc != NEG_INF) {
                    float w = std::exp(sc - row_max[h]) / row_sum_exp[h];
                    sum_w += w;
                }
            }
            mean_attn_weights[i] = sum_w / static_cast<float>(num_q_heads);
        }
    }
}