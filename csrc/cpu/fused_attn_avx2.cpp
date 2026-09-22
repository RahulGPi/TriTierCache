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

// 8 independent vector accumulators in flight to saturate FMA pipeline
static inline float dot_avx2(const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), acc4);
        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), acc5);
        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), acc6);
        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), acc7);
    }
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc4 = _mm256_add_ps(acc4, acc5);
    acc6 = _mm256_add_ps(acc6, acc7);
    acc0 = _mm256_add_ps(acc0, acc2);
    acc4 = _mm256_add_ps(acc4, acc6);
    acc0 = _mm256_add_ps(acc0, acc4);

    float sum = hsum8(acc0);
    for (; i < n; ++i) sum += a[i] * b[i]; // tail, only if head_dim % 8 != 0
    return sum;
}

// out[0..n) += scale * x[0..n) with 8 unrolled vectors
static inline void axpy_avx2(float* out, const float* x, float scale, int n) {
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m256 vo0 = _mm256_loadu_ps(out + i);
        __m256 vx0 = _mm256_loadu_ps(x + i);
        vo0 = _mm256_fmadd_ps(vx0, vscale, vo0);
        _mm256_storeu_ps(out + i, vo0);

        __m256 vo1 = _mm256_loadu_ps(out + i + 8);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        vo1 = _mm256_fmadd_ps(vx1, vscale, vo1);
        _mm256_storeu_ps(out + i + 8, vo1);

        __m256 vo2 = _mm256_loadu_ps(out + i + 16);
        __m256 vx2 = _mm256_loadu_ps(x + i + 16);
        vo2 = _mm256_fmadd_ps(vx2, vscale, vo2);
        _mm256_storeu_ps(out + i + 16, vo2);

        __m256 vo3 = _mm256_loadu_ps(out + i + 24);
        __m256 vx3 = _mm256_loadu_ps(x + i + 24);
        vo3 = _mm256_fmadd_ps(vx3, vscale, vo3);
        _mm256_storeu_ps(out + i + 24, vo3);

        __m256 vo4 = _mm256_loadu_ps(out + i + 32);
        __m256 vx4 = _mm256_loadu_ps(x + i + 32);
        vo4 = _mm256_fmadd_ps(vx4, vscale, vo4);
        _mm256_storeu_ps(out + i + 32, vo4);

        __m256 vo5 = _mm256_loadu_ps(out + i + 40);
        __m256 vx5 = _mm256_loadu_ps(x + i + 40);
        vo5 = _mm256_fmadd_ps(vx5, vscale, vo5);
        _mm256_storeu_ps(out + i + 40, vo5);

        __m256 vo6 = _mm256_loadu_ps(out + i + 48);
        __m256 vx6 = _mm256_loadu_ps(x + i + 48);
        vo6 = _mm256_fmadd_ps(vx6, vscale, vo6);
        _mm256_storeu_ps(out + i + 48, vo6);

        __m256 vo7 = _mm256_loadu_ps(out + i + 56);
        __m256 vx7 = _mm256_loadu_ps(x + i + 56);
        vo7 = _mm256_fmadd_ps(vx7, vscale, vo7);
        _mm256_storeu_ps(out + i + 56, vo7);
    }
    for (; i + 32 <= n; i += 32) {
        __m256 vo0 = _mm256_loadu_ps(out + i);
        __m256 vx0 = _mm256_loadu_ps(x + i);
        vo0 = _mm256_fmadd_ps(vx0, vscale, vo0);
        _mm256_storeu_ps(out + i, vo0);

        __m256 vo1 = _mm256_loadu_ps(out + i + 8);
        __m256 vx1 = _mm256_loadu_ps(x + i + 8);
        vo1 = _mm256_fmadd_ps(vx1, vscale, vo1);
        _mm256_storeu_ps(out + i + 8, vo1);

        __m256 vo2 = _mm256_loadu_ps(out + i + 16);
        __m256 vx2 = _mm256_loadu_ps(x + i + 16);
        vo2 = _mm256_fmadd_ps(vx2, vscale, vo2);
        _mm256_storeu_ps(out + i + 16, vo2);

        __m256 vo3 = _mm256_loadu_ps(out + i + 24);
        __m256 vx3 = _mm256_loadu_ps(x + i + 24);
        vo3 = _mm256_fmadd_ps(vx3, vscale, vo3);
        _mm256_storeu_ps(out + i + 24, vo3);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 vo = _mm256_loadu_ps(out + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        vo = _mm256_fmadd_ps(vx, vscale, vo);
        _mm256_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += scale * x[i];
}

// Reusable thread-local storage to avoid repeated heap allocation
static thread_local std::vector<float> tls_scores;
static thread_local std::vector<float> tls_pbs_k_dequant;
static thread_local std::vector<float> tls_pbs_v_dequant; 
static thread_local std::vector<float> tls_row_max;
static thread_local std::vector<float> tls_row_sum_exp;

template <typename MetaT>
static void fused_attention_decode_impl(
    const float* Q,
    const float* dense_K,
    const float* dense_V,
    int dense_count,
    const int32_t* PBS_K_Packed,
    const MetaT* PBS_K_Scales,
    const MetaT* PBS_K_Zeroes,
    const int32_t* PBS_V_Packed,
    const MetaT* PBS_V_Scales,
    const MetaT* PBS_V_Zeroes,
    const int64_t* PBS_token_ids,
    int num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float* attn_output,
    float* mean_attn_weights,
    int k_group_size)
{
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int total_pbs     = num_blocks * k_group_size;
    const int total_tokens  = dense_count + total_pbs;
    const float NEG_INF     = -std::numeric_limits<float>::infinity();
    const int gqa_ratio     = (num_kv_heads > 0) ? (num_q_heads / num_kv_heads) : 1;
    const int words_per_k_block = k_group_size / 16;

    // Allocate score matrix: [num_q_heads, total_tokens]
    size_t req_scores = static_cast<size_t>(num_q_heads) * total_tokens;
    if (tls_scores.size() < req_scores) {
        tls_scores.resize(req_scores);
    }
    float* scores = tls_scores.data();

    // Dequantize PBS K and V blocks in parallel if num_blocks > 0
    float* pbs_k_dequant_ptr = nullptr;
    float* pbs_v_dequant_ptr = nullptr;

    if (num_blocks > 0) {
        size_t req_pbs_elems = static_cast<size_t>(total_pbs) * num_kv_heads * head_dim;
        if (tls_pbs_k_dequant.size() < req_pbs_elems) {
            tls_pbs_k_dequant.resize(req_pbs_elems);
        }
        if (tls_pbs_v_dequant.size() < req_pbs_elems) {
            tls_pbs_v_dequant.resize(req_pbs_elems);
        }
        pbs_k_dequant_ptr = tls_pbs_k_dequant.data();
        pbs_v_dequant_ptr = tls_pbs_v_dequant.data();

        #pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < num_blocks; ++b) {
            dequantize_k_avx2(
                PBS_K_Packed + static_cast<size_t>(b) * words_per_k_block * num_kv_heads * head_dim,
                PBS_K_Scales + static_cast<size_t>(b) * num_kv_heads * head_dim,
                PBS_K_Zeroes + static_cast<size_t>(b) * num_kv_heads * head_dim,
                pbs_k_dequant_ptr + static_cast<size_t>(b) * k_group_size * num_kv_heads * head_dim,
                /*num_blocks=*/1, num_kv_heads, head_dim, k_group_size);

            dequantize_v_avx2(
                PBS_V_Packed + static_cast<size_t>(b) * k_group_size * num_kv_heads * ((head_dim + 15) / 16),
                PBS_V_Scales + static_cast<size_t>(b) * k_group_size * num_kv_heads,
                PBS_V_Zeroes + static_cast<size_t>(b) * k_group_size * num_kv_heads,
                pbs_v_dequant_ptr + static_cast<size_t>(b) * k_group_size * num_kv_heads * head_dim,
                /*total_tokens=*/k_group_size, num_kv_heads, head_dim);
        }
    }

    // Softmax stats per query head
    if (tls_row_max.size() < static_cast<size_t>(num_q_heads)) {
        tls_row_max.resize(num_q_heads);
        tls_row_sum_exp.resize(num_q_heads);
    }
    float* row_max = tls_row_max.data();
    float* row_sum_exp = tls_row_sum_exp.data();

    // ---------------- Parallel execution across Query Heads ----------------
    #pragma omp parallel for schedule(dynamic, 1)
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
            for (int t = 0; t < k_group_size; ++t) {
                int global_idx = dense_count + b * k_group_size + t;
                if (PBS_token_ids[b * k_group_size + t] == -1) {
                    head_scores[global_idx] = NEG_INF;
                } else {
                    const float* k = pbs_k_dequant_ptr + (static_cast<size_t>(b * k_group_size + t) * num_kv_heads + kv_h) * head_dim;
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
            for (int t = 0; t < k_group_size; ++t) {
                if (PBS_token_ids[b * k_group_size + t] == -1) continue;
                int global_idx = dense_count + b * k_group_size + t;
                float w = std::exp(head_scores[global_idx] - mx) * inv_sum;
                const float* v = pbs_v_dequant_ptr + (static_cast<size_t>(b * k_group_size + t) * num_kv_heads + kv_h) * head_dim;
                axpy_avx2(out_h, v, w, head_dim);
            }
        }
    }

    // Mean attention weights across heads for global scoring feedback
    if (mean_attn_weights) {
        #pragma omp parallel for schedule(dynamic, 1)
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
    float* mean_attn_weights,
    int k_group_size)
{
    fused_attention_decode_impl<float>(
        Q, dense_K, dense_V, dense_count,
        PBS_K_Packed, PBS_K_Scales, PBS_K_Zeroes,
        PBS_V_Packed, PBS_V_Scales, PBS_V_Zeroes,
        PBS_token_ids, num_blocks, num_q_heads, num_kv_heads, head_dim,
        attn_output, mean_attn_weights, k_group_size);
}

void fused_attention_decode_avx2(
    const float* Q,
    const float* dense_K,
    const float* dense_V,
    int dense_count,
    const int32_t* PBS_K_Packed,
    const uint16_t* PBS_K_Scales,
    const uint16_t* PBS_K_Zeroes,
    const int32_t* PBS_V_Packed,
    const uint16_t* PBS_V_Scales,
    const uint16_t* PBS_V_Zeroes,
    const int64_t* PBS_token_ids,
    int num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float* attn_output,
    float* mean_attn_weights,
    int k_group_size)
{
    fused_attention_decode_impl<uint16_t>(
        Q, dense_K, dense_V, dense_count,
        PBS_K_Packed, PBS_K_Scales, PBS_K_Zeroes,
        PBS_V_Packed, PBS_V_Scales, PBS_V_Zeroes,
        PBS_token_ids, num_blocks, num_q_heads, num_kv_heads, head_dim,
        attn_output, mean_attn_weights, k_group_size);
}