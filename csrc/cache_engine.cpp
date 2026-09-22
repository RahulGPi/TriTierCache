#include "cache_engine.h"
#include "fused_attn_avx2.h"
#include "quantize_k_avx2.h"
#include "quantize_v_avx2.h"
#include <immintrin.h>
#include <iostream>

TriTierCacheEngine::TriTierCacheEngine(
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int sink_size,
    int rw_size,
    float h_ratio,
    float score_decay,
    int update_interval,
    int k_group_size,
    const std::string& pbs_metadata_dtype
) : num_q_heads(num_q_heads),
    num_kv_heads(num_kv_heads),
    head_dim(head_dim),
    max_seq_len(max_seq_len),
    sink_size(sink_size),
    rw_size(rw_size),
    h_ratio(h_ratio),
    score_decay(score_decay),
    update_interval(update_interval),
    k_group_size((k_group_size == 32) ? 32 : 16),
    pbs_metadata_dtype(pbs_metadata_dtype),
    use_fp16_meta(pbs_metadata_dtype == "fp16"),
    s_count(0),
    rw_count(0),
    rw_head_index(0),
    hh_count(0),
    wr_count(0),
    pbs_allocated_blocks(0),
    pbs_blocks_used(0),
    pbs_count(0),
    total_processed_tokens(0),
    cached_threshold(0.0f),
    last_calc_pos(-1),
    total_evictions(0)
{
    chunk_size = this->k_group_size;
    quant_head_dim = (head_dim + 15) / 16;
    max_hh = static_cast<int>(std::ceil(max_seq_len * h_ratio));
    if (max_hh < 1) max_hh = 1;

    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;

    // Sinks
    S_K.resize(static_cast<size_t>(sink_size) * kv_stride, 0.0f);
    S_V.resize(static_cast<size_t>(sink_size) * kv_stride, 0.0f);

    // Recent Window
    RW_K.resize(static_cast<size_t>(rw_size) * kv_stride, 0.0f);
    RW_V.resize(static_cast<size_t>(rw_size) * kv_stride, 0.0f);

    // Heavy Hitters
    HH_K.resize(static_cast<size_t>(max_hh) * kv_stride, 0.0f);
    HH_V.resize(static_cast<size_t>(max_hh) * kv_stride, 0.0f);
    HH_token_ids.assign(max_hh, -1);
    HH_scores.assign(max_hh, 0.0f);

    // Waiting Room
    WR_K.resize(static_cast<size_t>(chunk_size) * kv_stride, 0.0f);
    WR_V.resize(static_cast<size_t>(chunk_size) * kv_stride, 0.0f);
    WR_token_ids.assign(chunk_size, -1);

    // Initial PBS allocation: chunked lazy growth (start with 0 blocks, grow dynamically)
    pbs_allocated_blocks = 0;

    // Global attention scores
    global_attn_scores.assign(max_seq_len, 0.0f);

    // Scratch buffers
    size_t max_dense = static_cast<size_t>(sink_size + max_hh + rw_size);
    dense_K.resize(max_dense * kv_stride, 0.0f);
    dense_V.resize(max_dense * kv_stride, 0.0f);
    mean_attn_weights.resize(max_dense + 1024, 0.0f);
}

void TriTierCacheEngine::ensure_pbs_capacity(int min_blocks) {
    if (min_blocks <= pbs_allocated_blocks) return;

    // Grow in chunks of 64 blocks
    int new_blocks = std::max(pbs_allocated_blocks + 64, min_blocks);
    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;
    int words_per_k_block = k_group_size / 16;

    PBS_K_Packed.resize(static_cast<size_t>(new_blocks) * words_per_k_block * kv_stride, 0);

    if (use_fp16_meta) {
        PBS_K_Scales_fp16.resize(static_cast<size_t>(new_blocks) * kv_stride, 0);
        PBS_K_Zeroes_fp16.resize(static_cast<size_t>(new_blocks) * kv_stride, 0);
        PBS_V_Scales_fp16.resize(static_cast<size_t>(new_blocks) * chunk_size * num_kv_heads, 0);
        PBS_V_Zeroes_fp16.resize(static_cast<size_t>(new_blocks) * chunk_size * num_kv_heads, 0);
    } else {
        PBS_K_Scales_fp32.resize(static_cast<size_t>(new_blocks) * kv_stride, 0.0f);
        PBS_K_Zeroes_fp32.resize(static_cast<size_t>(new_blocks) * kv_stride, 0.0f);
        PBS_V_Scales_fp32.resize(static_cast<size_t>(new_blocks) * chunk_size * num_kv_heads, 0.0f);
        PBS_V_Zeroes_fp32.resize(static_cast<size_t>(new_blocks) * chunk_size * num_kv_heads, 0.0f);
    }

    size_t v_packed_stride = static_cast<size_t>(num_kv_heads) * quant_head_dim;
    PBS_V_Packed.resize(static_cast<size_t>(new_blocks) * chunk_size * v_packed_stride, 0);

    size_t prev_tok_size = PBS_token_ids.size();
    PBS_token_ids.resize(static_cast<size_t>(new_blocks) * chunk_size, -1);
    for (size_t i = prev_tok_size; i < PBS_token_ids.size(); ++i) {
        PBS_token_ids[i] = -1;
    }

    pbs_allocated_blocks = new_blocks;
}

float TriTierCacheEngine::fetch_or_calc_threshold() {
    int cand_start = sink_size;
    int cand_end = total_processed_tokens - rw_size;
    if (cand_end <= cand_start) {
        cached_threshold = 0.0f;
        return cached_threshold;
    }

    if (last_calc_pos == -1 || (total_processed_tokens - last_calc_pos >= update_interval)) {
        std::vector<float> valid_scores;
        valid_scores.reserve(cand_end - cand_start);
        for (int i = cand_start; i < cand_end; ++i) {
            valid_scores.push_back(global_attn_scores[i]);
        }
        if (valid_scores.empty()) {
            cached_threshold = 0.0f;
        } else {
            size_t n = valid_scores.size();
            float q = 1.0f - h_ratio;
            if (q < 0.0f) q = 0.0f;
            if (q > 1.0f) q = 1.0f;
            size_t k = static_cast<size_t>(q * (n - 1));
            std::nth_element(valid_scores.begin(), valid_scores.begin() + k, valid_scores.end());
            cached_threshold = valid_scores[k];
        }
        last_calc_pos = total_processed_tokens;
    }
    return cached_threshold;
}

void TriTierCacheEngine::flush_waiting_room_to_pbs() {
    ensure_pbs_capacity(pbs_blocks_used + 1);
    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;
    int words_per_k_block = k_group_size / 16;
    size_t v_packed_stride = static_cast<size_t>(num_kv_heads) * quant_head_dim;

    uint32_t* k_packed_dest = reinterpret_cast<uint32_t*>(
        PBS_K_Packed.data() + static_cast<size_t>(pbs_blocks_used) * words_per_k_block * kv_stride
    );
    int32_t* v_packed_dest = PBS_V_Packed.data() + static_cast<size_t>(pbs_blocks_used) * chunk_size * v_packed_stride;

    if (use_fp16_meta) {
        quantize_k_block_avx2(
            WR_K.data(),
            k_packed_dest,
            PBS_K_Scales_fp16.data() + static_cast<size_t>(pbs_blocks_used) * kv_stride,
            PBS_K_Zeroes_fp16.data() + static_cast<size_t>(pbs_blocks_used) * kv_stride,
            num_kv_heads,
            head_dim,
            k_group_size
        );
        quantize_v_block_avx2(
            WR_V.data(),
            v_packed_dest,
            PBS_V_Scales_fp16.data() + static_cast<size_t>(pbs_blocks_used) * chunk_size * num_kv_heads,
            PBS_V_Zeroes_fp16.data() + static_cast<size_t>(pbs_blocks_used) * chunk_size * num_kv_heads,
            num_kv_heads,
            head_dim,
            chunk_size
        );
    } else {
        quantize_k_block_avx2(
            WR_K.data(),
            k_packed_dest,
            PBS_K_Scales_fp32.data() + static_cast<size_t>(pbs_blocks_used) * kv_stride,
            PBS_K_Zeroes_fp32.data() + static_cast<size_t>(pbs_blocks_used) * kv_stride,
            num_kv_heads,
            head_dim,
            k_group_size
        );
        quantize_v_block_avx2(
            WR_V.data(),
            v_packed_dest,
            PBS_V_Scales_fp32.data() + static_cast<size_t>(pbs_blocks_used) * chunk_size * num_kv_heads,
            PBS_V_Zeroes_fp32.data() + static_cast<size_t>(pbs_blocks_used) * chunk_size * num_kv_heads,
            num_kv_heads,
            head_dim,
            chunk_size
        );
    }

    // Copy token IDs
    for (int t = 0; t < chunk_size; ++t) {
        PBS_token_ids[static_cast<size_t>(pbs_blocks_used) * chunk_size + t] = WR_token_ids[t];
    }

    pbs_blocks_used++;
    pbs_count += chunk_size;
    wr_count = 0;
}

void TriTierCacheEngine::route_evicted_token(
    const float* k_ptr,
    const float* v_ptr,
    int64_t token_id
) {
    total_evictions++;
    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;
    float score = (token_id >= 0 && token_id < static_cast<int64_t>(global_attn_scores.size())) ?
                  global_attn_scores[token_id] : 0.0f;
    float thresh = fetch_or_calc_threshold();

    if (score >= thresh) {
        if (hh_count < max_hh) {
            std::memcpy(HH_K.data() + static_cast<size_t>(hh_count) * kv_stride, k_ptr, kv_stride * sizeof(float));
            std::memcpy(HH_V.data() + static_cast<size_t>(hh_count) * kv_stride, v_ptr, kv_stride * sizeof(float));
            HH_token_ids[hh_count] = token_id;
            HH_scores[hh_count] = score;
            hh_count++;
            return;
        }

        // Find minimum score in HH
        int min_idx = 0;
        float min_score = HH_scores[0];
        for (int i = 1; i < hh_count; ++i) {
            if (HH_scores[i] < min_score) {
                min_score = HH_scores[i];
                min_idx = i;
            }
        }
        if (score > min_score) {
            // Demote existing HH[min_idx] to Waiting Room
            std::memcpy(WR_K.data() + static_cast<size_t>(wr_count) * kv_stride,
                        HH_K.data() + static_cast<size_t>(min_idx) * kv_stride, kv_stride * sizeof(float));
            std::memcpy(WR_V.data() + static_cast<size_t>(wr_count) * kv_stride,
                        HH_V.data() + static_cast<size_t>(min_idx) * kv_stride, kv_stride * sizeof(float));
            WR_token_ids[wr_count] = HH_token_ids[min_idx];
            wr_count++;

            // Replace HH[min_idx] with newly evicted token
            std::memcpy(HH_K.data() + static_cast<size_t>(min_idx) * kv_stride, k_ptr, kv_stride * sizeof(float));
            std::memcpy(HH_V.data() + static_cast<size_t>(min_idx) * kv_stride, v_ptr, kv_stride * sizeof(float));
            HH_token_ids[min_idx] = token_id;
            HH_scores[min_idx] = score;

            if (wr_count == chunk_size) {
                flush_waiting_room_to_pbs();
            }
            return;
        }
    }

    // Route evicted token directly to Waiting Room
    std::memcpy(WR_K.data() + static_cast<size_t>(wr_count) * kv_stride, k_ptr, kv_stride * sizeof(float));
    std::memcpy(WR_V.data() + static_cast<size_t>(wr_count) * kv_stride, v_ptr, kv_stride * sizeof(float));
    WR_token_ids[wr_count] = token_id;
    wr_count++;
    if (wr_count == chunk_size) {
        flush_waiting_room_to_pbs();
    }
}

void TriTierCacheEngine::prefill(
    const float* K_all,
    const float* V_all,
    int q_len
) {
    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;

    if (q_len <= sink_size + rw_size) {
        int s = std::min(sink_size, q_len);
        if (s > 0) {
            std::memcpy(S_K.data(), K_all, static_cast<size_t>(s) * kv_stride * sizeof(float));
            std::memcpy(S_V.data(), V_all, static_cast<size_t>(s) * kv_stride * sizeof(float));
            s_count = s;
        }
        int r = q_len - s;
        if (r > 0) {
            std::memcpy(RW_K.data(), K_all + static_cast<size_t>(s) * kv_stride, static_cast<size_t>(r) * kv_stride * sizeof(float));
            std::memcpy(RW_V.data(), V_all + static_cast<size_t>(s) * kv_stride, static_cast<size_t>(r) * kv_stride * sizeof(float));
            rw_count = r;
            rw_head_index = 0;
        }
        total_processed_tokens = q_len;
        return;
    }

    // 1. Sinks
    std::memcpy(S_K.data(), K_all, static_cast<size_t>(sink_size) * kv_stride * sizeof(float));
    std::memcpy(S_V.data(), V_all, static_cast<size_t>(sink_size) * kv_stride * sizeof(float));
    s_count = sink_size;

    // 2. Recent Window: last rw_size tokens
    int rw_start = q_len - rw_size;
    std::memcpy(RW_K.data(), K_all + static_cast<size_t>(rw_start) * kv_stride, static_cast<size_t>(rw_size) * kv_stride * sizeof(float));
    std::memcpy(RW_V.data(), V_all + static_cast<size_t>(rw_start) * kv_stride, static_cast<size_t>(rw_size) * kv_stride * sizeof(float));
    rw_count = rw_size;
    rw_head_index = 0;

    // 3. Intermediate tokens: [sink_size, rw_start)
    int n_inter = rw_start - sink_size;
    int num_blocks_to_quant = n_inter / chunk_size;
    int rem = n_inter % chunk_size;

    if (num_blocks_to_quant > 0) {
        ensure_pbs_capacity(pbs_blocks_used + num_blocks_to_quant);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int b = 0; b < num_blocks_to_quant; ++b) {
            int block_dest = pbs_blocks_used + b;
            int src_tok_idx = sink_size + b * chunk_size;
            const float* k_src = K_all + static_cast<size_t>(src_tok_idx) * kv_stride;
            const float* v_src = V_all + static_cast<size_t>(src_tok_idx) * kv_stride;
            int words_per_k_block = k_group_size / 16;
            size_t v_packed_stride = static_cast<size_t>(num_kv_heads) * quant_head_dim;

            uint32_t* k_packed_dest = reinterpret_cast<uint32_t*>(
                PBS_K_Packed.data() + static_cast<size_t>(block_dest) * words_per_k_block * kv_stride
            );
            int32_t* v_packed_dest = PBS_V_Packed.data() + static_cast<size_t>(block_dest) * chunk_size * v_packed_stride;

            if (use_fp16_meta) {
                quantize_k_block_avx2(
                    k_src,
                    k_packed_dest,
                    PBS_K_Scales_fp16.data() + static_cast<size_t>(block_dest) * kv_stride,
                    PBS_K_Zeroes_fp16.data() + static_cast<size_t>(block_dest) * kv_stride,
                    num_kv_heads,
                    head_dim,
                    k_group_size
                );
                quantize_v_block_avx2(
                    v_src,
                    v_packed_dest,
                    PBS_V_Scales_fp16.data() + static_cast<size_t>(block_dest) * chunk_size * num_kv_heads,
                    PBS_V_Zeroes_fp16.data() + static_cast<size_t>(block_dest) * chunk_size * num_kv_heads,
                    num_kv_heads,
                    head_dim,
                    chunk_size
                );
            } else {
                quantize_k_block_avx2(
                    k_src,
                    k_packed_dest,
                    PBS_K_Scales_fp32.data() + static_cast<size_t>(block_dest) * kv_stride,
                    PBS_K_Zeroes_fp32.data() + static_cast<size_t>(block_dest) * kv_stride,
                    num_kv_heads,
                    head_dim,
                    k_group_size
                );
                quantize_v_block_avx2(
                    v_src,
                    v_packed_dest,
                    PBS_V_Scales_fp32.data() + static_cast<size_t>(block_dest) * chunk_size * num_kv_heads,
                    PBS_V_Zeroes_fp32.data() + static_cast<size_t>(block_dest) * chunk_size * num_kv_heads,
                    num_kv_heads,
                    head_dim,
                    chunk_size
                );
            }

            for (int t = 0; t < chunk_size; ++t) {
                PBS_token_ids[static_cast<size_t>(block_dest) * chunk_size + t] = src_tok_idx + t;
            }
        }
        pbs_blocks_used += num_blocks_to_quant;
        pbs_count += num_blocks_to_quant * chunk_size;
        total_evictions += num_blocks_to_quant * chunk_size;
    }

    // Remaining intermediate tokens go to Waiting Room
    if (rem > 0) {
        int rem_start = sink_size + num_blocks_to_quant * chunk_size;
        const float* k_rem = K_all + static_cast<size_t>(rem_start) * kv_stride;
        const float* v_rem = V_all + static_cast<size_t>(rem_start) * kv_stride;
        std::memcpy(WR_K.data(), k_rem, static_cast<size_t>(rem) * kv_stride * sizeof(float));
        std::memcpy(WR_V.data(), v_rem, static_cast<size_t>(rem) * kv_stride * sizeof(float));
        for (int t = 0; t < rem; ++t) {
            WR_token_ids[t] = rem_start + t;
        }
        wr_count = rem;
    } else {
        wr_count = 0;
    }

    total_processed_tokens = q_len;
}

void TriTierCacheEngine::step(
    const float* Q,
    const float* K_new,
    const float* V_new,
    float* attn_output
) {
    size_t kv_stride = static_cast<size_t>(num_kv_heads) * head_dim;

    // 1. Ingest new token into Tier 1 (Sink / Recent Window)
    if (s_count < sink_size) {
        std::memcpy(S_K.data() + static_cast<size_t>(s_count) * kv_stride, K_new, kv_stride * sizeof(float));
        std::memcpy(S_V.data() + static_cast<size_t>(s_count) * kv_stride, V_new, kv_stride * sizeof(float));
        s_count++;
    } else if (rw_count < rw_size) {
        std::memcpy(RW_K.data() + static_cast<size_t>(rw_count) * kv_stride, K_new, kv_stride * sizeof(float));
        std::memcpy(RW_V.data() + static_cast<size_t>(rw_count) * kv_stride, V_new, kv_stride * sizeof(float));
        rw_count++;
    } else {
        // Ring buffer eviction
        int slot = rw_head_index;
        int64_t evicted_id = static_cast<int64_t>(total_processed_tokens) - rw_size;

        std::vector<float> ev_k(kv_stride);
        std::vector<float> ev_v(kv_stride);
        std::memcpy(ev_k.data(), RW_K.data() + static_cast<size_t>(slot) * kv_stride, kv_stride * sizeof(float));
        std::memcpy(ev_v.data(), RW_V.data() + static_cast<size_t>(slot) * kv_stride, kv_stride * sizeof(float));

        // Overwrite ring buffer slot with new token
        std::memcpy(RW_K.data() + static_cast<size_t>(slot) * kv_stride, K_new, kv_stride * sizeof(float));
        std::memcpy(RW_V.data() + static_cast<size_t>(slot) * kv_stride, V_new, kv_stride * sizeof(float));

        rw_head_index = (rw_head_index + 1) % rw_size;
        route_evicted_token(ev_k.data(), ev_v.data(), evicted_id);
    }
    total_processed_tokens++;

    // 2. Assemble Dense K and V for attention
    int dense_count = 0;
    // Sinks
    if (s_count > 0) {
        std::memcpy(dense_K.data() + static_cast<size_t>(dense_count) * kv_stride,
                    S_K.data(), static_cast<size_t>(s_count) * kv_stride * sizeof(float));
        std::memcpy(dense_V.data() + static_cast<size_t>(dense_count) * kv_stride,
                    S_V.data(), static_cast<size_t>(s_count) * kv_stride * sizeof(float));
        dense_count += s_count;
    }
    // Heavy Hitters
    if (hh_count > 0) {
        std::memcpy(dense_K.data() + static_cast<size_t>(dense_count) * kv_stride,
                    HH_K.data(), static_cast<size_t>(hh_count) * kv_stride * sizeof(float));
        std::memcpy(dense_V.data() + static_cast<size_t>(dense_count) * kv_stride,
                    HH_V.data(), static_cast<size_t>(hh_count) * kv_stride * sizeof(float));
        dense_count += hh_count;
    }
    // Recent Window in logical order
    for (int i = 0; i < rw_count; ++i) {
        int ring_idx = (rw_head_index + i) % rw_size;
        std::memcpy(dense_K.data() + static_cast<size_t>(dense_count) * kv_stride,
                    RW_K.data() + static_cast<size_t>(ring_idx) * kv_stride, kv_stride * sizeof(float));
        std::memcpy(dense_V.data() + static_cast<size_t>(dense_count) * kv_stride,
                    RW_V.data() + static_cast<size_t>(ring_idx) * kv_stride, kv_stride * sizeof(float));
        dense_count++;
    }

    // 3. Fused attention decode
    int total_tokens = dense_count + pbs_blocks_used * chunk_size;
    if (mean_attn_weights.size() < static_cast<size_t>(total_tokens)) {
        mean_attn_weights.resize(total_tokens);
    }

    if (use_fp16_meta) {
        fused_attention_decode_avx2(
            Q,
            dense_K.data(),
            dense_V.data(),
            dense_count,
            PBS_K_Packed.data(),
            PBS_K_Scales_fp16.data(),
            PBS_K_Zeroes_fp16.data(),
            PBS_V_Packed.data(),
            PBS_V_Scales_fp16.data(),
            PBS_V_Zeroes_fp16.data(),
            PBS_token_ids.data(),
            pbs_blocks_used,
            num_q_heads,
            num_kv_heads,
            head_dim,
            attn_output,
            mean_attn_weights.data(),
            k_group_size
        );
    } else {
        fused_attention_decode_avx2(
            Q,
            dense_K.data(),
            dense_V.data(),
            dense_count,
            PBS_K_Packed.data(),
            PBS_K_Scales_fp32.data(),
            PBS_K_Zeroes_fp32.data(),
            PBS_V_Packed.data(),
            PBS_V_Scales_fp32.data(),
            PBS_V_Zeroes_fp32.data(),
            PBS_token_ids.data(),
            pbs_blocks_used,
            num_q_heads,
            num_kv_heads,
            head_dim,
            attn_output,
            mean_attn_weights.data(),
            k_group_size
        );
    }

    // 4. Score feedback (exponential decay scoring)
    // Sinks
    for (int i = 0; i < s_count; ++i) {
        if (i < max_seq_len) {
            global_attn_scores[i] = score_decay * global_attn_scores[i] + mean_attn_weights[i];
        }
    }
    // Heavy Hitters
    for (int i = 0; i < hh_count; ++i) {
        int64_t tid = HH_token_ids[i];
        if (tid >= 0 && tid < max_seq_len) {
            global_attn_scores[tid] = score_decay * global_attn_scores[tid] + mean_attn_weights[s_count + i];
            HH_scores[i] = global_attn_scores[tid];
        }
    }
    // Recent Window
    int rw_base_id = total_processed_tokens - rw_count;
    for (int i = 0; i < rw_count; ++i) {
        int tid = rw_base_id + i;
        if (tid >= 0 && tid < max_seq_len) {
            global_attn_scores[tid] = score_decay * global_attn_scores[tid] + mean_attn_weights[s_count + hh_count + i];
        }
    }
    // PBS
    for (int i = 0; i < pbs_blocks_used * chunk_size; ++i) {
        int64_t tid = PBS_token_ids[i];
        if (tid >= 0 && tid < max_seq_len) {
            global_attn_scores[tid] = score_decay * global_attn_scores[tid] + mean_attn_weights[dense_count + i];
        }
    }
}

size_t TriTierCacheEngine::get_buffer_bytes() const {
    size_t bytes = 0;
    bytes += S_K.size() * sizeof(float) + S_V.size() * sizeof(float);
    bytes += RW_K.size() * sizeof(float) + RW_V.size() * sizeof(float);
    bytes += HH_K.size() * sizeof(float) + HH_V.size() * sizeof(float);
    bytes += HH_token_ids.size() * sizeof(int64_t) + HH_scores.size() * sizeof(float);
    bytes += WR_K.size() * sizeof(float) + WR_V.size() * sizeof(float);
    bytes += WR_token_ids.size() * sizeof(int64_t);
    // Explicitly accounted PBS allocated bytes
    bytes += PBS_K_Packed.size() * sizeof(int32_t);
    if (use_fp16_meta) {
        bytes += PBS_K_Scales_fp16.size() * sizeof(uint16_t);
        bytes += PBS_K_Zeroes_fp16.size() * sizeof(uint16_t);
        bytes += PBS_V_Scales_fp16.size() * sizeof(uint16_t);
        bytes += PBS_V_Zeroes_fp16.size() * sizeof(uint16_t);
    } else {
        bytes += PBS_K_Scales_fp32.size() * sizeof(float);
        bytes += PBS_K_Zeroes_fp32.size() * sizeof(float);
        bytes += PBS_V_Scales_fp32.size() * sizeof(float);
        bytes += PBS_V_Zeroes_fp32.size() * sizeof(float);
    }
    bytes += PBS_V_Packed.size() * sizeof(int32_t);
    bytes += PBS_token_ids.size() * sizeof(int64_t);
    bytes += global_attn_scores.size() * sizeof(float);
    return bytes;
}
