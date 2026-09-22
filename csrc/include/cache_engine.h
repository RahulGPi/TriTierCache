#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

class TriTierCacheEngine {
public:
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int sink_size;
    int rw_size;
    float h_ratio;
    float score_decay;
    int update_interval;
    int k_group_size; // 16 or 32
    std::string pbs_metadata_dtype; // "fp16" or "fp32"
    bool use_fp16_meta;
    int chunk_size; // equals k_group_size (16 or 32)
    int quant_head_dim; // (head_dim + 15) / 16
    int max_hh;

    // Sinks
    int s_count;
    std::vector<float> S_K;
    std::vector<float> S_V;

    // Recent Window (Ring buffer)
    int rw_count;
    int rw_head_index;
    std::vector<float> RW_K;
    std::vector<float> RW_V;

    // Heavy Hitters
    int hh_count;
    std::vector<float> HH_K;
    std::vector<float> HH_V;
    std::vector<int64_t> HH_token_ids;
    std::vector<float> HH_scores;

    // Waiting Room
    int wr_count;
    std::vector<float> WR_K;
    std::vector<float> WR_V;
    std::vector<int64_t> WR_token_ids;

    // PBS (Tier 3 Compressed) - lazy chunked growth
    int pbs_allocated_blocks;
    int pbs_blocks_used;
    int pbs_count; // tokens in PBS
    std::vector<int32_t> PBS_K_Packed;
    std::vector<float> PBS_K_Scales_fp32;
    std::vector<float> PBS_K_Zeroes_fp32;
    std::vector<uint16_t> PBS_K_Scales_fp16;
    std::vector<uint16_t> PBS_K_Zeroes_fp16;

    std::vector<int32_t> PBS_V_Packed;
    std::vector<float> PBS_V_Scales_fp32;
    std::vector<float> PBS_V_Zeroes_fp32;
    std::vector<uint16_t> PBS_V_Scales_fp16;
    std::vector<uint16_t> PBS_V_Zeroes_fp16;
    std::vector<int64_t> PBS_token_ids;

    // Global attention scoring
    std::vector<float> global_attn_scores;
    int total_processed_tokens;
    float cached_threshold;
    int last_calc_pos;
    int total_evictions;

    // Scratch buffers for decode step
    std::vector<float> dense_K;
    std::vector<float> dense_V;
    std::vector<float> mean_attn_weights;

    TriTierCacheEngine(
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int max_seq_len,
        int sink_size = 4,
        int rw_size = 64,
        float h_ratio = 0.1f,
        float score_decay = 0.999f,
        int update_interval = 16,
        int k_group_size = 16,
        const std::string& pbs_metadata_dtype = "fp16"
    );

    ~TriTierCacheEngine() = default;

    void ensure_pbs_capacity(int min_blocks);

    void step(
        const float* Q,
        const float* K_new,
        const float* V_new,
        float* attn_output
    );

    void prefill(
        const float* K_all,
        const float* V_all,
        int q_len
    );

    float fetch_or_calc_threshold();

    void route_evicted_token(
        const float* k_ptr,
        const float* v_ptr,
        int64_t token_id
    );

    void flush_waiting_room_to_pbs();

    size_t get_buffer_bytes() const;
};
