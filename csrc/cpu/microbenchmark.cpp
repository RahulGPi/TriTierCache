#include "fused_attn_avx2.h"
#include <vector>
#include <chrono>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    int num_q_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int dense_count = 68; // 4 sinks + 64 RW
    int num_blocks = 32; // 32 * 16 = 512 PBS tokens -> 580 tokens total
    int iters = (argc > 1) ? std::atoi(argv[1]) : 5000;

    std::vector<float> Q(num_q_heads * head_dim, 1.0f);
    std::vector<float> dense_K(dense_count * num_kv_heads * head_dim, 1.0f);
    std::vector<float> dense_V(dense_count * num_kv_heads * head_dim, 1.0f);
    
    std::vector<int32_t> PBS_K_Packed(num_blocks * num_kv_heads * head_dim, 1);
    std::vector<float> PBS_K_Scales(num_blocks * num_kv_heads * head_dim, 1.0f);
    std::vector<float> PBS_K_Zeroes(num_blocks * num_kv_heads * head_dim, 0.0f);
    
    std::vector<int32_t> PBS_V_Packed(num_blocks * 16 * num_kv_heads * ((head_dim + 15) / 16), 1);
    std::vector<float> PBS_V_Scales(num_blocks * 16 * num_kv_heads, 1.0f);
    std::vector<float> PBS_V_Zeroes(num_blocks * 16 * num_kv_heads, 0.0f);
    
    std::vector<int64_t> PBS_token_ids(num_blocks * 16, 0);
    for(int i = 0; i < num_blocks * 16; i++) PBS_token_ids[i] = i;
    
    std::vector<float> attn_output(num_q_heads * head_dim, 0.0f);
    std::vector<float> mean_attn(dense_count + num_blocks * 16, 0.0f);
    
    // warmup
    for (int i = 0; i < 50; ++i) {
        fused_attention_decode_avx2(
            Q.data(), dense_K.data(), dense_V.data(), dense_count,
            PBS_K_Packed.data(), PBS_K_Scales.data(), PBS_K_Zeroes.data(),
            PBS_V_Packed.data(), PBS_V_Scales.data(), PBS_V_Zeroes.data(),
            PBS_token_ids.data(), num_blocks, num_q_heads, num_kv_heads, head_dim,
            attn_output.data(), mean_attn.data()
        );
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        fused_attention_decode_avx2(
            Q.data(), dense_K.data(), dense_V.data(), dense_count,
            PBS_K_Packed.data(), PBS_K_Scales.data(), PBS_K_Zeroes.data(),
            PBS_V_Packed.data(), PBS_V_Scales.data(), PBS_V_Zeroes.data(),
            PBS_token_ids.data(), num_blocks, num_q_heads, num_kv_heads, head_dim,
            attn_output.data(), mean_attn.data()
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    double avg_ms = diff.count() / iters;
    std::cout << "Iterations: " << iters << "\n";
    std::cout << "Total time: " << diff.count() << " ms\n";
    std::cout << "Latency: " << avg_ms << " ms/step\n";
    return 0;
}
