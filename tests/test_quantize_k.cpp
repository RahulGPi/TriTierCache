// csrc/cpu/test_quantize_k.cpp
#include "../csrc/include/quantize_k_avx2.h"
#include <cstdio>
#include <vector>

int main() {
    int num_heads = 2, head_dim = 8;
    std::vector<float> K(16 * num_heads * head_dim);
    for (size_t i = 0; i < K.size(); ++i) K[i] = static_cast<float>(i % 7) - 3.0f;

    std::vector<uint32_t> packed(num_heads * head_dim);
    std::vector<float> scale(num_heads * head_dim), zero(num_heads * head_dim);

    quantize_k_block_avx2(K.data(), packed.data(), scale.data(), zero.data(),
                           num_heads, head_dim);

    for (int i = 0; i < num_heads * head_dim; ++i)
        printf("[%d] min=%.4f scale=%.4f packed=0x%08x\n", i, zero[i], scale[i], packed[i]);
    return 0;
}