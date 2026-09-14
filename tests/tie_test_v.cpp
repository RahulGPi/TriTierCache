// tie_test_v.cpp
#include "../csrc/include/quantize_v_avx2.h"
#include <cstdio>
#include <vector>
#include <cstdint>

int main() {
    int num_heads = 1, head_dim = 16;
    std::vector<float> V(16 * head_dim, 3.0f);

    float* row = V.data(); // token 0
    row[0] = 0.0f;   // min
    row[1] = 6.0f;   // max -> scale = 2.0
    row[2] = 1.0f;   // normalized 0.5 -> want 0 (round-to-even)
    row[3] = 3.0f;   // normalized 1.5 -> want 2
    row[10] = 5.0f;  // normalized 2.5 -> want 2 (second 8-lane chunk)

    std::vector<int32_t> packed_full(16 * num_heads * ((head_dim + 15) / 16));
    std::vector<float> scale(16 * num_heads), zero(16 * num_heads);

    quantize_v_block_avx2(V.data(), packed_full.data(), scale.data(), zero.data(), num_heads, head_dim);

    uint32_t p = (uint32_t)packed_full[0];
    int q2  = (p >> (2*2))  & 0b11;
    int q3  = (p >> (2*3))  & 0b11;
    int q10 = (p >> (2*10)) & 0b11;
    printf("scale=%.4f  ch2=%d(want 0)  ch3=%d(want 2)  ch10=%d(want 2)  %s\n",
           scale[0], q2, q3, q10, (q2==0 && q3==2 && q10==2) ? "OK" : "FAIL");
}