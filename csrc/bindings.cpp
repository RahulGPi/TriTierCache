// csrc/bindings.cpp
#include <pybind11/pybind11.h>
#include "include/fused_attn_avx2.h"

namespace py = pybind11;

void py_fused_attention_decode(
    uintptr_t Q_ptr,
    uintptr_t dense_K_ptr, uintptr_t dense_V_ptr, int dense_count,
    uintptr_t PBS_K_Packed_ptr, uintptr_t PBS_K_Scales_ptr, uintptr_t PBS_K_Zeroes_ptr,
    uintptr_t PBS_V_Packed_ptr, uintptr_t PBS_V_Scales_ptr, uintptr_t PBS_V_Zeroes_ptr,
    uintptr_t PBS_token_ids_ptr,
    int num_blocks, int num_heads, int head_dim,
    uintptr_t attn_output_ptr)
{
    fused_attention_decode_avx2(
        reinterpret_cast<const float*>(Q_ptr),
        reinterpret_cast<const float*>(dense_K_ptr),
        reinterpret_cast<const float*>(dense_V_ptr),
        dense_count,
        reinterpret_cast<const int32_t*>(PBS_K_Packed_ptr),
        reinterpret_cast<const float*>(PBS_K_Scales_ptr),
        reinterpret_cast<const float*>(PBS_K_Zeroes_ptr),
        reinterpret_cast<const int32_t*>(PBS_V_Packed_ptr),
        reinterpret_cast<const float*>(PBS_V_Scales_ptr),
        reinterpret_cast<const float*>(PBS_V_Zeroes_ptr),
        reinterpret_cast<const int64_t*>(PBS_token_ids_ptr),
        num_blocks, num_heads, head_dim,
        reinterpret_cast<float*>(attn_output_ptr));
}

PYBIND11_MODULE(_C, m) {
    m.def("fused_attention_decode", &py_fused_attention_decode,
          "Fused streaming decode attention (K/V never fully materialized)");
}