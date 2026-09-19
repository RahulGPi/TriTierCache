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
    int num_blocks, int num_q_heads, int num_kv_heads, int head_dim,
    uintptr_t attn_output_ptr,
    uintptr_t mean_attn_weights_ptr = 0)
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
        num_blocks, num_q_heads, num_kv_heads, head_dim,
        reinterpret_cast<float*>(attn_output_ptr),
        mean_attn_weights_ptr ? reinterpret_cast<float*>(mean_attn_weights_ptr) : nullptr);
}

PYBIND11_MODULE(_C, m) {
    m.def("fused_attention_decode", &py_fused_attention_decode,
          py::arg("Q_ptr"),
          py::arg("dense_K_ptr"), py::arg("dense_V_ptr"), py::arg("dense_count"),
          py::arg("PBS_K_Packed_ptr"), py::arg("PBS_K_Scales_ptr"), py::arg("PBS_K_Zeroes_ptr"),
          py::arg("PBS_V_Packed_ptr"), py::arg("PBS_V_Scales_ptr"), py::arg("PBS_V_Zeroes_ptr"),
          py::arg("PBS_token_ids_ptr"),
          py::arg("num_blocks"), py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("head_dim"),
          py::arg("attn_output_ptr"),
          py::arg("mean_attn_weights_ptr") = 0,
          "Fused streaming decode attention (K/V never fully materialized)");
}