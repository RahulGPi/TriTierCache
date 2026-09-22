#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include <sstream>
#include <stdexcept>

#include "include/cache_engine.h"
#include "include/fused_attn_avx2.h"
#include "include/quantize_k_avx2.h"
#include "include/quantize_v_avx2.h"
#include "include/dequant_avx2.h"

namespace py = pybind11;

/**
 * Helper to extract direct continuous buffer pointers from:
 * 1. Raw integer memory addresses (e.g., tensor.data_ptr())
 * 2. PyTorch tensors (with continuity verification)
 * 3. NumPy arrays and Python buffer objects (with C-continuity verification)
 *
 * Performs zero data copies and ensures strict memory continuity before C++ execution.
 */
template <typename T>
static T* get_contiguous_ptr(py::object obj, const char* name, bool writable = false, bool nullable = false) {
    if (obj.is_none()) {
        if (nullable) {
            return nullptr;
        }
        throw std::invalid_argument(std::string("Argument '") + name + "' cannot be None.");
    }

    // 1. Direct raw memory address passed as Python integer
    if (py::isinstance<py::int_>(obj)) {
        uintptr_t addr = obj.cast<uintptr_t>();
        if (addr == 0) {
            if (nullable) {
                return nullptr;
            }
            throw std::invalid_argument(std::string("Argument '") + name + "' pointer address is NULL.");
        }
        return reinterpret_cast<T*>(addr);
    }

    // 2. PyTorch Tensor (inspect continuity and data pointer)
    if (py::hasattr(obj, "data_ptr") && py::hasattr(obj, "is_contiguous")) {
        bool is_contig = obj.attr("is_contiguous")().cast<bool>();
        if (!is_contig) {
            throw std::invalid_argument(
                std::string("Argument '") + name + "' PyTorch tensor is not contiguous in memory. Call .contiguous() before passing."
            );
        }
        uintptr_t addr = obj.attr("data_ptr")().cast<uintptr_t>();
        if (addr == 0 && !nullable) {
            throw std::invalid_argument(std::string("Argument '") + name + "' tensor data pointer is NULL.");
        }
        return reinterpret_cast<T*>(addr);
    }

    // 3. Python Buffer Protocol (NumPy array, memoryview, etc.)
    if (py::isinstance<py::buffer>(obj)) {
        py::buffer buf = py::reinterpret_borrow<py::buffer>(obj);
        py::buffer_info info = buf.request(writable);

        // Verify C-contiguity: stride[d] must equal itemsize * prod(shape[d+1..ndim-1])
        bool is_c_contig = true;
        ssize_t expected_stride = info.itemsize;
        for (int i = info.ndim - 1; i >= 0; --i) {
            if (info.shape[i] > 1) {
                if (info.strides[i] != expected_stride) {
                    is_c_contig = false;
                    break;
                }
                expected_stride *= info.shape[i];
            }
        }

        if (!is_c_contig) {
            throw std::invalid_argument(
                std::string("Argument '") + name + "' buffer is not C-contiguous in memory."
            );
        }

        if (info.ptr == nullptr && !nullable) {
            throw std::invalid_argument(std::string("Argument '") + name + "' buffer data pointer is NULL.");
        }

        return static_cast<T*>(info.ptr);
    }

    std::ostringstream oss;
    oss << "Argument '" << name << "' has unsupported type '"
        << py::str(py::type::of(obj)).cast<std::string>()
        << "'. Expected PyTorch tensor, NumPy array, buffer object, or integer memory address.";
    throw std::invalid_argument(oss.str());
}

static void py_fused_attention_decode(
    py::object Q_obj,
    py::object dense_K_obj,
    py::object dense_V_obj,
    int dense_count,
    py::object PBS_K_Packed_obj,
    py::object PBS_K_Scales_obj,
    py::object PBS_K_Zeroes_obj,
    py::object PBS_V_Packed_obj,
    py::object PBS_V_Scales_obj,
    py::object PBS_V_Zeroes_obj,
    py::object PBS_token_ids_obj,
    int num_blocks,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    py::object attn_output_obj,
    py::object mean_attn_weights_obj = py::none(),
    int k_group_size = 16,
    const std::string& pbs_metadata_dtype = "fp32")
{
    if (dense_count < 0) {
        throw std::invalid_argument("dense_count must be non-negative.");
    }
    if (num_blocks < 0) {
        throw std::invalid_argument("num_blocks must be non-negative.");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("num_q_heads, num_kv_heads, and head_dim must be positive integers.");
    }

    bool is_fp16 = (pbs_metadata_dtype == "fp16");
    if (py::hasattr(PBS_K_Scales_obj, "dtype")) {
        std::string dt_str = py::str(PBS_K_Scales_obj.attr("dtype")).cast<std::string>();
        if (dt_str.find("float16") != std::string::npos || dt_str.find("half") != std::string::npos) {
            is_fp16 = true;
        }
    }

    const float* Q_ptr = get_contiguous_ptr<const float>(Q_obj, "Q", false);
    const float* dense_K_ptr = get_contiguous_ptr<const float>(dense_K_obj, "dense_K", false, dense_count == 0);
    const float* dense_V_ptr = get_contiguous_ptr<const float>(dense_V_obj, "dense_V", false, dense_count == 0);

    const int32_t* PBS_K_Packed_ptr = get_contiguous_ptr<const int32_t>(PBS_K_Packed_obj, "PBS_K_Packed", false, num_blocks == 0);
    const int32_t* PBS_V_Packed_ptr = get_contiguous_ptr<const int32_t>(PBS_V_Packed_obj, "PBS_V_Packed", false, num_blocks == 0);
    const int64_t* PBS_token_ids_ptr = get_contiguous_ptr<const int64_t>(PBS_token_ids_obj, "PBS_token_ids", false, num_blocks == 0);

    float* attn_output_ptr = get_contiguous_ptr<float>(attn_output_obj, "attn_output", true);
    float* mean_attn_weights_ptr = get_contiguous_ptr<float>(mean_attn_weights_obj, "mean_attn_weights", true, true);

    if (is_fp16) {
        const uint16_t* PBS_K_Scales_ptr = get_contiguous_ptr<const uint16_t>(PBS_K_Scales_obj, "PBS_K_Scales", false, num_blocks == 0);
        const uint16_t* PBS_K_Zeroes_ptr = get_contiguous_ptr<const uint16_t>(PBS_K_Zeroes_obj, "PBS_K_Zeroes", false, num_blocks == 0);
        const uint16_t* PBS_V_Scales_ptr = get_contiguous_ptr<const uint16_t>(PBS_V_Scales_obj, "PBS_V_Scales", false, num_blocks == 0);
        const uint16_t* PBS_V_Zeroes_ptr = get_contiguous_ptr<const uint16_t>(PBS_V_Zeroes_obj, "PBS_V_Zeroes", false, num_blocks == 0);

        py::gil_scoped_release release;
        fused_attention_decode_avx2(
            Q_ptr, dense_K_ptr, dense_V_ptr, dense_count,
            PBS_K_Packed_ptr, PBS_K_Scales_ptr, PBS_K_Zeroes_ptr,
            PBS_V_Packed_ptr, PBS_V_Scales_ptr, PBS_V_Zeroes_ptr,
            PBS_token_ids_ptr, num_blocks, num_q_heads, num_kv_heads, head_dim,
            attn_output_ptr, mean_attn_weights_ptr, k_group_size
        );
    } else {
        const float* PBS_K_Scales_ptr = get_contiguous_ptr<const float>(PBS_K_Scales_obj, "PBS_K_Scales", false, num_blocks == 0);
        const float* PBS_K_Zeroes_ptr = get_contiguous_ptr<const float>(PBS_K_Zeroes_obj, "PBS_K_Zeroes", false, num_blocks == 0);
        const float* PBS_V_Scales_ptr = get_contiguous_ptr<const float>(PBS_V_Scales_obj, "PBS_V_Scales", false, num_blocks == 0);
        const float* PBS_V_Zeroes_ptr = get_contiguous_ptr<const float>(PBS_V_Zeroes_obj, "PBS_V_Zeroes", false, num_blocks == 0);

        py::gil_scoped_release release;
        fused_attention_decode_avx2(
            Q_ptr, dense_K_ptr, dense_V_ptr, dense_count,
            PBS_K_Packed_ptr, PBS_K_Scales_ptr, PBS_K_Zeroes_ptr,
            PBS_V_Packed_ptr, PBS_V_Scales_ptr, PBS_V_Zeroes_ptr,
            PBS_token_ids_ptr, num_blocks, num_q_heads, num_kv_heads, head_dim,
            attn_output_ptr, mean_attn_weights_ptr, k_group_size
        );
    }
}

static void py_quantize_k_block(
    py::object K_obj,
    py::object packed_obj,
    py::object scale_obj,
    py::object zero_obj,
    int num_heads,
    int head_dim,
    int k_group_size = 16,
    const std::string& pbs_metadata_dtype = "fp32")
{
    const float* K_ptr = get_contiguous_ptr<const float>(K_obj, "K", false);
    uint32_t* packed_ptr = reinterpret_cast<uint32_t*>(get_contiguous_ptr<int32_t>(packed_obj, "packed", true));

    bool is_fp16 = (pbs_metadata_dtype == "fp16");
    if (py::hasattr(scale_obj, "dtype")) {
        std::string dt_str = py::str(scale_obj.attr("dtype")).cast<std::string>();
        if (dt_str.find("float16") != std::string::npos || dt_str.find("half") != std::string::npos) {
            is_fp16 = true;
        }
    }

    if (is_fp16) {
        uint16_t* scale_ptr = get_contiguous_ptr<uint16_t>(scale_obj, "scale", true);
        uint16_t* zero_ptr = get_contiguous_ptr<uint16_t>(zero_obj, "zero", true);
        py::gil_scoped_release release;
        quantize_k_block_avx2(K_ptr, packed_ptr, scale_ptr, zero_ptr, num_heads, head_dim, k_group_size);
    } else {
        float* scale_ptr = get_contiguous_ptr<float>(scale_obj, "scale", true);
        float* zero_ptr = get_contiguous_ptr<float>(zero_obj, "zero", true);
        py::gil_scoped_release release;
        quantize_k_block_avx2(K_ptr, packed_ptr, scale_ptr, zero_ptr, num_heads, head_dim, k_group_size);
    }
}

static void py_quantize_v_block(
    py::object V_obj,
    py::object packed_obj,
    py::object scale_obj,
    py::object zero_obj,
    int num_heads,
    int head_dim,
    int chunk_size = 16,
    const std::string& pbs_metadata_dtype = "fp32")
{
    const float* V_ptr = get_contiguous_ptr<const float>(V_obj, "V", false);
    int32_t* packed_ptr = get_contiguous_ptr<int32_t>(packed_obj, "packed", true);

    bool is_fp16 = (pbs_metadata_dtype == "fp16");
    if (py::hasattr(scale_obj, "dtype")) {
        std::string dt_str = py::str(scale_obj.attr("dtype")).cast<std::string>();
        if (dt_str.find("float16") != std::string::npos || dt_str.find("half") != std::string::npos) {
            is_fp16 = true;
        }
    }

    if (is_fp16) {
        uint16_t* scale_ptr = get_contiguous_ptr<uint16_t>(scale_obj, "scale", true);
        uint16_t* zero_ptr = get_contiguous_ptr<uint16_t>(zero_obj, "zero", true);
        py::gil_scoped_release release;
        quantize_v_block_avx2(V_ptr, packed_ptr, scale_ptr, zero_ptr, num_heads, head_dim, chunk_size);
    } else {
        float* scale_ptr = get_contiguous_ptr<float>(scale_obj, "scale", true);
        float* zero_ptr = get_contiguous_ptr<float>(zero_obj, "zero", true);
        py::gil_scoped_release release;
        quantize_v_block_avx2(V_ptr, packed_ptr, scale_ptr, zero_ptr, num_heads, head_dim, chunk_size);
    }
}

static void py_dequantize_k(
    py::object packed_obj,
    py::object scale_obj,
    py::object zero_obj,
    py::object out_obj,
    int num_blocks,
    int num_heads,
    int head_dim,
    int k_group_size = 16,
    const std::string& pbs_metadata_dtype = "fp32")
{
    const int32_t* packed_ptr = get_contiguous_ptr<const int32_t>(packed_obj, "packed", false);
    float* out_ptr = get_contiguous_ptr<float>(out_obj, "out", true);

    bool is_fp16 = (pbs_metadata_dtype == "fp16");
    if (py::hasattr(scale_obj, "dtype")) {
        std::string dt_str = py::str(scale_obj.attr("dtype")).cast<std::string>();
        if (dt_str.find("float16") != std::string::npos || dt_str.find("half") != std::string::npos) {
            is_fp16 = true;
        }
    }

    if (is_fp16) {
        const uint16_t* scale_ptr = get_contiguous_ptr<const uint16_t>(scale_obj, "scale", false);
        const uint16_t* zero_ptr = get_contiguous_ptr<const uint16_t>(zero_obj, "zero", false);
        py::gil_scoped_release release;
        dequantize_k_avx2(packed_ptr, scale_ptr, zero_ptr, out_ptr, num_blocks, num_heads, head_dim, k_group_size);
    } else {
        const float* scale_ptr = get_contiguous_ptr<const float>(scale_obj, "scale", false);
        const float* zero_ptr = get_contiguous_ptr<const float>(zero_obj, "zero", false);
        py::gil_scoped_release release;
        dequantize_k_avx2(packed_ptr, scale_ptr, zero_ptr, out_ptr, num_blocks, num_heads, head_dim, k_group_size);
    }
}

static void py_dequantize_v(
    py::object packed_obj,
    py::object scale_obj,
    py::object zero_obj,
    py::object out_obj,
    int total_tokens,
    int num_heads,
    int head_dim,
    const std::string& pbs_metadata_dtype = "fp32")
{
    const int32_t* packed_ptr = get_contiguous_ptr<const int32_t>(packed_obj, "packed", false);
    float* out_ptr = get_contiguous_ptr<float>(out_obj, "out", true);

    bool is_fp16 = (pbs_metadata_dtype == "fp16");
    if (py::hasattr(scale_obj, "dtype")) {
        std::string dt_str = py::str(scale_obj.attr("dtype")).cast<std::string>();
        if (dt_str.find("float16") != std::string::npos || dt_str.find("half") != std::string::npos) {
            is_fp16 = true;
        }
    }

    if (is_fp16) {
        const uint16_t* scale_ptr = get_contiguous_ptr<const uint16_t>(scale_obj, "scale", false);
        const uint16_t* zero_ptr = get_contiguous_ptr<const uint16_t>(zero_obj, "zero", false);
        py::gil_scoped_release release;
        dequantize_v_avx2(packed_ptr, scale_ptr, zero_ptr, out_ptr, total_tokens, num_heads, head_dim);
    } else {
        const float* scale_ptr = get_contiguous_ptr<const float>(scale_obj, "scale", false);
        const float* zero_ptr = get_contiguous_ptr<const float>(zero_obj, "zero", false);
        py::gil_scoped_release release;
        dequantize_v_avx2(packed_ptr, scale_ptr, zero_ptr, out_ptr, total_tokens, num_heads, head_dim);
    }
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "TriTier C++ Extension with AVX2-accelerated fused attention and caching engine";

    // Convert C++ exceptions into standard Python exceptions
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_IndexError, e.what());
        } catch (const std::bad_alloc& e) {
            PyErr_SetString(PyExc_MemoryError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception occurred in TriTier compute engine.");
        }
    });

    // Fused streaming decode attention
    m.def(
        "fused_attention_decode",
        &py_fused_attention_decode,
        py::arg("Q_ptr"),
        py::arg("dense_K_ptr"),
        py::arg("dense_V_ptr"),
        py::arg("dense_count"),
        py::arg("PBS_K_Packed_ptr"),
        py::arg("PBS_K_Scales_ptr"),
        py::arg("PBS_K_Zeroes_ptr"),
        py::arg("PBS_V_Packed_ptr"),
        py::arg("PBS_V_Scales_ptr"),
        py::arg("PBS_V_Zeroes_ptr"),
        py::arg("PBS_token_ids_ptr"),
        py::arg("num_blocks"),
        py::arg("num_q_heads"),
        py::arg("num_kv_heads"),
        py::arg("head_dim"),
        py::arg("attn_output_ptr"),
        py::arg("mean_attn_weights_ptr") = py::none(),
        py::arg("k_group_size") = 16,
        py::arg("pbs_metadata_dtype") = "fp32",
        "Fused streaming decode attention over dense KV and 2-bit compressed PBS KV without intermediate materialization."
    );

    // Standalone AVX2 quantization functions
    m.def("quantize_k_block", &py_quantize_k_block,
          py::arg("K"), py::arg("packed"), py::arg("scale"), py::arg("zero"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("k_group_size") = 16, py::arg("pbs_metadata_dtype") = "fp32",
          "AVX2-accelerated 2-bit channel-wise quantization for Key 16-token blocks.");
    m.def("quantize_v_block", &py_quantize_v_block,
          py::arg("V"), py::arg("packed"), py::arg("scale"), py::arg("zero"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("chunk_size") = 16, py::arg("pbs_metadata_dtype") = "fp32",
          "AVX2-accelerated 2-bit token-wise quantization for Value 16-token blocks.");
    m.def("dequantize_k", &py_dequantize_k,
          py::arg("packed"), py::arg("scale"), py::arg("zero"), py::arg("out"), py::arg("num_blocks"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("k_group_size") = 16, py::arg("pbs_metadata_dtype") = "fp32",
          "AVX2-accelerated dequantization for Key blocks.");
    m.def("dequantize_v", &py_dequantize_v,
          py::arg("packed"), py::arg("scale"), py::arg("zero"), py::arg("out"), py::arg("total_tokens"), py::arg("num_heads"), py::arg("head_dim"),
          py::arg("pbs_metadata_dtype") = "fp32",
          "AVX2-accelerated dequantization for Value tokens.");

    // TriTierCacheEngine class
    py::class_<TriTierCacheEngine>(m, "TriTierCacheEngine")
        .def(
            py::init<int, int, int, int, int, int, float, float, int, int, const std::string&>(),
            py::arg("num_q_heads"),
            py::arg("num_kv_heads"),
            py::arg("head_dim"),
            py::arg("max_seq_len"),
            py::arg("sink_size") = 4,
            py::arg("rw_size") = 64,
            py::arg("h_ratio") = 0.1f,
            py::arg("score_decay") = 0.999f,
            py::arg("update_interval") = 16,
            py::arg("k_group_size") = 16,
            py::arg("pbs_metadata_dtype") = "fp16"
        )
        .def(
            "step",
            [](TriTierCacheEngine& self, py::object q_obj, py::object k_obj, py::object v_obj, py::object out_obj) {
                const float* q_ptr = get_contiguous_ptr<const float>(q_obj, "Q", false);
                const float* k_ptr = get_contiguous_ptr<const float>(k_obj, "K_new", false);
                const float* v_ptr = get_contiguous_ptr<const float>(v_obj, "V_new", false);
                float* out_ptr = get_contiguous_ptr<float>(out_obj, "attn_output", true);

                {
                    py::gil_scoped_release release;
                    self.step(q_ptr, k_ptr, v_ptr, out_ptr);
                }
            },
            py::arg("q_ptr"),
            py::arg("k_ptr"),
            py::arg("v_ptr"),
            py::arg("out_ptr"),
            "Executes single-token ingestion, fused AVX2 attention decode, and score feedback without Python GIL."
        )
        .def(
            "prefill",
            [](TriTierCacheEngine& self, py::object k_obj, py::object v_obj, int q_len) {
                if (q_len < 0) {
                    throw std::invalid_argument("q_len must be non-negative.");
                }
                const float* k_ptr = get_contiguous_ptr<const float>(k_obj, "K_all", false);
                const float* v_ptr = get_contiguous_ptr<const float>(v_obj, "V_all", false);

                {
                    py::gil_scoped_release release;
                    self.prefill(k_ptr, v_ptr, q_len);
                }
            },
            py::arg("k_ptr"),
            py::arg("v_ptr"),
            py::arg("q_len"),
            "Prefills KV cache with batch prompt tokens without Python GIL."
        )
        .def_readonly("num_q_heads", &TriTierCacheEngine::num_q_heads)
        .def_readonly("num_kv_heads", &TriTierCacheEngine::num_kv_heads)
        .def_readonly("head_dim", &TriTierCacheEngine::head_dim)
        .def_readonly("max_seq_len", &TriTierCacheEngine::max_seq_len)
        .def_readonly("sink_size", &TriTierCacheEngine::sink_size)
        .def_readonly("rw_size", &TriTierCacheEngine::rw_size)
        .def_readonly("h_ratio", &TriTierCacheEngine::h_ratio)
        .def_readonly("score_decay", &TriTierCacheEngine::score_decay)
        .def_readonly("update_interval", &TriTierCacheEngine::update_interval)
        .def_readonly("k_group_size", &TriTierCacheEngine::k_group_size)
        .def_readonly("chunk_size", &TriTierCacheEngine::chunk_size)
        .def_readonly("pbs_metadata_dtype", &TriTierCacheEngine::pbs_metadata_dtype)
        .def_readonly("use_fp16_meta", &TriTierCacheEngine::use_fp16_meta)
        .def_readonly("pbs_allocated_blocks", &TriTierCacheEngine::pbs_allocated_blocks)
        .def_readonly("s_count", &TriTierCacheEngine::s_count)
        .def_readonly("rw_count", &TriTierCacheEngine::rw_count)
        .def_readonly("rw_head_index", &TriTierCacheEngine::rw_head_index)
        .def_readonly("hh_count", &TriTierCacheEngine::hh_count)
        .def_readonly("pbs_count", &TriTierCacheEngine::pbs_count)
        .def_readonly("pbs_blocks_used", &TriTierCacheEngine::pbs_blocks_used)
        .def_readonly("total_processed_tokens", &TriTierCacheEngine::total_processed_tokens)
        .def_readonly("total_evictions", &TriTierCacheEngine::total_evictions)
        .def("get_buffer_bytes", &TriTierCacheEngine::get_buffer_bytes,
             "Returns the total allocated buffer memory in bytes.")
        .def("get_global_attn_scores", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.max_seq_len)},
                {static_cast<ssize_t>(sizeof(float))},
                self.global_attn_scores.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around global attention scores.")
        .def("get_S_K", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.sink_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.S_K.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Sink Key buffer.")
        .def("get_S_V", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.sink_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.S_V.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Sink Value buffer.")
        .def("get_RW_K", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.rw_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.RW_K.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Recent Window Key buffer.")
        .def("get_RW_V", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.rw_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.RW_V.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Recent Window Value buffer.")
        .def("get_HH_K", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.max_hh), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.HH_K.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Heavy Hitter Key buffer.")
        .def("get_HH_V", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.max_hh), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                {static_cast<ssize_t>(self.num_kv_heads * self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(self.head_dim * sizeof(float)),
                 static_cast<ssize_t>(sizeof(float))},
                self.HH_V.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Heavy Hitter Value buffer.")
        .def("get_HH_token_ids", [](TriTierCacheEngine& self) {
            return py::array_t<int64_t>(
                {static_cast<ssize_t>(self.max_hh)},
                {static_cast<ssize_t>(sizeof(int64_t))},
                self.HH_token_ids.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Heavy Hitter token IDs.")
        .def("get_HH_scores", [](TriTierCacheEngine& self) {
            return py::array_t<float>(
                {static_cast<ssize_t>(self.max_hh)},
                {static_cast<ssize_t>(sizeof(float))},
                self.HH_scores.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around Heavy Hitter scores.")
        .def("get_PBS_K_Packed", [](TriTierCacheEngine& self) {
            int words = self.k_group_size / 16;
            return py::array_t<int32_t>(
                {static_cast<ssize_t>(self.pbs_allocated_blocks), static_cast<ssize_t>(words), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                self.PBS_K_Packed.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around PBS Key packed buffer.")
        .def("get_PBS_K_Scales", [](TriTierCacheEngine& self) -> py::object {
            if (self.use_fp16_meta) {
                return py::array_t<uint16_t>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                    self.PBS_K_Scales_fp16.data(),
                    py::cast(&self)
                );
            } else {
                return py::array_t<float>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                    self.PBS_K_Scales_fp32.data(),
                    py::cast(&self)
                );
            }
        }, "Zero-copy NumPy array wrapper around PBS Key scales buffer.")
        .def("get_PBS_K_Zeroes", [](TriTierCacheEngine& self) -> py::object {
            if (self.use_fp16_meta) {
                return py::array_t<uint16_t>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                    self.PBS_K_Zeroes_fp16.data(),
                    py::cast(&self)
                );
            } else {
                return py::array_t<float>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.head_dim)},
                    self.PBS_K_Zeroes_fp32.data(),
                    py::cast(&self)
                );
            }
        }, "Zero-copy NumPy array wrapper around PBS Key zeroes buffer.")
        .def("get_PBS_V_Packed", [](TriTierCacheEngine& self) {
            return py::array_t<int32_t>(
                {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(self.quant_head_dim)},
                self.PBS_V_Packed.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around PBS Value packed buffer.")
        .def("get_PBS_V_Scales", [](TriTierCacheEngine& self) -> py::object {
            if (self.use_fp16_meta) {
                return py::array_t<uint16_t>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(1)},
                    self.PBS_V_Scales_fp16.data(),
                    py::cast(&self)
                );
            } else {
                return py::array_t<float>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(1)},
                    self.PBS_V_Scales_fp32.data(),
                    py::cast(&self)
                );
            }
        }, "Zero-copy NumPy array wrapper around PBS Value scales buffer.")
        .def("get_PBS_V_Zeroes", [](TriTierCacheEngine& self) -> py::object {
            if (self.use_fp16_meta) {
                return py::array_t<uint16_t>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(1)},
                    self.PBS_V_Zeroes_fp16.data(),
                    py::cast(&self)
                );
            } else {
                return py::array_t<float>(
                    {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size), static_cast<ssize_t>(self.num_kv_heads), static_cast<ssize_t>(1)},
                    self.PBS_V_Zeroes_fp32.data(),
                    py::cast(&self)
                );
            }
        }, "Zero-copy NumPy array wrapper around PBS Value zeroes buffer.")
        .def("get_PBS_token_ids", [](TriTierCacheEngine& self) {
            return py::array_t<int64_t>(
                {static_cast<ssize_t>(self.pbs_allocated_blocks * self.chunk_size)},
                self.PBS_token_ids.data(),
                py::cast(&self)
            );
        }, "Zero-copy NumPy array wrapper around PBS token IDs buffer.");
}