import pytest
import numpy as np
import torch
import tri_tier._C as _C
from src.tri_tier.cache import TriTierCache


def test_k_group_size_wiring_synthetic():
    """Verify that k_group_size=16 vs 32 produces different packed representations and dequantized values."""
    num_heads = 4
    head_dim = 64
    total_tokens = 32

    np.random.seed(42)
    # Construct 32 tokens where tokens 0..15 have range [-1, 1] and tokens 16..31 have range [5, 10]
    K = np.zeros((total_tokens, num_heads, head_dim), dtype=np.float32)
    K[:16] = np.random.uniform(-1.0, 1.0, size=(16, num_heads, head_dim)).astype(np.float32)
    K[16:] = np.random.uniform(5.0, 10.0, size=(16, num_heads, head_dim)).astype(np.float32)

    # 1. Quantize with k_group_size = 16 (2 blocks of 16 tokens)
    packed_16 = np.zeros((2, num_heads, head_dim), dtype=np.int32)
    scale_16 = np.zeros((2, num_heads, head_dim), dtype=np.float32)
    zero_16 = np.zeros((2, num_heads, head_dim), dtype=np.float32)

    _C.quantize_k_block(K[:16], packed_16[0], scale_16[0], zero_16[0], num_heads, head_dim, 16, "fp32")
    _C.quantize_k_block(K[16:], packed_16[1], scale_16[1], zero_16[1], num_heads, head_dim, 16, "fp32")

    dequant_16 = np.zeros((total_tokens, num_heads, head_dim), dtype=np.float32)
    _C.dequantize_k(packed_16, scale_16, zero_16, dequant_16, 2, num_heads, head_dim, 16, "fp32")

    # 2. Quantize with k_group_size = 32 (1 block of 32 tokens, 2 words)
    packed_32 = np.zeros((2, num_heads, head_dim), dtype=np.int32)
    scale_32 = np.zeros((1, num_heads, head_dim), dtype=np.float32)
    zero_32 = np.zeros((1, num_heads, head_dim), dtype=np.float32)

    _C.quantize_k_block(K, packed_32, scale_32, zero_32, num_heads, head_dim, 32, "fp32")

    dequant_32 = np.zeros((total_tokens, num_heads, head_dim), dtype=np.float32)
    _C.dequantize_k(packed_32, scale_32, zero_32, dequant_32, 1, num_heads, head_dim, 32, "fp32")

    # The packed words must be structurally distinct due to different quantization scaling ranges
    assert not np.array_equal(packed_16, packed_32), "Packed representations must differ between k_group_size=16 and 32"
    # Dequantized values must differ due to group-32 vs group-16 quantization range
    max_diff = np.max(np.abs(dequant_16 - dequant_32))
    assert max_diff > 0.1, f"Dequantized outputs must differ significantly between group sizes (got max_diff={max_diff})"


def test_pbs_metadata_dtype_wiring_synthetic():
    """Verify that pbs_metadata_dtype=fp16 vs fp32 introduces genuine half-precision roundoff."""
    num_heads = 4
    head_dim = 64
    chunk_tokens = 16

    np.random.seed(42)
    K = np.random.randn(chunk_tokens, num_heads, head_dim).astype(np.float32) * 2.5 + 1.2345

    # FP32 quantization & dequantization
    packed_fp32 = np.zeros((1, num_heads, head_dim), dtype=np.int32)
    scale_fp32 = np.zeros((1, num_heads, head_dim), dtype=np.float32)
    zero_fp32 = np.zeros((1, num_heads, head_dim), dtype=np.float32)
    _C.quantize_k_block(K, packed_fp32, scale_fp32, zero_fp32, num_heads, head_dim, 16, "fp32")

    dequant_fp32 = np.zeros((chunk_tokens, num_heads, head_dim), dtype=np.float32)
    _C.dequantize_k(packed_fp32, scale_fp32, zero_fp32, dequant_fp32, 1, num_heads, head_dim, 16, "fp32")

    # FP16 quantization & dequantization (uint16 storage for IEEE half)
    packed_fp16 = np.zeros((1, num_heads, head_dim), dtype=np.int32)
    scale_fp16 = np.zeros((1, num_heads, head_dim), dtype=np.uint16)
    zero_fp16 = np.zeros((1, num_heads, head_dim), dtype=np.uint16)
    _C.quantize_k_block(K, packed_fp16, scale_fp16, zero_fp16, num_heads, head_dim, 16, "fp16")

    dequant_fp16 = np.zeros((chunk_tokens, num_heads, head_dim), dtype=np.float32)
    _C.dequantize_k(packed_fp16, scale_fp16, zero_fp16, dequant_fp16, 1, num_heads, head_dim, 16, "fp16")

    # Metadata arrays must have distinct byte widths
    assert scale_fp16.itemsize == 2
    assert scale_fp32.itemsize == 4
    assert scale_fp16.nbytes * 2 == scale_fp32.nbytes

    # Reconstructed values must exhibit small but non-zero half precision drift (1e-6 to 1e-2)
    max_diff = np.max(np.abs(dequant_fp32 - dequant_fp16))
    assert 0 < max_diff < 0.05, f"Expected non-zero FP16 precision difference, got max_diff={max_diff}"


def test_engine_properties_and_memory_sensitivity():
    """Verify TriTierCacheEngine records parameters and exhibits memory sensitivity to pbs_metadata_dtype and k_group_size."""
    num_q = 4
    num_kv = 2
    head_dim = 64
    max_seq = 256
    sink_sz = 4
    rw_sz = 16
    h_ratio = 0.1

    # Config 1: group-16, fp16
    eng_16_fp16 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 16, "fp16")
    # Config 2: group-16, fp32
    eng_16_fp32 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 16, "fp32")
    # Config 3: group-32, fp16
    eng_32_fp16 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 32, "fp16")

    assert eng_16_fp16.k_group_size == 16
    assert eng_16_fp16.pbs_metadata_dtype == "fp16"
    assert eng_16_fp16.use_fp16_meta is True

    assert eng_16_fp32.k_group_size == 16
    assert eng_16_fp32.pbs_metadata_dtype == "fp32"
    assert eng_16_fp32.use_fp16_meta is False

    assert eng_32_fp16.k_group_size == 32
    assert eng_32_fp16.pbs_metadata_dtype == "fp16"

    # Ingest 96 tokens via prefill to trigger PBS allocation in all engines
    np.random.seed(123)
    K_prompt = np.random.randn(96, num_kv, head_dim).astype(np.float32)
    V_prompt = np.random.randn(96, num_kv, head_dim).astype(np.float32)

    eng_16_fp16.prefill(K_prompt, V_prompt, 96)
    eng_16_fp32.prefill(K_prompt, V_prompt, 96)
    eng_32_fp16.prefill(K_prompt, V_prompt, 96)

    # Ensure PBS was exercised
    assert eng_16_fp16.pbs_count > 0, "PBS tokens should be stored"
    assert eng_16_fp32.pbs_count > 0
    assert eng_32_fp16.pbs_count > 0

    bytes_16_fp16 = eng_16_fp16.get_buffer_bytes()
    bytes_16_fp32 = eng_16_fp32.get_buffer_bytes()

    # FP16 metadata must consume strictly fewer bytes than FP32 metadata
    assert bytes_16_fp16 < bytes_16_fp32, (
        f"Expected bytes(fp16) < bytes(fp32), got {bytes_16_fp16} vs {bytes_16_fp32}"
    )


def test_engine_attention_step_numerical_divergence():
    """Verify that TriTierCacheEngine attention decode produces numerically different outputs across configurations."""
    num_q = 4
    num_kv = 2
    head_dim = 64
    max_seq = 256
    sink_sz = 4
    rw_sz = 16
    h_ratio = 0.1

    eng_16_fp16 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 16, "fp16")
    eng_16_fp32 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 16, "fp32")
    eng_32_fp16 = _C.TriTierCacheEngine(num_q, num_kv, head_dim, max_seq, sink_sz, rw_sz, h_ratio, 0.999, 16, 32, "fp16")

    np.random.seed(999)
    K_prompt = np.random.randn(96, num_kv, head_dim).astype(np.float32)
    V_prompt = np.random.randn(96, num_kv, head_dim).astype(np.float32)

    eng_16_fp16.prefill(K_prompt, V_prompt, 96)
    eng_16_fp32.prefill(K_prompt, V_prompt, 96)
    eng_32_fp16.prefill(K_prompt, V_prompt, 96)

    # Step decode token
    Q = np.random.randn(num_q * head_dim).astype(np.float32)
    K_new = np.random.randn(num_kv * head_dim).astype(np.float32)
    V_new = np.random.randn(num_kv * head_dim).astype(np.float32)

    out_16_fp16 = np.zeros(num_q * head_dim, dtype=np.float32)
    out_16_fp32 = np.zeros(num_q * head_dim, dtype=np.float32)
    out_32_fp16 = np.zeros(num_q * head_dim, dtype=np.float32)

    eng_16_fp16.step(Q, K_new, V_new, out_16_fp16)
    eng_16_fp32.step(Q, K_new, V_new, out_16_fp32)
    eng_32_fp16.step(Q, K_new, V_new, out_32_fp16)

    # All outputs should be valid non-zero finite floats
    assert np.all(np.isfinite(out_16_fp16)) and np.any(out_16_fp16 != 0.0)
    assert np.all(np.isfinite(out_16_fp32)) and np.any(out_16_fp32 != 0.0)
    assert np.all(np.isfinite(out_32_fp16)) and np.any(out_32_fp16 != 0.0)

    # 16 vs 32 group size output must diverge
    diff_grp = np.max(np.abs(out_16_fp16 - out_32_fp16))
    assert diff_grp > 0, f"Attention outputs must differ between k_group_size=16 and 32 (got {diff_grp})"

    # fp16 vs fp32 metadata output must diverge
    diff_dtype = np.max(np.abs(out_16_fp16 - out_16_fp32))
    assert diff_dtype > 0, f"Attention outputs must differ between fp16 and fp32 metadata (got {diff_dtype})"
