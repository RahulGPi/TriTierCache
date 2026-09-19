# TriTierCache: Hierarchical Memory-Compressed KV Cache

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://github.com/huggingface/transformers)
[![Acceleration](https://img.shields.io/badge/SIMD-AVX2%20%7C%20FMA%20%7C%20OpenMP-0071C5.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**TriTierCache** is a high-performance, hierarchical Key-Value (KV) cache architecture designed for long-context autoregressive Large Language Model (LLM) inference on modern x86_64 architectures (featuring AVX2, FMA, BMI2, and OpenMP multi-threading). 

By combining **attention sinks**, **full-precision recent windows**, **importance-routed heavy hitters**, and **asymmetric 2-bit packed background storage** with a custom **C++ AVX2 fused attention engine**, TriTierCache reduces KV cache memory footprints by up to 4× while preserving generative perplexity and high decoding throughput.

---

## Architecture Overview

Standard full-precision KV caches grow linearly with sequence length ($O(N)$), rapidly bottlenecking memory bandwidth and cache capacity during long autoregressive generation. TriTierCache organizes KV storage into three distinct tiers plus attention sinks:

```text
+---------------------------------------------------------------------------------------------------------+
|                                    TRITIERCACHE MEMORY HIERARCHY                                        |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  +---------------------------+   +---------------------------+   +-----------------------------------+  |
|  |     Attention Sinks       |   |   Tier 1: Recent Window   |   |     Tier 2: Heavy Hitters (HH)    |  |
|  |     (Initial 4 Tokens)    |   |     (Latest 256 Tokens)   |   |     (Top 5% Important Tokens)     |  |
|  |     - FP32 Exact Storage  |   |     - FP32 Ring Buffer    |   |     - FP32 Dynamic Eviction       |  |
|  |     - Softmax Stability   |   |     - Exact Local Context |   |     - Global Attention Scoring    |  |
|  +-------------+-------------+   +-------------+-------------+   +-----------------+-----------------+  |
|                |                               |                               |                        |
|                +-------------------------------+-------------------------------+                        |
|                                                |                                                        |
|                                                v                                                        |
|                              +-----------------------------------+                                      |
|                              |   Tier 3: Background Storage      |                                      |
|                              |   - 2-bit Asymmetric Quantization |                                      |
|                              |   - K: Per-Channel Block Packing  |                                      |
|                              |   - V: Per-Token Channel Packing  |                                      |
|                              |   - 16-Token Packed Blocks (PBS)  |                                      |
|                              +-----------------+-----------------+                                      |
|                                                |                                                        |
+------------------------------------------------+--------------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |        C++ AVX2 / OpenMP Fused Attention          |
                       |        - Multi-threaded Query Head Parallelism    |
                       |        - Direct In-Kernel 2-Bit Dequantization    |
                       |        - AVX2 Vectorized Dot-Products & AXPY      |
                       |        - Grouped Query Attention (GQA) Support    |
                       +---------------------------------------------------+
```

### The Tier Storage Hierarchy

| Tier / Buffer | Data Representation | Sizing & Allocation | Purpose & Policy |
| :--- | :--- | :--- | :--- |
| **Attention Sinks** | FP32 (Exact) | Fixed `SINK_SIZE = 4` | Retains initial sequence tokens to preserve attention distribution stability (StreamingLLM). |
| **Tier 1: Recent Window (RW)** | FP32 (Exact) | Fixed `R_size` (e.g., 256 tokens) | Ring buffer storing latest tokens in full precision for exact local context. FIFO eviction. |
| **Tier 2: Heavy Hitters (HH)** | FP32 (Exact) | Dynamic `H_ratio` (e.g., top 5% of `max_seq_len`) | Stores critical historical tokens with the highest cumulative attention scores (`Global_Attn_Scr`). Dynamically demoted when stronger tokens arrive. |
| **Tier 3: Background Storage (PBS)** | 2-bit Quantized (Packed `int32`) | Remaining token capacity (`max_background_tokens`) | KIVI-style asymmetric 2-bit quantization for background tokens. Staged via 16-token Packed Block Storage (`PBS`). |

---

## High-Performance C++ AVX2 & OpenMP Acceleration

TriTierCache features an optimized native C++ kernel (`tri_tier._C`) compiled with `-mavx2`, `-mbmi2`, `-mfma`, and `-fopenmp`:

1. **Two-Pass Fused Decode Attention (`fused_attention_decode_avx2`)**:
   - **Pass A (Score Projection)**: Computes dot products between query vectors $Q$ and dense KV (Sinks + Recent Window + Heavy Hitters) alongside 2-bit packed background blocks (PBS) using 256-bit AVX2 SIMD operations.
   - **Softmax Normalization**: Computes numerically stable vector softmax scaling (`row_max` and `row_sum_exp`) per query head.
   - **Pass B (Value Aggregation)**: Accumulates weighted $V$ vectors into the attention output using vectorized `axpy_avx2` FMA instructions.
2. **OpenMP Multi-Threading**:
   - Scales linearly across CPU cores by distributing query head processing across threads (`#pragma omp parallel for`).
3. **On-the-Fly 2-Bit Dequantization**:
   - Dequantizes 2-bit packed keys and values directly in CPU cache during attention passes, avoiding intermediate full-tensor memory reallocations.
4. **Grouped Query Attention (GQA)**:
   - Full native support for GQA architectures (such as LLaMA 3 / LLaMA 3.2), mapping query head groups to key-value heads.
5. **Instant Batched Prefill**:
   - Parallel prompt processing during prefill that seamlessly ingests prompt KV states into the tiered hierarchy in a single pass.

---

## Project Structure

```text
├── csrc/                              # High-performance C++ AVX2 extension
│   ├── bindings.cpp                   # PyBind11 bindings for tri_tier._C
│   ├── cpu/
│   │   ├── dequantize_avx2.cpp        # AVX2 2-bit K/V dequantization kernels
│   │   ├── fused_attn_avx2.cpp        # OpenMP & AVX2 multi-head fused attention
│   │   ├── quantize_k_avx2.cpp        # AVX2 2-bit per-channel key quantizer
│   │   └── quantize_v_avx2.cpp        # AVX2 2-bit per-token value quantizer
│   └── include/
│       ├── dequant_avx2.h             # Header for dequantization routines
│       ├── fused_attn_avx2.h          # Header for fused attention kernel
│       └── quant_avx2.h               # Header for quantization routines
├── tri_tier/                          # Core Python package
│   ├── __init__.py                    # Package initialization & exports
│   ├── constants.py                   # Default hyperparameters (CHUNK_SIZE, SINK_SIZE, etc.)
│   ├── cache/
│   │   ├── __init__.py
│   │   └── tri_tier_cache.py          # TriTierCache implementation
│   └── integration/
│       ├── __init__.py
│       └── patch_llama.py             # Hugging Face LLaMA attention monkey-patch
├── benchmarks/                        # Comprehensive evaluation suite
│   ├── utils.py                       # Benchmark helpers and baseline loaders
│   ├── benchmark_correctness.py       # Greedy token match and cosine similarity tests
│   ├── benchmark_latency.py           # TTFT and decode latency profiling
│   ├── benchmark_memory.py            # Peak RSS and KV footprint profiling
│   ├── benchmark_perplexity.py        # Long-sequence perplexity evaluation
│   └── run_all_benchmarks.py          # Unified benchmark runner
├── tests/                             # Unit tests
│   ├── test_cache_init.py             # Initialization & buffer sizing tests
│   ├── test_cache_methods.py          # Ingestion, routing, and scoring tests
│   └── test_patch_llama.py            # LLaMA patch & GQA verification tests
├── pyproject.toml                     # Build system specifications
├── setup.py                           # C++ extension build script (AVX2 + OpenMP)
└── smoke_test.py                      # End-to-end smoke test script
```

---

## Installation & Build

### Prerequisites
- Linux x86_64 CPU with **AVX2**, **FMA**, and **BMI2** support
- GCC / G++ $\ge$ 9.0 (with OpenMP support)
- Python $\ge$ 3.10
- PyTorch $\ge$ 2.1.0

### Build from Source

```bash
# 1. Clone the repository
git clone https://github.com/RahulGPi/TriTierCache.git
cd TriTierCache

# 2. Install build dependencies
pip install pybind11 setuptools

# 3. Compile the C++ AVX2 extension in-place
python setup.py build_ext --inplace

# 4. Verify extension loading
python -c "import tri_tier._C as _C; print('Extension built successfully:', _C.fused_attention_decode)"
```

---

## Quickstart

### 1. Seamless Hugging Face LLaMA Integration

You can monkey-patch Hugging Face's `LlamaAttention` layers with a single call:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tri_tier.integration.patch_llama import apply_patch

# 1. Apply the TriTierCache patch
apply_patch()

# 2. Load model and tokenizer
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 3. Generate text autoregressively
prompt = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Standalone Cache Usage

You can also use [`TriTierCache`](file:///home/rahulpai/Global_Programming/idk_what_to_name_this/tri_tier/cache/tri_tier_cache.py) directly:

```python
import torch
from tri_tier import TriTierCache

# Initialize cache (e.g. 32 heads, 128 head dim, max 2048 tokens)
cache = TriTierCache(
    max_seq_len=2048,
    head_dim=128,
    num_heads=32,
    R_size=256,
    H_ratio=0.05
)

# Ingest new token projection [num_heads, head_dim]
k_new = torch.randn(32, 128)
v_new = torch.randn(32, 128)
cache.ingest_token(k_new, v_new)

# Reconstruct full ordered cache for attention
k_full, v_full, full_ids = cache.reconstruct_full_cache()

# Compute attention and accumulate feedback scores
attn_weights = torch.randn(1, 32, 1, k_full.shape[0]).softmax(dim=-1)
cache.accumulate_attn_scrs(attn_weights, full_ids)
```

---

## Benchmarks & Evaluation

TriTierCache includes an automated benchmark suite comparing TriTierCache against Vanilla Hugging Face Attention (`DynamicCache`):

```bash
# Run quick benchmark validation across all metrics
PYTHONPATH=. python benchmarks/run_all_benchmarks.py --quick

# Or run individual benchmark modules:
PYTHONPATH=. python benchmarks/benchmark_correctness.py
PYTHONPATH=. python benchmarks/benchmark_latency.py
PYTHONPATH=. python benchmarks/benchmark_memory.py
PYTHONPATH=. python benchmarks/benchmark_perplexity.py
```

Generated metrics are automatically exported to CSV in `benchmarks/results/`:
- `correctness_results.csv`: Output token match and cosine similarity.
- `latency_results.csv`: TTFT (prefill) and per-token decode latency across prompt lengths.
- `memory_results.csv`: Peak RSS memory and KV cache buffer footprint.
- `perplexity_results.csv`: Perplexity evaluation across sequence lengths.

---

## Running Unit Tests

Run the test suite with `pytest`:

```bash
PYTHONPATH=. pytest -v
```

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
