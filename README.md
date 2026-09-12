# TriTierCache: Hierarchical Memory-Compressed KV Cache

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://github.com/huggingface/transformers)
[![Architecture](https://img.shields.io/badge/Hardware%20Target-Intel%2012th%20Gen%20Alder%20Lake-0071C5.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An optimized, multi-tier Key-Value (KV) cache architecture designed specifically for long-context autoregressive LLM inference on **Intel® 12th Gen "Alder Lake" Hybrid Architecture** (CPU + Integrated Xe-LP Iris Xe Graphics). **TriTierCache** drastically reduces KV cache memory footprint and bandwidth bottlenecks during long-sequence generation while maintaining generation accuracy and low latency.

---

## Hardware Target: Intel® 12th Gen Alder Lake Architecture

This project is tailored specifically for the **Intel Alder Lake microarchitecture**, mapping memory tiers and compute workloads directly onto the Alder Lake SoC hardware blocks.

```text
+---------------------------------------------------------------------------------------------------------+
|                               INTEL® ALDER LAKE HYBRID SOC ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  +-------------------------------------+   +------------------------------------+                       |
|  |     Golden Cove (P-Cores)           |   |       Gracemont (E-Cores)          |                       |
|  |     - High Single-Thread IPC        |   |       - High-Density Throughput    |                       |
|  |     - AVX2 / FMA3 / AVX-VNNI        |   |       - AVX2 / AVX-VNNI            |                       |
|  |     - Decode & Attention MatMul     |   |       - Async Chunking & Eviction  |                       |
|  +------------------+------------------+   +-----------------+------------------+                       |
|                     |                                        |                                          |
|  ===================+========================================+================== Coherent Ring Bus     |
|                     |                                        |                                          |
|  +------------------+------------------+   +-----------------+------------------+                       |
|  |   Shared Intel® Smart Cache (L3)    |   |   Intel® Xe-LP GT2 Graphics (iGPU) |                       |
|  |   - Up to 30 MB LLC                 |   |   - 80–96 Execution Units (EUs)    |                       |
|  |   - High-Bandwidth Ring Routing     |   |   - Native DP4A AI Acceleration    |                       |
|  |   - Ring Buffer & Tier 1/2 Resident |   |   - INT8 / INT4 / INT2 Dot-Product |                       |
|  +------------------+------------------+   +-----------------+------------------+                       |
|                     |                                        |                                          |
|  ===================+========================================+================== Memory Crossbar        |
|                                        |                                                                |
|                     +------------------+-------------------+                                            |
|                     | Dual-Channel Unified Memory (UMA)    |                                            |
|                     | - Shared DDR4 / DDR5 / LPDDR5        |                                            |
|                     | - Zero-Copy Host <-> iGPU Access     |                                            |
|                     +--------------------------------------+                                            |
+---------------------------------------------------------------------------------------------------------+
```

### Alder Lake Hardware Subsystems & Workload Mapping

1. **Golden Cove (Performance Cores / P-Cores)**
   - **Role**: Executes high-priority, latency-sensitive decode steps, RoPE calculations, FP32 projection matmuls, and full-precision Tier 1/Tier 2 attention calculations.
   - **Instruction Sets**: **AVX2**, **FMA3**, and **Intel® AVX-VNNI** (Vector Neural Network Instructions) for 256-bit SIMD matrix math and bit-manipulation (unpacking 2-bit values directly to FP32).

2. **Gracemont (Efficient Cores / E-Cores)**
   - **Role**: Offloads asynchronous background operations, including 16-token Waiting Room chunking, KIVI-style 2-bit quantization, and global attention score sorting/eviction routing without starving P-core compute threads.

3. **Intel® Iris® Xe Graphics (Xe-LP GT2 iGPU)**
   - **Role**: Accelerates batched dequantization and integer dot-products directly on the integrated GPU.
   - **Compute Hardware**: Up to 96 Execution Units (EUs) / 768 ALUs with hardware **DP4A** (Dot Product 4-element Accumulate) instructions for high-throughput INT8/INT4/INT2 acceleration.
   - **Programming Interfaces**: **Intel® oneAPI / SYCL**, **Level Zero**, and **Intel Extension for PyTorch (IPEX)**.

4. **Zero-Copy Unified Memory Architecture (UMA)**
   - The CPU and Xe-LP iGPU share the same physical memory space across the system bus. TriTierCache eliminates PCIe transfer bottlenecks when demoting/promoting tokens between Tier 1 (FP32), Tier 2 (FP32), and Tier 3 (2-bit packed).

---

## Top 5 Alder Lake Hardware Configurations Using this Architecture

The Intel 12th Gen Alder Lake series powers multiple form factors sharing this unified CPU + Xe-LP architecture. TriTierCache targets the following top 5 hardware platform tiers:

| Tier / Platform | Representative SKUs | CPU Core Config | iGPU Specs (Xe-LP) | Target TDP | Primary Optimization Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Alder Lake-H** *(Primary Reference)* | **Core™ i5-12500H**<br>Core™ i7-12700H<br>Core™ i9-12900H | Up to 14C / 20T<br>(6P + 8E or 4P + 8E) | **Iris® Xe GT2**<br>80–96 EUs (640–768 ALUs)<br>DP4A AI Accel. | **45W** (Base)<br>95W–115W (Boost) | Maximum P+E throughput with full 80/96 EU iGPU acceleration. |
| **2. Alder Lake-P** | Core™ i7-1280P<br>Core™ i7-1260P<br>Core™ i5-1240P | Up to 14C / 20T<br>(6P + 8E or 4P + 8E) | **Iris® Xe GT2**<br>80–96 EUs (640–768 ALUs)<br>DP4A AI Accel. | **28W** (Base)<br>64W (Boost) | Performance thin-and-light laptop deployment with high EU density. |
| **3. Alder Lake-U** | Core™ i7-1265U<br>Core™ i7-1255U<br>Core™ i5-1235U | Up to 10C / 12T<br>(2P + 8E) | **Iris® Xe GT2**<br>80–96 EUs (640–768 ALUs)<br>DP4A AI Accel. | **15W** (Base)<br>55W (Boost) | Ultrabook & handheld low-power inference using E-core chunking. |
| **4. Alder Lake-S** | Core™ i9-12900K<br>Core™ i7-12700K<br>Core™ i5-12600K | Up to 16C / 24T<br>(8P + 8E) | **Intel® UHD 770**<br>32 EUs (256 ALUs)<br>Xe-LP Architecture | **65W–125W** (Base)<br>150W–241W (Boost) | Desktop platform with high P-core frequencies and up to 30 MB L3 cache. |
| **5. Alder Lake-HX** | Core™ i9-12950HX<br>Core™ i7-12850HX<br>Core™ i5-12600HX | Up to 16C / 24T<br>(8P + 8E) | **Intel® UHD 770**<br>32 EUs (256 ALUs)<br>Xe-LP Architecture | **55W** (Base)<br>157W (Boost) | Enthusiast mobile workstation with 16-core desktop die packaging. |

---

## Architecture Overview

Standard full-precision KV caches grow linearly with context length ($O(N)$), quickly exhausting memory bandwidth and LLC capacity during long auto-regressive generation. TriTierCache introduces a **three-tier hierarchical storage strategy with attention sinks**, categorizing tokens based on recency and cumulative attention importance.

![Tri-Tier KV Cache High-Level Architecture](tri_tier_cache_hld.svg)

### The Storage Hierarchy

| Tier / Buffer | Data Representation | Sizing & Allocation | Purpose & Policy |
| :--- | :--- | :--- | :--- |
| **Attention Sinks** | FP32 (Exact) | Fixed `SINK_SIZE = 4` | Retains initial sequence tokens to preserve attention distribution stability (StreamingLLM). |
| **Tier 1: Recent Window (RW)** | FP32 (Exact) | Fixed `R_size` (e.g., 256 tokens) | Ring buffer storing latest tokens in full precision for exact local context. FIFO eviction. |
| **Tier 2: Heavy Hitters (HH)** | FP32 (Exact) | Dynamic `H_ratio` (e.g., top 5% of `max_seq_len`) | Stores the most critical historical tokens with the highest cumulative attention scores. Dynamically demoted when stronger tokens arrive. |
| **Tier 3: Background Storage (PBS)** | 2-bit Quantized (Packed `int32`) | Remaining token capacity (`max_background_tokens`) | KIVI-style asymmetric 2-bit quantization for background tokens. Staged via a 16-token Waiting Room (`WR`). |

---

## Detailed Pipeline & Token Lifecycle

![Tri-Tier KV Cache Low-Level Design](tri_tier_cache_lld.svg)

### 1. Ingestion & FIFO Eviction
- Incoming tokens ($K, V$) are first placed into the **Attention Sink** buffer until full (`SINK_SIZE = 4`).
- Subsequent tokens populate the **Recent Window ring buffer** (`RW_K_Buffer`, `RW_V_Buffer`).
- When the ring buffer reaches capacity, the oldest token at `RW_head_index` is evicted and passed to the eviction router.

### 2. Eviction Routing & Heavy Hitter Management
- Attention scores are tracked globally in `Global_Attn_Scr`.
- A 95th-percentile attention score threshold is recalculated dynamically every `UPDATE_THRESHOLD = 32` tokens via `torch.quantile`.
- **Routing Decision**:
  - **Score $\ge$ Threshold**: 
    - If Tier 2 (Heavy Hitters) has capacity, the token is added directly.
    - If Tier 2 is full, the token's score is compared against the weakest Heavy Hitter (`argmin(HH_scores)`). If it beats the weakest HH, the weakest token is demoted to Tier 3 via `compress_and_store()` and replaced by the new token.
  - **Score $<$ Threshold**: Routed directly to Tier 3 via `compress_and_store()`.

### 3. Asymmetric 2-bit Quantization (KIVI-Style)
Tokens demoted or routed to Tier 3 enter an intermediate 16-token **Waiting Room** (`WR`). Once `CHUNK_SIZE = 16` tokens accumulate:
- **Key ($K$) Quantization**: Quantized **per-channel** across the 16 tokens in the chunk. 16 2-bit values are packed into a single `int32` per channel. Block-level scales and zero-points are preserved.
- **Value ($V$) Quantization**: Quantized **per-token** across head channels. 16 2-bit values are packed into a single `int32` per token. Token-level scales and zero-points are preserved.
- The packed tensors are stored in `PBS_K_Packed` and `PBS_V_Packed`.

### 4. Cache Reconstruction (Before Attention)
To compute attention accurately:
1. Tier 3 background tokens are unpacked and dequantized back to FP32 using bitwise shifts, masks (`0b11`), scales, and zero-points.
2. Tier 2 (Heavy Hitters) and Tier 3 (Background) are merged and sorted chronologically by token ID (`argsort(Middle_ids)`).
3. Tier 1 (Recent Window) is unrolled in chronological order from the ring buffer using `torch.roll`.
4. Sinks, Middle, and Recent Window tokens are concatenated into contiguous tensors: `K_Full`, `V_Full`, and `Full_ids`.

### 5. Attention & Score Accumulation Loop
- Scaled dot-product attention is computed: $\text{softmax}\left(\frac{Q \cdot K_{\text{full}}^T}{\sqrt{d_k}}\right) \cdot V_{\text{full}}$.
- Resulting attention weights are averaged across heads and added into `Global_Attn_Scr` via `index_add_` matching `Full_ids`, feeding the next eviction decision.

---

## Project Structure

```text
├── tri_tier/
│   ├── __init__.py            # Package initialization & exports
│   ├── cache.py               # Core TriTierCache class & quantization logic
│   ├── constants.py           # Config constants (CHUNK_SIZE, SINK_SIZE, etc.)
│   └── integration/
│       ├── __init__.py        # Integration exports
│       └── patch_llama.py     # Monkey patch for Hugging Face LLaMA attention
├── csrc/                      # High-performance C++ & hardware extension scaffolding
│   ├── cpu/                   # AVX2 & AVX-VNNI optimized SIMD kernels (Alder Lake P/E cores)
│   └── include/               # Header definitions for packing/unpacking
├── benchmarks/                # Benchmarking suite (Memory, Latency, Perplexity)
├── tests/                     # Unit test suites
│   ├── Test_cache_init.py     # Cache initialization and buffer sizing tests
│   └── test_cache_methods.py  # Ingestion, routing, and scoring unit tests
├── tri_tier_cache_hld.svg     # High-Level Architecture Diagram
├── tri_tier_cache_lld.svg     # Low-Level Design Flowchart
└── pyproject.toml             # Project configuration & dependencies
```

---

## Quickstart

### Prerequisites
- Python $\ge$ 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`
- PyTorch $\ge$ 2.1.0

### Installation

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/RahulGPi/TriTierCache.git
cd TriTierCache

# Install dependencies using uv
uv sync
```

### Usage with Hugging Face LLaMA

You can monkey-patch Hugging Face's `LlamaAttention` layer with a single line:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tri_tier.integration.patch_llama import apply_patch

# 1. Apply the TriTierCache patch to LlamaAttention
apply_patch()

# 2. Load your model as normal
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 3. Generate tokens autoregressively (decode mode)
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Standalone Cache Usage

You can also instantiate and manipulate the cache directly:

```python
import torch
from tri_tier import TriTierCache

# Initialize cache
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
# attn_weights shape: [batch_size, num_heads, 1, seq_len]
attn_weights = torch.randn(1, 32, 1, k_full.shape[0]).softmax(dim=-1)
cache.accumulate_attn_scrs(attn_weights, full_ids)
```

---

## Running Tests

Run the test suite with `pytest`:

```bash
PYTHONPATH=. uv run pytest -v
```

---

## Roadmap & Hardware Optimization Goals

- [x] Pure PyTorch 3-Tier KV Cache prototype (Sinks + Recent Window + Heavy Hitters + 2-bit Background).
- [x] KIVI-style asymmetric 2-bit quantization (per-channel for $K$, per-token for $V$).
- [x] Hugging Face `LlamaAttention` integration hook with GQA support.
- [ ] **Alder Lake CPU SIMD (`csrc/cpu/`)**: Custom AVX2 + AVX-VNNI vectorized kernels for fast 2-bit packing and dequantization on Golden Cove / Gracemont cores.
- [ ] **Intel Iris Xe (Xe-LP) iGPU Acceleration**: SYCL / Level Zero / IPEX kernels utilizing DP4A for INT8/INT4/INT2 matrix multiply on Iris Xe.
- [ ] Direct fused attention on compressed KV tiers without intermediate FP32 buffer allocation.
- [ ] Pre-fill phase batching support.
- [ ] Comprehensive latency and memory benchmarks on Intel Core i5-12500H + Iris Xe Graphics.
