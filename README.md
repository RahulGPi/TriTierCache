<div align="center">

# TriTierCache

**A hierarchical, memory-compressed KV cache for long-context LLM inference on consumer CPUs.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-LLaMA%20%7C%20SmolLM-yellow.svg)](https://github.com/huggingface/transformers)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20FMA%20%7C%20BMI2%20%7C%20OpenMP-0071C5.svg)](https://www.intel.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## The problem

Every token a language model generates is appended to a Key-Value cache. That cache grows linearly with context length and is read in full on every decoding step. At 32k context with a 32-head model, it is hundreds of megabytes of traffic per token. On a machine with a discrete datacenter GPU this is annoying. On a laptop with an integrated GPU and shared system RAM, it is the wall you hit first, long before compute becomes the limit.

The usual answers are to truncate old context, or to quantize the entire cache uniformly. Truncation loses information the model actually needs. Uniform quantization spends the same number of bits on a token nobody attends to as it does on the one the answer depends on.

## The idea

Not all cached tokens matter equally, so they should not all be stored at the same precision. TriTierCache sorts every token into one of four places and keeps precision where attention actually lands.

| Where the token lives | Precision | Who goes there |
| :--- | :--- | :--- |
| Attention sinks | Exact FP32 | The first 4 tokens of the sequence. Never evicted. They anchor the softmax distribution and keep generation stable over long runs. |
| Tier 1, recent window | Exact FP32 | The most recent 256 tokens, in a ring buffer. Local context is where most attention lands, so it stays lossless. |
| Tier 2, heavy hitters | Exact FP32 | Older tokens that keep attracting attention, tracked by a running attention score. Roughly the top 5 percent. |
| Tier 3, background store | 2-bit packed | Everything else. Quantized asymmetrically and packed 16 values to an int32. |

When a token falls out of the recent window, it is scored. High scorers are promoted into the heavy-hitter tier. The rest are staged in a 16-token waiting room, quantized in one batch, and written into the background store. Heavy hitters that go cold later get demoted into the background store too, so the tiers keep rebalancing as generation continues.

<div align="center">

<!-- High-level architecture -->
<img src="tri_tier_cache_hld.svg" alt="TriTierCache high level architecture" width="820"/>

</div>

## Why it is fast, not just small

Compression usually costs speed, because something has to decompress the data before attention can read it. TriTierCache avoids that by never materializing a decompressed cache.

A native C++ extension (`tri_tier._C`) runs the whole decode step in one fused kernel. It streams the compressed tier one 16-token block at a time through a small scratch buffer that stays resident in L2, unpacks 2-bit values inline with AVX2, and folds them straight into the dot product. Peak scratch memory during attention is measured in kilobytes rather than the hundreds of megabytes a full reconstruction would need. Query heads are split across cores with OpenMP.

If the extension is not compiled, the package falls back to a pure PyTorch reference path. Same results, slower. This also doubles as the correctness oracle the C++ kernels are tested against.

## Results

Measured on `meta-llama/Llama-3.2-1B` and `HuggingFaceTB/SmolLM-135M`, Linux x86_64, AVX2 + FMA + OpenMP. Baseline is Hugging Face uncompressed FP32 Attention (`DynamicCache`).

### Core Benchmark Summary

| Category | Metric | Vanilla Baseline | TriTierCache | Result / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel latency** | Fused attention decode (`SmolLM-135M`) | 1050.00 µs/step | **373.79 µs/step** | **2.81x faster** |
| **Thread scaling** | 1 thread $\to$ 8 threads (OpenMP) | 978.04 µs | **274.46 µs** | **3.56x speedup** |
| **Decode latency** | Single token decode, 256 ctx | 41.25 ms/tok | **40.33 ms/tok** | 24.79 tok/s |
| **Prefill latency** | 128-token prompt (TTFT) | 149.53 ms | **183.24 ms** | 1.22x slower (one-time routing) |
| **Active memory (256 ctx)** | Initial window KV footprint | 16.00 MB | **16.00 MB** | Exact parity (lossless FP32) |
| **Active memory (8k ctx)** | Footprint @ 8192 tokens (`Llama-3.2-1B`) | 512.00 MB | **109.03 MB** | **4.70x vs FP32** (2.35x vs FP16) |
| **Active memory (32k ctx)** | Footprint @ 32768 tokens (`Llama-3.2-1B`) | 2048.00 MB | **394.11 MB** | **5.20x vs FP32** (2.60x vs FP16, Config A)<br>**7.10x vs FP32** (3.55x vs FP16, Config B) |
| **Canonical Perplexity** | Multi-sample 1024 tokens (`Llama-3.2-1B`) | 7.2107 ± 2.9186 | **7.2235 ± 2.9183** (Config A)<br>**7.2478 ± 2.9500** (Config B) | **+0.0128 PPL** (+0.18%, Config A)<br>**+0.0370 PPL** (+0.51%, Config B) |
| **Teacher-Forced Agreement** | Next-token accuracy (1024 prompt tokens) | 100.0% | **97.36%** | Logit cosine **0.999259** |
| **Autoregressive Agreement** | 500 generated tokens from 4096 prompt | 100.0% | **92.40%** (462/500, Config A)<br>**91.60%** (458/500, Config B) | Step 24 div (Config A)<br>Step 14 div (Config B) |
| **NIAH Retrieval (Llama 8k–16k)** | Needle retrieval across $1.0\times$–$2.0\times$ base | 27/27 (100.0%) | **54/54 (100.0%)** (Mode 'a') | **Flawless 100% retrieval** up to $2.0\times$ base |
| **NIAH Retrieval (SmolLM 2k–3k)** | Needle retrieval across $1.0\times$–$1.5\times$ base | 6/21 (28.6%) | **8/42 (19.0%)** (Mode 'a') | 100% @ 1.0x & 1.1x (RW); model-level cliff @ 1.2x+ |
---

<!-- DOWNSTREAM_ACCURACY_START -->
### Downstream Long-Context Task Accuracy

Evaluated on long-context tasks (context $\ge 1024$ tokens) where $>75\%$ of KV tokens reside in Tier 3 (2-bit PBS). Baseline is uncompressed FP32 Vanilla Hugging Face Attention.

#### `TinyLlama-1.1B-Chat-v1.0` (LLAMA arch, 1100M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **60.0%** / 80.0% | **90.0%** / 100.0% | **0.0%** / 0.0% | **88.3%** | 77.9% | 1.01x (194.8 ms) |
| **2,048** | **0.0%** / 0.0% | **0.0%** / 0.0% | **0.0%** / 0.0% | **100.0%** | 57.9% | 0.99x (184.0 ms) |
<!-- DOWNSTREAM_ACCURACY_END -->

---

Evaluated across both `meta-llama/Llama-3.2-1B` (base context 8192) and `HuggingFaceTB/SmolLM-135M` (base context 2048) using a diverse, non-repeating corpus (1,198 strictly unique paragraphs, >33k tokens, zero repeated 8-grams).

#### `meta-llama/Llama-3.2-1B` Graduated Sweep (8,192 to 16,384 Tokens — 135 Rows Evaluated)
Evaluated across 9 context lengths spanning $1.0\times$ ($8192$) up to $2.0\times$ ($16384$), depths $0.20, 0.70, 0.95$, and $H_{\text{ratio}} \in \{0.05, 0.15\}$ against Vanilla FP32 baseline:

| Length Ratio | Context Length | Vanilla FP32 Pass Rate | TriTier Mode 'a' Pass Rate | TriTier Mode 'b' Pass Rate | Mean PBS K Cosine | Mean PBS V Cosine |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.00x** | 8,192 | **100.0%** (3/3) | **100.0%** (6/6) | 66.7% (4/6) | 0.98754197 | 0.89593019 |
| **1.10x** | 9,011 | **100.0%** (3/3) | **100.0%** (6/6) | 33.3% (2/6) | 0.98754197 | 0.89593019 |
| **1.20x** | 9,830 | **100.0%** (3/3) | **100.0%** (6/6) | 100.0% (6/6) | 0.98754197 | 0.89593019 |
| **1.25x** | 10,240 | **100.0%** (3/3) | **100.0%** (6/6) | 66.7% (4/6) | 0.98738475 | 0.89321886 |
| **1.30x** | 10,650 | **100.0%** (3/3) | **100.0%** (6/6) | 33.3% (2/6) | 0.98822898 | 0.89893341 |
| **1.40x** | 11,469 | **100.0%** (3/3) | **100.0%** (6/6) | 0.0% (0/6) | 0.98890083 | 0.89728348 |
| **1.50x** | 12,288 | **100.0%** (3/3) | **100.0%** (6/6) | 100.0% (6/6) | 0.98914598 | 0.89308047 |
| **1.75x** | 14,336 | **100.0%** (3/3) | **100.0%** (6/6) | 66.7% (4/6) | 0.98860122 | 0.89772713 |
| **2.00x** | 16,384 | **100.0%** (3/3) | **100.0%** (6/6) | 0.0% (0/6) | 0.98722045 | 0.89842809 |
| **OVERALL** | **8,192–16,384** | **100.0% (27/27)** | **100.0% (54/54)** | **51.9% (28/54)** | **0.98812400** | **0.89735000** |

* **Repetition Trap Resolution**: Previous failure at 8,188 tokens was 100% an artifact of repeated sentence prompts inducing greedy decoding loops (`"\xa0 with varying"` in both Vanilla and TriTier). On non-repeating text, Llama-3.2-1B achieves 100% pass across all lengths through 16,384 tokens ($2.0\times$ base context).
* **RoPE Mode Invariance**: Native RoPE (Mode 'a') achieves **100.0% retrieval across all 54 configurations**. Window position clamping (Mode 'b') degrades to 51.9% due to query-key rotary phase misalignment.

#### `HuggingFaceTB/SmolLM-135M` Graduated Sweep (2,048 to 3,072 Tokens — 105 Rows Evaluated)
* **At 1.00x (2048 ctx)**: Vanilla FP32 = **100.0%** (3/3), TriTier Mode 'a' = **100.0%** (6/6).
* **At 1.10x (2252 ctx)**: Vanilla FP32 = **100.0%** (3/3), TriTier Mode 'a' depth 0.95 (RW) = **100.0%** (2/2).
* **At 1.20x–1.50x (2457–3072 ctx)**: Vanilla FP32 drops to **0.0%** and TriTier drops to **0.0%**, collapsing into identical filler tokens.
* **Model Attention Limit**: In all 35 rows at depth 0.95, the needle resides in Recent Window storage with **`1.00000000` bit-exact cosine similarity** (`storage_intact = True`). The cliff past 1.1x is an intrinsic model capability boundary of SmolLM-135M (which lacks RoPE frequency extension), not cache corruption.

---

### Canonical Perplexity (Multi-Sample Multi-Seed Evaluation)

Evaluated on `meta-llama/Llama-3.2-1B` over 1024 evaluation tokens on non-repeating natural language:

| Sample ID | Vanilla FP32 PPL | Config A (Grp16 / FP32 Meta) | Config A $\Delta$PPL | Config B (Grp32 / FP16 Meta) | Config B $\Delta$PPL | Quality Delta (B vs A) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sample #1** | 9.0617 | 9.0765 | +0.0147 (+0.16%) | 9.1423 | +0.0806 (+0.89%) | +0.0659 |
| **Sample #2** | 9.4801 | 9.4906 | +0.0105 (+0.11%) | 9.5195 | +0.0394 (+0.42%) | +0.0289 |
| **Sample #3** | 3.0903 | 3.1033 | +0.0130 (+0.42%) | 3.0814 | -0.0088 (-0.29%) | -0.0219 |
| **MEAN $\pm$ STD** | **7.2107 $\pm$ 2.9186** | **7.2235 $\pm$ 2.9183** | **+0.0128 (+0.18%)** | **7.2478 $\pm$ 2.9500** | **+0.0370 (+0.51%)** | **+0.0243 (+0.34%)** |

* **Metadata Precision Impact**: Switching metadata from FP32 to FP16 adds only **`+0.0129` PPL** on Sample 1 (9.0893 vs 9.0765) and $<0.015$ overall, halving metadata bytes with negligible quality degradation.

---

### Memory RSS & Physical Compression Ratios

Physical memory allocation measured after AVX2 kernel dynamic rewiring on `meta-llama/Llama-3.2-1B`:

| Context Length | Vanilla FP16 (MB) | Vanilla FP32 (MB) | TriTier Cache (MB) | Sink Bytes | RW Bytes | HH Bytes | PBS Payload Bytes | PBS Metadata Bytes | Compression vs FP16 | Compression vs FP32 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 256 | 8.0 | 16.0 | 16.00 | 262,144 | 16,515,072 | 0 | 0 | 0 | **0.50x** | **1.00x** |
| 512 | 16.0 | 32.0 | 19.95 | 262,144 | 16,777,216 | 1,708,928 | 925,696 | 1,243,392 | **0.80x** | **1.60x** |
| 1024 | 32.0 | 64.0 | 25.89 | 262,144 | 16,777,216 | 3,417,856 | 2,916,352 | 3,769,344 | **1.24x** | **2.47x** |
| 2048 | 64.0 | 128.0 | 37.76 | 262,144 | 16,777,216 | 6,769,984 | 6,901,760 | 8,887,936 | **1.69x** | **3.39x** |
| 4096 | 128.0 | 256.0 | 61.46 | 262,144 | 16,777,216 | 13,474,240 | 14,872,576 | 19,059,584 | **2.08x** | **4.17x** |
| 8192 | 256.0 | 512.0 | 109.03 | 262,144 | 16,777,216 | 26,948,480 | 30,810,112 | 39,532,800 | **2.35x** | **4.70x** |
| 16384 | 512.0 | 1024.0 | 204.06 | 262,144 | 16,777,216 | 53,896,960 | 62,685,184 | 80,348,160 | **2.51x** | **5.02x** |
| 32768 | 1024.0 | 2048.0 | 394.11 | 262,144 | 16,777,216 | 107,728,192 | 126,439,424 | 162,045,568 | **2.60x** | **5.20x** |

* **Reconciliation of Prior 3.21x Figure**: Prior to dynamic kernel rewiring, scripts evaluated theoretical Group-32 / FP16 formulas at context ~8192 ($\approx 3.21\text{x}$). Under Config A (Group-16 / FP32 metadata), metadata is $4\times$ larger ($39.53\text{ MB}$ vs $9.88\text{ MB}$ at 8k), yielding **`2.60x`** vs FP16 at 32k. Under Config B (Group-32 / FP16 metadata), asymptotic compression reaches **`3.55x`** vs FP16 (**`7.10x`** vs FP32).

---

### Autoregressive Generation & 4-Quadrant Root-Cause Attribution

Ablation isolating the Step 24 $\to$ 14 shift across 500 generated tokens from a 4096-token prompt on `meta-llama/Llama-3.2-1B`:

| Quadrant | Condition | RoPE Type | $K$ Group Size | Metadata Dtype | First Div Step | 500-Tok Agreement | Matching Tokens | Mean Logit Cosine |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | `a_neither_fix` | Old Clamped | 16 | FP32 | **Step 2** | **5.0%** | 25 / 500 | 0.6146 |
| **B** | `b_rope_fix_only` | New Corrected | 16 | FP32 | **Step 24** | **92.4%** | 462 / 500 | 0.9305 |
| **C** | `c_kernel_wiring_only` | Old Clamped | 32 | FP16 | **Step 2** | **0.2%** | 1 / 500 | 0.5373 |
| **D** | `d_both_fixes` | New Corrected | 32 | FP16 | **Step 14** | **91.6%** | 458 / 500 | 0.9159 |

* **Attribution**: The RoPE fix is 100% responsible for restoring generation from catastrophic divergence ($0.2\% \to 91.6\%$). Doubling group size to 32 shifts the first divergence from Step 24 to Step 14, with negligible change in overall token retention (**92.4% vs 91.6%**).
* **Intrinsic Quantization Fidelity**: Evaluated across 32 sequence chunks (2048 tokens), Key vector cosine similarity is **`0.956938`** (G16) vs **`0.936988`** (G32), a true delta of only **`-2.08%`** (Global Min: 0.8414 G16 vs 0.8161 G32). The $-30.2\%$ figure previously reported was strictly an autoregressive sequence divergence artifact following Step 7 argmax flip.


## Install

Requirements: Linux x86_64 with AVX2, FMA and BMI2; GCC 9 or newer with OpenMP; Python 3.10+; PyTorch 2.1+.

```bash
git clone https://github.com/RahulGPi/TriTierCache.git
cd TriTierCache

# Editable install, builds the C++ extension as part of the install
pip install -e .

# Verify the native extension loaded
python -c "import tri_tier._C as _C; print('extension ok:', _C.fused_attention_decode)"
```

Check your CPU has what the build needs:

```bash
lscpu | grep -o -E 'avx2|fma|bmi2' | sort -u
```

## Quickstart

### Patch a Hugging Face model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tri_tier.integration.patch_llama import apply_patch

apply_patch()

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cpu"
)

inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, use_cache=True)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Use the cache directly

```python
import torch
from tri_tier import TriTierCache

cache = TriTierCache(
    max_seq_len=2048,
    head_dim=128,
    num_heads=32,      # must be num_key_value_heads for GQA models
    R_size=256,
    H_ratio=0.05,
)

k_new = torch.randn(32, 128)
v_new = torch.randn(32, 128)
cache.ingest_token(k_new, v_new)

k_full, v_full, full_ids = cache.reconstruct_full_cache()

attn = torch.randn(1, 32, 1, k_full.shape[0]).softmax(dim=-1)
cache.accumulate_attn_scrs(attn, full_ids)
```

### Side-by-side demo

```bash
PYTHONPATH=src python example.py \
  --model HuggingFaceTB/SmolLM-135M \
  --prompt-tokens 512 \
  --new-tokens 32
```

## Repository layout

```text
.
├── csrc/                       # C++ AVX2 / OpenMP extension sources
│   ├── bindings.cpp            # pybind11 module -> tri_tier._C
│   ├── cache_engine.cpp        # native cache state and ring buffer management
│   ├── cpu/                    # quantize, dequantize, fused attention kernels
│   └── include/                # kernel headers
├── src/tri_tier/               # Python package
│   ├── cache.py                # TriTierCache
│   ├── cli.py                  # command line entry point
│   ├── constants.py            # SINK_SIZE, CHUNK_SIZE, R_size defaults
│   └── integration/
│       └── patch_llama.py      # LLaMA attention monkey-patch and fallback path
├── benchmarks/                 # latency, memory, perplexity, NIAH, ablations
│   └── results/                # exported CSVs
├── tests/                      # pytest suite plus standalone C++ kernel tests
├── example.py                  # end-to-end comparison demo
├── smoke_test.py               # fast end-to-end sanity check
└── setup.py / pyproject.toml   # build configuration
```

Full internals, quantization math, kernel contracts and the native ABI are documented in [`src/README.md`](src/README.md).

## Benchmarks

```bash
# Everything, quick mode
PYTHONPATH=. python benchmarks/run_all_benchmarks.py --quick

# Individual suites
PYTHONPATH=. python benchmarks/benchmark_correctness.py
PYTHONPATH=. python benchmarks/benchmark_latency.py
PYTHONPATH=. python benchmarks/benchmark_memory.py
PYTHONPATH=. python benchmarks/benchmark_perplexity.py

# Tier ablation grid
PYTHONPATH=. python benchmarks/run_isolation_grid.py

# Formatted report from the CSVs in benchmarks/results/
PYTHONPATH=. python benchmarks/generate_report.py
```

Outputs land in `benchmarks/results/` as CSV, one file per suite (`latency_results.csv`, `memory_rss_results.csv`, `perplexity_results.csv`, `niah_results.csv`, `thread_scaling_results.csv`, `ablation_results.csv`, and others).

## Tests

```bash
PYTHONPATH=. pytest -v
```

The Python suite covers buffer sizing, tier routing, attention scoring and the LLaMA patch including GQA head mapping. C++ kernels have their own standalone harnesses under `tests/`, checked against numpy-generated ground truth. See the src README for how to regenerate those fixtures.

## Configuration

| Parameter | Default | Supported / Options | Effect |
| :--- | :--- | :--- | :--- |
| `SINK_SIZE` | 4 | Integer | Pinned initial tokens. Raising it costs exact-tier memory and rarely helps. |
| `R_size` | 256 | Integer | Recent window size. The main quality lever. Larger means better local fidelity, more FP32 memory. |
| `H_ratio` | 0.05 | Float (0.0–1.0) | Fraction of `max_seq_len` kept as heavy hitters. |
| `CHUNK_SIZE` | 16 | Constant (16) | Tokens per packed block. Fixed by the 2-bit-into-int32 packing layout. Do not change. |
| `max_seq_len` | model dependent | Integer | Sizes the global attention tracker and the heavy-hitter budget. |
| `K_GROUP_SIZE` | 16 | `16`, `32` | Key quantization channel grouping across head dimension. `16` provides superior precision (0.957 K cosine similarity, Step 24 divergence); `32` halves scale/offset metadata storage (0.937 K cosine similarity, Step 14 divergence). |
| `PBS_METADATA_DTYPE` | `"fp16"` | `"fp16"`, `"fp32"` | Storage format for quantization scales and zero-points in the Packed Block Store. `"fp16"` halves metadata memory footprint with $<0.015$ PPL impact compared to `"fp32"`. |
| `ROPE_MODE` | `'a'` | `'a'`, `'b'` | Rotary position handling across cache tiers. Mode `'a'` retains absolute position IDs matching model pretraining (100% NIAH pass up to $2.0\times$ base on Llama-3.2-1B); Mode `'b'` clamps positions to recent window, causing query-key phase mismatch. |
| `score_decay` | 0.999 | Float (0.0–1.0) | Exponential decay factor for tracking cumulative heavy-hitter token attention scores. |

## Model support

| Model family | Status | Notes |
| :--- | :--- | :--- |
| LLaMA 3 / 3.2 | Supported | GQA handled natively in kernel (`head_dim=64, 128`), context to 131k |
| SmolLM / SmolLM2 | Supported | Primary benchmark targets (`head_dim=64`), MHA & GQA |
| TinyLlama | Supported | LLaMA GQA architecture (`head_dim=64`) |
| Qwen 2 / 2.5 / 3 | Supported | Qwen GQA architecture (`head_dim=64, 128`), context to 32k+ |
| Mistral / Mistral-Instruct | Supported | Mistral GQA architecture (`head_dim=128`), context to 32k+ |

## Roadmap

- Integer-domain attention using AVX-VNNI (`_mm256_dpbusd_epi32`), skipping FP32 reconstruction entirely
- Intel Xe-LP iGPU offload via DP4A under a zero-copy unified memory model
- Batch size greater than 1, and multi-token speculative decode
- Prefill path optimization to close the 1.22x TTFT gap
- Expose full per-head attention weights on the fused path

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

The tier design borrows from three lines of work: StreamingLLM for attention sinks, H2O for heavy-hitter eviction policy, and KIVI for asymmetric per-channel key and per-token value quantization. The routing between them, the packing layout and the fused CPU kernels are this project's own.
