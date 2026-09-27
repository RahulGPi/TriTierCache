# TriTierCache Benchmark Suite & Evaluation Results

Comprehensive evaluation results, methodology, and reproducibility guide for **TriTierCache** across downstream long-context accuracy, needle-in-a-haystack retrieval, perplexity, memory footprint, and decode latency.

---

## Contents

1. [Core Benchmark Summary](#1-core-benchmark-summary)
2. [Downstream Long-Context Task Accuracy](#2-downstream-long-context-task-accuracy)
3. [Graduated Needle-in-a-Haystack (NIAH) Sweeps](#3-graduated-needle-in-a-haystack-niah-sweeps)
   - [Llama-3.2-1B Graduated Sweep (8k–16k Tokens)](#llama-32-1b-graduated-sweep-8k16k-tokens)
   - [SmolLM-135M Graduated Sweep (2k–3k Tokens)](#smollm-135m-graduated-sweep-2k3k-tokens)
4. [Canonical Perplexity](#4-canonical-perplexity)
5. [Memory RSS & Physical Compression Ratios](#5-memory-rss--physical-compression-ratios)
6. [Autoregressive Agreement & Root-Cause Attribution](#6-autoregressive-agreement--root-cause-attribution)
7. [Running Benchmarks](#7-running-benchmarks)
   - [Full Benchmark Suite](#full-benchmark-suite)
   - [Downstream Accuracy Benchmark](#downstream-accuracy-benchmark)
   - [Individual Microbenchmarks](#individual-microbenchmarks)
   - [Results Directory](#results-directory)

---

## 1. Core Benchmark Summary

Measured on `meta-llama/Llama-3.2-1B` and `HuggingFaceTB/SmolLM-135M`, Linux x86_64, AVX2 + FMA + OpenMP. Baseline is Hugging Face uncompressed FP32 Attention (`DynamicCache`).

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

## 2. Downstream Long-Context Task Accuracy

Evaluated on long-context tasks (context $\ge 1024$ tokens) where $>75\%$ of KV tokens reside in Tier 3 (2-bit Packed Block Store). Baseline is uncompressed FP32 Vanilla Hugging Face Attention.

Tasks evaluated:
- **Long-Context QA**: Fact retrieval positioned at variable depth ratios with background filler paragraphs.
- **Multi-Variable Tracking**: Cross-variable state tracking across shuffled configuration assignments.
- **Many-Shot In-Context Learning (ICL)**: Many-shot demonstrations preceded by extensive context background text.

<!-- DOWNSTREAM_ACCURACY_START -->
### Downstream Long-Context Task Accuracy

Evaluated on long-context tasks (context $\ge 1024$ tokens) where $>75\%$ of KV tokens reside in Tier 3 (2-bit PBS). Baseline is uncompressed FP32 Vanilla Hugging Face Attention.

#### `SmolLM-135M` (LLAMA arch, 135M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **90.0%** / 90.0% | **10.0%** / 50.0% | **30.0%** / 30.0% | **73.3%** | 79.7% | **17.1 MB** / 45.0 MB (2.6x) | 1.03x (35.4 ms) |
| **2,048** | **90.0%** / 90.0% | **20.0%** / 20.0% | **40.0%** / 40.0% | **100.0%** | 75.6% | **23.9 MB** / 90.0 MB (3.8x) | 1.04x (35.2 ms) |

#### `TinyLlama-1.1B-Chat-v1.0` (LLAMA arch, 1100M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 100.0% | **80.0%** / 100.0% | **100.0%** / 100.0% | **93.3%** | 92.8% | **16.7 MB** / 44.0 MB (2.6x) | 1.01x (149.3 ms) |
| **2,048** | **80.0%** / 90.0% | **50.0%** / 100.0% | **100.0%** / 100.0% | **79.6%** | 84.8% | **23.3 MB** / 88.0 MB (3.8x) | 1.01x (157.4 ms) |

#### `SmolLM2-360M` (LLAMA arch, 362M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 100.0% | **80.0%** / 90.0% | **70.0%** / 70.0% | **96.3%** | 89.1% | **30.2 MB** / 80.0 MB (2.6x) | 1.00x (73.5 ms) |
| **2,048** | **100.0%** / 100.0% | **20.0%** / 90.0% | **60.0%** / 60.0% | **74.1%** | 73.1% | **42.2 MB** / 160.0 MB (3.8x) | 1.02x (81.0 ms) |
| **4,096** | **80.0%** / 100.0% | **50.0%** / 90.0% | **60.0%** / 60.0% | **78.5%** | 73.8% | **66.1 MB** / 320.0 MB (4.8x) | 0.99x (105.5 ms) |
| **8,192** | **70.0%** / 80.0% | **50.0%** / 70.0% | **60.0%** / 60.0% | **86.3%** | 65.3% | **114.0 MB** / 640.0 MB (5.6x) | 1.01x (163.6 ms) |

#### `SmolLM2-1.7B` (LLAMA arch, 1711M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 100.0% | **90.0%** / 100.0% | **100.0%** / 100.0% | **96.7%** | 90.1% | **144.4 MB** / 384.0 MB (2.7x) | 0.99x (286.0 ms) |
| **2,048** | **100.0%** / 100.0% | **90.0%** / 100.0% | **100.0%** / 100.0% | **96.7%** | 89.7% | **200.8 MB** / 768.0 MB (3.8x) | 1.03x (334.0 ms) |
| **4,096** | **100.0%** / 100.0% | **90.0%** / 100.0% | **90.0%** / 90.0% | **96.7%** | 90.3% | **313.4 MB** / 1536.0 MB (4.9x) | 1.00x (445.6 ms) |
| **8,192** | **90.0%** / 100.0% | **60.0%** / 90.0% | **100.0%** / 100.0% | **85.6%** | 84.4% | **539.4 MB** / 3072.0 MB (5.7x) | 1.33x (680.6 ms) |

#### `Qwen2.5-0.5B-Instruct` (QWEN2 arch, 494M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **90.0%** / 90.0% | **100.0%** / 100.0% | **90.0%** / 90.0% | **100.0%** | 75.1% | **9.2 MB** / 24.0 MB (2.6x) | 1.03x (98.3 ms) |
| **2,048** | **90.0%** / 80.0% | **80.0%** / 100.0% | **90.0%** / 90.0% | **97.5%** | 78.9% | **12.9 MB** / 48.0 MB (3.7x) | 1.02x (105.7 ms) |
| **4,096** | **100.0%** / 100.0% | **80.0%** / 100.0% | **70.0%** / 70.0% | **93.3%** | 68.9% | **20.3 MB** / 96.0 MB (4.7x) | 1.03x (136.9 ms) |
| **8,192** | **80.0%** / 90.0% | **80.0%** / 100.0% | **90.0%** / 90.0% | **89.6%** | 82.1% | **35.1 MB** / 192.0 MB (5.5x) | 1.00x (208.1 ms) |
| **16,384** | **90.0%** / 100.0% | **80.0%** / 100.0% | **80.0%** / 80.0% | **90.0%** | 65.5% | **64.8 MB** / 384.0 MB (5.9x) | 0.92x (194.8 ms) |

#### `Qwen2.5-1.5B-Instruct` (QWEN2 arch, 1544M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 90.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **103.7%** | 82.7% | **21.1 MB** / 56.0 MB (2.7x) | 1.00x (220.7 ms) |
| **2,048** | **90.0%** / 90.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 85.0% | **29.3 MB** / 112.0 MB (3.8x) | 1.00x (233.6 ms) |
| **4,096** | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 90.8% | **45.7 MB** / 224.0 MB (4.9x) | 0.98x (255.1 ms) |
| **8,192** | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 90.3% | **78.6 MB** / 448.0 MB (5.7x) | 0.98x (297.5 ms) |
| **16,384** | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 84.1% | **144.3 MB** / 896.0 MB (6.2x) | 0.88x (2434.7 ms) |

#### `Qwen3-0.6B` (QWEN3 arch, 596M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 90.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **103.7%** | 81.9% | **83.7 MB** / 224.0 MB (2.7x) | 1.00x (150.2 ms) |
| **2,048** | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 84.0% | **115.9 MB** / 448.0 MB (3.9x) | 1.02x (200.6 ms) |
| **4,096** | **100.0%** / 100.0% | **90.0%** / 100.0% | **100.0%** / 100.0% | **96.7%** | 82.7% | **180.1 MB** / 896.0 MB (5.0x) | 1.09x (250.2 ms) |
| **8,192** | **100.0%** / 100.0% | **80.0%** / 100.0% | **100.0%** / 100.0% | **93.3%** | 75.6% | **309.1 MB** / 1792.0 MB (5.8x) | 1.10x (430.2 ms) |
| **16,384** | **100.0%** / 100.0% | **90.0%** / 100.0% | **100.0%** / 100.0% | **96.7%** | 81.8% | **566.8 MB** / 3584.0 MB (6.3x) | 2.27x (1120.6 ms) |

#### `Llama-3.2-1B` (LLAMA arch, 1236M)

| Context | QA (TriTier / Base) | Tracking (TriTier / Base) | ICL (TriTier / Base) | Retention | Token Match | KV Memory (TriTier / Base) | Decode Speed |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,024** | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** | 82.4% | **24.1 MB** / 64.0 MB (2.6x) | 1.00x (176.6 ms) |
| **2,048** | **100.0%** / 100.0% | **90.0%** / 100.0% | **90.0%** / 90.0% | **93.3%** | 78.8% | **33.6 MB** / 128.0 MB (3.8x) | 1.00x (196.8 ms) |
| **4,096** | **90.0%** / 100.0% | **100.0%** / 100.0% | **100.0%** / 100.0% | **96.7%** | 77.0% | **52.6 MB** / 256.0 MB (4.9x) | 1.01x (219.4 ms) |
| **8,192** | **100.0%** / 100.0% | **100.0%** / 100.0% | **90.0%** / 100.0% | **96.7%** | 80.6% | **90.6 MB** / 512.0 MB (5.7x) | 1.05x (272.1 ms) |
| **16,384** | **80.0%** / 100.0% | **100.0%** / 100.0% | **90.0%** / 90.0% | **93.3%** | 79.0% | **166.7 MB** / 1024.0 MB (6.1x) | 1.34x (880.9 ms) |
<!-- DOWNSTREAM_ACCURACY_END -->

---

## 3. Graduated Needle-in-a-Haystack (NIAH) Sweeps

Evaluated across both `meta-llama/Llama-3.2-1B` (base context 8192) and `HuggingFaceTB/SmolLM-135M` (base context 2048) using a diverse, non-repeating corpus (1,198 strictly unique paragraphs, >33k tokens, zero repeated 8-grams).

### `meta-llama/Llama-3.2-1B` Graduated Sweep (8k–16k Tokens)

Evaluated across 9 context lengths spanning $1.0\times$ ($8192$) up to $2.0\times$ ($16384$), depths $0.20, 0.70, 0.95$, and $H_{\text{ratio}} \in \{0.05, 0.15\}$ against Vanilla FP32 baseline (135 total rows evaluated):

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

### `HuggingFaceTB/SmolLM-135M` Graduated Sweep (2k–3k Tokens)

Evaluated across 105 rows:
* **At 1.00x (2048 ctx)**: Vanilla FP32 = **100.0%** (3/3), TriTier Mode 'a' = **100.0%** (6/6).
* **At 1.10x (2252 ctx)**: Vanilla FP32 = **100.0%** (3/3), TriTier Mode 'a' depth 0.95 (RW) = **100.0%** (2/2).
* **At 1.20x–1.50x (2457–3072 ctx)**: Vanilla FP32 drops to **0.0%** and TriTier drops to **0.0%**, collapsing into identical filler tokens.
* **Model Attention Limit**: In all 35 rows at depth 0.95, the needle resides in Recent Window storage with **`1.00000000` bit-exact cosine similarity** (`storage_intact = True`). The cliff past 1.1x is an intrinsic model capability boundary of SmolLM-135M (which lacks RoPE frequency extension), not cache corruption.

---

## 4. Canonical Perplexity

Evaluated on `meta-llama/Llama-3.2-1B` over 1024 evaluation tokens on non-repeating natural language:

| Sample ID | Vanilla FP32 PPL | Config A (Grp16 / FP32 Meta) | Config A $\Delta$PPL | Config B (Grp32 / FP16 Meta) | Config B $\Delta$PPL | Quality Delta (B vs A) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sample #1** | 9.0617 | 9.0765 | +0.0147 (+0.16%) | 9.1423 | +0.0806 (+0.89%) | +0.0659 |
| **Sample #2** | 9.4801 | 9.4906 | +0.0105 (+0.11%) | 9.5195 | +0.0394 (+0.42%) | +0.0289 |
| **Sample #3** | 3.0903 | 3.1033 | +0.0130 (+0.42%) | 3.0814 | -0.0088 (-0.29%) | -0.0219 |
| **MEAN $\pm$ STD** | **7.2107 $\pm$ 2.9186** | **7.2235 $\pm$ 2.9183** | **+0.0128 (+0.18%)** | **7.2478 $\pm$ 2.9500** | **+0.0370 (+0.51%)** | **+0.0243 (+0.34%)** |

* **Metadata Precision Impact**: Switching metadata from FP32 to FP16 adds only **`+0.0129` PPL** on Sample 1 (9.0893 vs 9.0765) and $<0.015$ overall, halving metadata bytes with negligible quality degradation.

---

## 5. Memory RSS & Physical Compression Ratios

<div align="center">

<img src="../benchmarks_kv_memory.svg" alt="KV Cache Memory Footprint Comparison: Vanilla Baseline vs. TriTierCache across Context Lengths" width="850"/>

</div>

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

## 6. Autoregressive Agreement & Root-Cause Attribution

Ablation isolating the Step 24 $\to$ 14 shift across 500 generated tokens from a 4096-token prompt on `meta-llama/Llama-3.2-1B`:

| Quadrant | Condition | RoPE Type | $K$ Group Size | Metadata Dtype | First Div Step | 500-Tok Agreement | Matching Tokens | Mean Logit Cosine |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | `a_neither_fix` | Old Clamped | 16 | FP32 | **Step 2** | **5.0%** | 25 / 500 | 0.6146 |
| **B** | `b_rope_fix_only` | New Corrected | 16 | FP32 | **Step 24** | **92.4%** | 462 / 500 | 0.9305 |
| **C** | `c_kernel_wiring_only` | Old Clamped | 32 | FP16 | **Step 2** | **0.2%** | 1 / 500 | 0.5373 |
| **D** | `d_both_fixes` | New Corrected | 32 | FP16 | **Step 14** | **91.6%** | 458 / 500 | 0.9159 |

* **Attribution**: The RoPE fix is 100% responsible for restoring generation from catastrophic divergence ($0.2\% \to 91.6\%$). Doubling group size to 32 shifts the first divergence from Step 24 to Step 14, with negligible change in overall token retention (**92.4% vs 91.6%**).
* **Intrinsic Quantization Fidelity**: Evaluated across 32 sequence chunks (2048 tokens), Key vector cosine similarity is **`0.956938`** (G16) vs **`0.936988`** (G32), a true delta of only **`-2.08%`** (Global Min: 0.8414 G16 vs 0.8161 G32). The $-30.2\%$ figure previously reported was strictly an autoregressive sequence divergence artifact following Step 7 argmax flip.

---

## 7. Running Benchmarks

### Full Benchmark Suite

Run the full end-to-end benchmark suite:

```bash
# Run all benchmark suites in quick mode
PYTHONPATH=. python benchmarks/run_all_benchmarks.py --quick

# Generate formatted summary report from exported CSVs
PYTHONPATH=. python benchmarks/generate_report.py
```

### Downstream Accuracy Benchmark

Run the downstream long-context task accuracy suite across target models and context lengths:

```bash
# Evaluate specific model
PYTHONPATH=. python benchmarks/benchmark_accuracy.py \
    --models Qwen/Qwen3-0.6B \
    --context-lens 1024 2048 \
    --samples-per-task 10 \
    --update-readme

# Evaluate model tiers (e.g. tier 1, tier 2, tier 3, or all)
PYTHONPATH=. python benchmarks/benchmark_accuracy.py --tier 3 --context-lens 1024 2048
```

### Individual Microbenchmarks

```bash
# Correctness verification (FP32 lossless vs TriTier AVX2 fused path)
PYTHONPATH=. python benchmarks/benchmark_correctness.py

# Single-token decode and kernel execution latency
PYTHONPATH=. python benchmarks/benchmark_latency.py

# Physical memory RSS and per-tier footprint measurement
PYTHONPATH=. python benchmarks/benchmark_memory.py

# Canonical multi-sample perplexity evaluation
PYTHONPATH=. python benchmarks/benchmark_perplexity.py

# Multi-tier ablation grid across window sizes and heavy hitter ratios
PYTHONPATH=. python benchmarks/run_isolation_grid.py

# Graduated non-repeating NIAH sweep
PYTHONPATH=. python benchmarks/run_llama_nonrepeating_graduated_niah.py
```

### Results Directory

All benchmark scripts export raw, structured data to `benchmarks/results/`:

| Output CSV | Contents |
| :--- | :--- |
| `accuracy_results.csv` | Downstream task accuracy (QA, Tracking, ICL), EM, retention, KV memory, speedup |
| `latency_results.csv` | Decode latency and throughput across context lengths |
| `memory_rss_results.csv` | Physical process RSS and per-tier memory breakdowns |
| `perplexity_results.csv` | Multi-sample perplexity and cross-configuration comparisons |
| `llama_graduated_niah_results.csv` | Needle-in-a-haystack retrieval pass rates and cosine similarity |
| `thread_scaling_results.csv` | OpenMP multi-threaded kernel scaling (1 to 8 threads) |
| `ablation_results.csv` | Recent window and heavy-hitter ratio hyperparameter sweeps |
