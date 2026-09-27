# TriTierCache Internals

Implementation reference for the Python package and the native AVX2 extension. This document assumes you have read the [project README](../README.md) and want the buffer layouts, the quantization math, the kernel contracts and the ABI.

**Contents**

1. [Scope and constraints](#1-scope-and-constraints)
2. [Buffer inventory](#2-buffer-inventory)
3. [Token routing state machine](#3-token-routing-state-machine)
4. [Quantization math](#4-quantization-math)
5. [Bit-packing layout](#5-bit-packing-layout)
6. [Compression accounting](#6-compression-accounting)
7. [Native kernels](#7-native-kernels)
8. [Fused decode attention](#8-fused-decode-attention)
9. [Python to C++ ABI](#9-python-to-c-abi)
10. [Integration layer & Multi-model patching](#10-integration-layer--multi-model-patching)
11. [Configuration parameters](#11-configuration-parameters)
12. [Build](#12-build)
13. [Numerical contract](#13-numerical-contract)
14. [Testing methodology](#14-testing-methodology)
15. [Known limitations](#15-known-limitations)
16. [Extension points](#16-extension-points)

---

## 1. Scope and constraints

| Property | Value |
| :--- | :--- |
| Execution mode | Decode, `batch_size = 1`, `q_len = 1` |
| Prefill | Supported via batched ingestion, not via the fused kernel |
| Device | CPU only. FP32 only. |
| Vector ISA | 256-bit AVX2, FMA3, BMI2. No AVX-512 (disabled on Alder Lake hybrid client parts) |
| Threading | OpenMP over query heads |
| Attention variant | MHA and GQA. MQA works as a degenerate GQA case. |
| Positional encoding | RoPE is applied by the host model **before** `cache.update(...)`, so cached K is already rotated. Attention is therefore order-invariant across tiers and no position IDs are threaded into the kernel. |

The order-invariance point is load-bearing. It is why the fused path can read Sinks, RW, HH and PBS in whatever order is cheapest, and why `index_add_` scatter of attention scores does not need sorted token IDs.

---

## 2. Buffer inventory

`H = num_key_value_heads`, `D = head_dim`, `C = CHUNK_SIZE = 16`, `Dq = ceil(D / 16)`.

| Buffer | Shape | Dtype | Notes |
| :--- | :--- | :--- | :--- |
| `S_K_Buffer`, `S_V_Buffer` | `[4, H, D]` | float32 | Attention sinks. Written once, never evicted. |
| `RW_K_Buffer`, `RW_V_Buffer` | `[R_size, H, D]` | float32 | Recent window ring buffer, indexed by `RW_head_index`. |
| `RW_token_ids` | `[R_size]` | int64 | Absolute token ID per ring slot. |
| `HH_K_Buffer`, `HH_V_Buffer` | `[max_heavy_hitters, H, D]` | float32 | Heavy hitters. |
| `HH_token_ids` | `[max_heavy_hitters]` | int64 | `-1` marks a free slot. |
| `HH_scores` | `[max_heavy_hitters]` | float32 | Snapshot of `Global_Attn_Scr` at promotion time. |
| `WR_K_Buffer`, `WR_V_Buffer` | `[16, H, D]` | float32 | Waiting room. Flushed to PBS when full. |
| `PBS_K_Packed` | `[num_blocks, H, D]` | int32 | 16 tokens packed along the token axis, per channel. |
| `PBS_K_Scales`, `PBS_K_Zeroes` | `[num_blocks, H, D]` | float32 | Per-channel, per-block. |
| `PBS_V_Packed` | `[num_blocks * 16, H, Dq]` | int32 | 16 channels packed per int32, per token. |
| `PBS_V_Scales`, `PBS_V_Zeroes` | `[num_blocks * 16, H, 1]` | float32 | Per-token, per-head. |
| `PBS_token_ids` | `[num_blocks * 16]` | int64 | Absolute token ID per packed slot. |
| `Global_Attn_Scr` | `[max_seq_len]` | float32 | Cumulative head-averaged attention weight per token ID. |

**Sizing rule.** The V-side buffers are sized `num_blocks * CHUNK_SIZE`, not `max_background_tokens`. Using the latter silently under-allocates whenever `max_background_tokens` is not an exact multiple of 16, and the overflow lands in whatever tensor sits next in memory. This was a real bug, fixed, and the test suite pins it.

**GQA sizing rule.** The cache is sized by `config.num_key_value_heads`, never `config.num_attention_heads`. On transformers 5.16.1, `LlamaAttention.num_heads` does not exist, so do not reach for it.

---

## 3. Token routing state machine

```text
  new token (K, V)
        │
        ├─ token_id < SINK_SIZE ──────────────────────────► Sink buffer (permanent)
        │
        └─ otherwise ────────────────────────────────────► RW ring slot (RW_head_index)
                                                                 │
                                        evicted token from ring  │
                                                                 ▼
                                              score = Global_Attn_Scr[token_id]
                                                                 │
                            ┌────────────────────────────────────┴───────────────────┐
                            │                                                        │
              score beats weakest HH                                    score does not qualify
                            │                                                        │
                            ▼                                                        ▼
                  write into HH slot                                      Waiting Room slot
                  displaced HH token ──────────────────────────────────►  Waiting Room slot
                                                                                     │
                                                                      16 tokens staged
                                                                                     ▼
                                                        quantize_k_avx2 + quantize_v_avx2
                                                                                     ▼
                                                              PBS block (ring, wraps for
                                                              unbounded generation)
```

Attention feedback closes the loop. After each decode step, `accumulate_attn_scrs` adds the head-averaged softmax weights into `Global_Attn_Scr` by absolute token ID, which is what the eviction comparison reads on the next round. Cumulative score, not instantaneous, so a token that mattered once does not survive forever on that alone once its relative weight decays.

---

## 4. Quantization math

Both schemes are asymmetric 2-bit, 4 levels, zero point stored as the group minimum.

### Keys, per-channel across a 16-token block

For block `b`, head `h`, channel `c`, over `t` in `[0, 15]`:

```
min_{h,c}   = min_t  K[t, h, c]
max_{h,c}   = max_t  K[t, h, c]
scale_{h,c} = max(max_{h,c} - min_{h,c}, 1e-9) / 3.0
q[t,h,c]    = clamp( round_half_even( (K[t,h,c] - min_{h,c}) / scale_{h,c} ), 0, 3 )
```

Reduction axis is the token axis, so each of the `D` channels gets its own scale and zero. This is the KIVI observation: key channels have wildly different dynamic ranges and outlier channels are persistent, so per-channel grouping keeps them from poisoning the whole block.

### Values, per-token across channels

For token `t`, head `h`, over `c` in `[0, D-1]`:

```
min_{t,h}   = min_c  V[t, h, c]
max_{t,h}   = max_c  V[t, h, c]
scale_{t,h} = max(max_{t,h} - min_{t,h}, 1e-9) / 3.0
q[t,h,c]    = clamp( round_half_even( (V[t,h,c] - min_{t,h}) / scale_{t,h} ), 0, 3 )
```

Values are attention-weighted and summed, so per-token grouping bounds the error each token can contribute to the output independently.

### Dequantization

```
x = q * scale + zero          // zero == the stored group minimum
```

One FMA per element, `_mm256_fmadd_ps(q_f32, scale_vec, zero_vec)`.

### Two things the spec document gets wrong

The architecture spec in `tri_tier_cache.txt` writes the quantizer as `floor(|x - min| / scale + 0.5)`. The implementation does neither the absolute value nor the `+0.5`.

- **Order.** Divide first, then clamp. Clamping before the divide changes results near the range edges.
- **Rounding.** `_mm256_cvtps_epi32` under default MXCSR rounds half to even. Adding `0.5` and truncating gives half-away-from-zero, which disagrees on exact ties and produces off-by-one packed nibbles. The Python reference uses `torch.round`, which is also half-to-even. They match. The `+0.5` form does not.

Where the spec and the Python source disagree, the Python source wins. It has been the correct one twice.

---

## 5. Bit-packing layout

Two bits per value, 16 values per `int32`, little-endian by index.

**Keys**, packing along the token axis:

```
Packed_K[b, h, c] = Σ_{t=0}^{15} ( q[t, h, c] << (2t) )
```

**Values**, packing along the channel axis in groups of 16. With `g = c / 16` and `i = c % 16`:

```
Packed_V[t, h, g] = Σ_{i=0}^{15} ( q[t, h, 16g + i] << (2i) )
```

Unpacking, for lane index `j`:

```
q = (packed >> (2j)) & 0x3
```

**On `_pdep_u32`.** It looks like the right instruction for this and it is not. PDEP is scalar, so using it forces a gather out of vector registers, a scalar deposit, and a scatter back, at which point you have spent more on data movement than the packing saves. Two approaches that do work:

- Fixed shift amounts, known at compile time, vectorized OR-accumulate across an unrolled loop. This is what the K kernel does since the shift `2t` is uniform across a lane.
- `_mm256_sllv_epi32` for variable per-lane shifts. This is what the V kernel does since each lane in the vector holds a different channel index `i` and therefore needs a different shift.

---

## 6. Compression accounting

For `head_dim = 128`, FP32 baseline at 32 bits per element.

**Keys.** Per block, per head, per channel: one int32 of packed data plus one FP32 scale plus one FP32 zero, covering 16 token values.

```
(32 + 32 + 32) bits / 16 values = 6.00 bits/value   ->  5.33x
```

**Values.** Per token, per head: `Dq = 8` int32 words covering 128 channels, plus one scale and one zero for the whole token.

```
(8*32 + 32 + 32) bits / 128 values = 2.50 bits/value  ->  12.80x
```

**Combined K + V.**

```
(6.00 + 2.50) / 2 = 4.25 bits/value  ->  7.53x
```

7.53x is the Tier 3 storage ratio, measured at realistic shapes. It is not the end-to-end cache ratio, which is lower because Sinks, RW and HH stay FP32. End-to-end sits near 3.86x at 32k context and approaches 7.5x asymptotically as the background tier dominates. Anyone quoting "16x" from the 2-bit-versus-32-bit arithmetic is ignoring the scale and zero metadata; that framing was wrong and is not used here.

---

## 7. Native kernels

All under `csrc/cpu/`, headers in `csrc/include/`.

### `quantize_k_avx2.cpp`

- Input `[16, H, D]` FP32, contiguous.
- Output: `packed [H, D]` int32, `scales [H, D]` FP32, `zeroes [H, D]` FP32.
- Pass 1: loop-accumulated `_mm256_min_ps` / `_mm256_max_ps` down the token axis, 8 channels per register.
- Pass 2: `_mm256_sub_ps`, multiply by reciprocal scale, `_mm256_cvtps_epi32`, clamp to `[0, 3]` with `_mm256_min_epi32` / `_mm256_max_epi32`.
- Pack: shift amount `2t` is uniform across the register, so a compile-time-constant `_mm256_slli_epi32` plus `_mm256_or_si256` accumulate over the unrolled 16-iteration loop.

### `quantize_v_avx2.cpp`

- Input `[16, H, D]` FP32, contiguous.
- Output: `packed [16, H, Dq]` int32, `scales [16, H, 1]` FP32, `zeroes [16, H, 1]` FP32.
- Pass 1: horizontal reduction across `D` per `(token, head)` using `hmin8` / `hmax8` helpers, which fold 256 bits down to a scalar through `_mm256_extractf128_ps` then two `_mm_shuffle_ps` stages.
- Pass 2: broadcast the scalar scale and zero back to a vector, quantize 8 channels at a time.
- Pack: each lane holds a different within-group index `i`, so the shift is per-lane. `_mm256_sllv_epi32`, then `hor32` to horizontally OR the eight lanes into the final int32.

The fold helpers are where bugs live. A previous revision of `hmax8` used `_mm_min_ps` on the first fold step, which produced a correct-looking result whenever the true maximum happened to land in the lower half of the register and a wrong one otherwise. Intermittent, data-dependent, and invisible to any test that only uses random uniform input. Both helper pairs are now unit-tested independently.

### `dequantize_avx2.cpp`

- `dequantize_k_avx2` and `dequantize_v_avx2`. Structural mirror of the quantizers: unpack and FMA instead of round and pack.
- Unpack: `_mm256_srlv_epi32` (or `_mm256_srli_epi32` for the uniform-shift key case) then mask with `0x3`.
- Convert: `_mm256_cvtepi32_ps`.
- Dequantize: `_mm256_fmadd_ps(val, scale, zero)`.
- Writes directly into a caller-provided output buffer. No intermediate allocation.
- Requires `-mfma`.

### `fused_attn_avx2.cpp`

See section 8.

### `cache_engine.cpp`

Native cache state and ring buffer index management, exposed as `TriTierCacheEngine`. Keeps head indices, block cursors and token ID arrays on the C++ side so the decode loop does not pay for Python-side slice churn.

### `microbenchmark.cpp`

Standalone timing harness for individual kernels. Not part of the extension build.

---

## 8. Fused decode attention

`fused_attention_decode_avx2` computes the entire decode-step attention in two passes without ever materializing a dequantized cache.

**Pass A, scores.**

For each query head `qh`, mapped to KV head `kvh = qh / group_size`:

1. Dot `Q[qh]` against the dense tiers directly: Sinks, then RW, then occupied HH slots. AVX2 8-way unrolled FMA accumulate.
2. For each PBS block, dequantize the block into a small per-thread scratch buffer, dot against it, discard. Block by block, so the working set stays in L2.
3. Track `row_max` while accumulating.

**Softmax.** Numerically stable, subtract `row_max`, exponentiate, accumulate `row_sum_exp`, normalize.

**Pass B, values.**

Same tier walk, same block streaming, accumulating `axpy_avx2(out, V_row, weight)` into the output vector.

**Threading.** `#pragma omp parallel for` over query heads. Each thread owns a private scratch buffer, so there is no sharing on the hot path. Measured 3.56x going from 1 to 8 threads on an 8-thread part.

**Memory behaviour.** At 32 heads, `head_dim = 128`, 3200 background tokens, peak scratch is 256 KB against 104.9 MB for the materialize-then-attend approach, roughly 410x less. Correctness on that configuration was 0 mismatches in 4096 output elements against the full-materialization reference.

**Attention feedback.** The kernel does not expose per-head attention weights, because it normalizes and consumes them inside Pass B. It returns `mean_attn_weights`, the softmax weight averaged over query heads, which is exactly what the heavy-hitter router needs. Anything wanting full per-head weights has to use the reference path.

---

## 9. Python to C++ ABI

`csrc/bindings.cpp` exposes `tri_tier._C` through pybind11. The entry points take **raw pointers** obtained from `.data_ptr()`, not `torch::Tensor`. This keeps the extension buildable without libtorch C++ headers and without ABI-matching against the installed PyTorch build.

The cost is that the C++ side performs no validation. It cannot see dtype, shape, strides or device. The caller owns every one of these invariants:

| Invariant | Consequence if violated |
| :--- | :--- |
| `dtype == torch.float32` (or `int32` for packed) | Silent garbage. Reinterpreted bit patterns. |
| `.is_contiguous()` | Silent garbage. Stride assumptions are hardcoded. |
| `.device.type == "cpu"` | Segfault. |
| Shapes match the documented layout exactly | Out-of-bounds write into adjacent tensor memory. |
| Tensors stay alive across the call | Use after free. |

Add the assertions on the Python side before every `.data_ptr()` call. They are cheap relative to a decode step and they turn a memory-corruption class of bug into an exception. This is currently an open item in `patch_llama.py`.

---

## 10. Integration layer & Multi-model patching

TriTierCache provides two integration interfaces:
- `src/tri_tier/integration/patch_model.py`: Universal multi-model monkey-patcher supporting LLaMA, Mistral, and Qwen architectures.
- `src/tri_tier/integration/patch_llama.py`: Specialized monkey-patcher targeting `LlamaAttention`.

### Supported Model Architectures

| Architecture Class | Supported Families | Key Architectural Details |
| :--- | :--- | :--- |
| `LlamaAttention` | LLaMA 3, 3.1, 3.2, SmolLM, SmolLM2, TinyLlama | Standard RoPE, MHA / GQA (`head_dim=64, 128`) |
| `MistralAttention` | Mistral-7B, Mistral-Instruct | Standard RoPE, GQA (`head_dim=128`), sliding window compatibility |
| `Qwen2Attention` | Qwen 2, Qwen 2.5 | Standard RoPE, GQA (`head_dim=64, 128`), context capacity 32k+ |
| `Qwen3Attention` | Qwen 3 (e.g., `Qwen3-0.6B`) | Per-head Q-K RMS normalization (`q_norm`, `k_norm`), 3D RoPE unsqueezing |

### Execution Flow in `patched_forward`

1. **Lazy Initialization**: Upon the first forward pass through an attention module, a per-layer `TriTierCache` instance is lazily allocated and stored on the module as `self.tri_tier_cache`, sized to `max(32768, max_position_embeddings)`.
2. **Projection & Q-K Normalization**:
   ```python
   q_proj = self.q_proj(hidden_states).view(bsz, q_len, num_q_heads, head_dim)
   k_proj = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
   v_proj = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)

   # Per-head RMSNorm (Qwen 3 architectural requirement)
   if hasattr(self, "q_norm") and self.q_norm is not None:
       q_proj = self.q_norm(q_proj)
   if hasattr(self, "k_norm") and self.k_norm is not None:
       k_proj = self.k_norm(k_proj)
   ```
3. **RoPE Rotary Application**:
   RoPE is applied upstream to both $Q$ and newly arrived $K$. When position embeddings `cos`, `sin` are 3D (`[batch, seq_len, head_dim]`), they are unsqueezed to 4D (`[batch, 1, seq_len, head_dim]`) to broadcast correctly across all attention heads.
4. **Prefill (`q_len > 1`)**:
   Runs standard scaled dot-product attention (`torch.nn.functional.scaled_dot_product_attention`) across all prompt tokens for prompt ingestion, then bulk-ingests $K$ and $V$ via `cache.prefill(K_tokens.float(), V_tokens.float())`.
5. **Decode (`q_len == 1`)**:
   - **Fused Path (extension present, `cache._engine is not None`)**: Bypasses full cache reconstruction. Calls `cache.step(Q_flat, K_flat, V_flat, attn_output)`, passing pointers directly to `fused_attention_decode_avx2`. GQA mapping is handled natively inside the kernel via `kvh = qh / group_size`.
   - **Reference Path (fallback or `output_attentions=True`)**: Ingests new $K, V$ into the Python cache via `cache.ingest_token(...)`, calls `cache.reconstruct_full_cache()`, repeats GQA heads along the KV dimension, runs PyTorch scaled dot-product, and updates cumulative attention scores via `cache.accumulate_attn_scrs(...)`.

### Patch Management API

```python
from tri_tier.integration.patch_model import apply_patch, remove_patch, reset_caches, is_patched

apply_patch()        # Monkey-patches all supported attention classes
# ... run generation or benchmarks ...
reset_caches(model)  # Frees per-layer TriTierCache instances
remove_patch()       # Restores original Hugging Face attention forward methods
```

---

## 11. Configuration parameters

All parameters can be tuned in `src/tri_tier/constants.py` or passed directly to `TriTierCache(...)`:

| Parameter | Default | Supported / Options | Effect |
| :--- | :--- | :--- | :--- |
| `SINK_SIZE` | 4 | Integer | Pinned initial tokens. Anchors attention distribution; raising it costs exact-tier memory and rarely helps. |
| `R_size` | 256 | Integer | Recent window size (ring buffer). The primary quality lever. Larger window improves local fidelity at the expense of FP32 memory. |
| `H_ratio` | 0.05 | Float (0.0–1.0) | Fraction of sequence context preserved in FP32 heavy-hitter store (Tier 2). |
| `CHUNK_SIZE` | 16 | Constant (16) | Number of tokens per packed PBS block. Fixed by the 2-bit-into-int32 packing format. Do not change. |
| `max_seq_len` | 32768 | Integer | Maximum context length capacity. Sizes the global attention tracker and heavy-hitter budget. |
| `K_GROUP_SIZE` | 16 | `16`, `32` | Key quantization channel grouping across head dimension. `16` delivers superior precision (0.957 K cosine similarity, Step 24 divergence); `32` halves scale/offset metadata storage (0.937 K cosine similarity, Step 14 divergence). |
| `PBS_METADATA_DTYPE` | `"fp16"` | `"fp16"`, `"fp32"` | Storage format for quantization scales and zero-points in the Packed Block Store. `"fp16"` halves metadata memory footprint with $<0.015$ PPL impact compared to `"fp32"`. |
| `ROPE_MODE` | `'a'` | `'a'`, `'b'` | Rotary position handling across cache tiers. Mode `'a'` retains absolute position IDs matching model pretraining (100% NIAH pass up to $2.0\times$ base on Llama-3.2-1B); Mode `'b'` clamps positions to recent window, causing query-key phase mismatch. |
| `score_decay` | 0.999 | Float (0.0–1.0) | Exponential decay factor for tracking cumulative heavy-hitter token attention scores. Prevents transiently hot tokens from remaining in Tier 2 indefinitely. |

---

## 12. Build

```bash
pip install -e .            # builds the extension via setup.py
```

Required flags, set in `setup.py`:

| Flag | Needed by |
| :--- | :--- |
| `-O2` or `-O3` | everything |
| `-mavx2` | all kernels |
| `-mbmi2` | bit manipulation paths |
| `-mfma` | `dequantize_avx2.cpp`, `fused_attn_avx2.cpp`. Omitting it fails the build on `_mm256_fmadd_ps`. |
| `-fopenmp` | `fused_attn_avx2.cpp`. Also needs `-lgomp` at link. |

Standalone kernel compile for debugging:

```bash
g++ -O2 -mavx2 -mbmi2 -mfma -fopenmp \
    -Icsrc/include csrc/cpu/quantize_k_avx2.cpp tests/test_quantize_k.cpp -o /tmp/test_k
/tmp/test_k
```

Thread count at runtime:

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/benchmark_latency.py
```

---

## 13. Numerical contract

| Property | Value | Why it matters |
| :--- | :--- | :--- |
| Rounding | Half to even, default MXCSR via `_mm256_cvtps_epi32` | Matches `torch.round`. Half-away-from-zero does not. |
| Scale epsilon | `1e-9` | Guards constant-valued blocks against divide by zero. |
| Level count | 4, divisor `3.0` | 2-bit asymmetric. |
| Clamp | `[0, 3]`, after the divide | Clamping before the divide is a different function. |
| Zero point | Group minimum, stored FP32 | Not a quantized zero point. No zero-point packing. |
| Accumulation | FP32 throughout | No FP16 or bf16 anywhere in the kernels. |

Scale underflow is tracked as a standing check, `benchmarks/check_scale_underflow.py`, because a block whose range collapses to near the epsilon produces a scale so small that dequantized values saturate.

---

## 14. Testing methodology

**Ground truth generation.** Two-script pipeline per kernel. `gen_ref_*.py` produces numpy ground truth. `gen_test_*.py` emits a compilable C++ test with expected values. For anything above toy scale, fixtures go to raw binary files read through ctypes rather than inline float arrays, because multi-megabyte literal arrays make the compiler unusable.

**Tie-breaking regression.** Test inputs deliberately include values landing exactly on `.5` quantization boundaries. This is not optional. Random uniform data almost never hits an exact tie in FP32, so a rounding-convention bug passes every randomized test and then shows up as a one-level error on real activations. Keep the `.5` cases in the suite permanently.

**Diagnostic order.** When a kernel test fails, read the hex pattern of the mismatched packed values before reading the code. The failure shape identifies the bug class directly: every lane wrong points at scale or zero computation, alternating lanes wrong points at shift amounts, one half of the register wrong points at a horizontal fold helper.

**Python suite.**

```bash
PYTHONPATH=src pytest -v
```

`test_cache_init.py` covers buffer sizing including the `num_blocks * CHUNK_SIZE` rule. `test_cache_methods.py` covers ingestion, tier routing, promotion and demotion, and score accumulation. `test_patch_llama.py` and `test_multi_model_patch.py` cover patching and GQA head mapping across LLaMA, Mistral, and Qwen.

---

## 15. Known limitations

| Issue | Impact | Status |
| :--- | :--- | :--- |
| `V_Grouped.reshape(..., quant_head_dim, 16)` assumes `head_dim % 16 == 0` | Silent wrong reshape for other head dims. Dormant at 64 and 128. | Open, needs an assert at minimum |
| No dtype, contiguity or device assertions before `.data_ptr()` | Memory corruption instead of an exception | Open |
| Fused path returns `attn_weights=None` | Breaks `output_attentions=True`, attention hooks, any logging of per-head weights | Open by design, needs documenting at the API level |
| Prefill is 1.22x slower than vanilla | Routing and quantization on the ingestion path | Open, not yet profiled |
| Batch size fixed at 1 | No batched serving | Out of scope currently |
| PPL rises 6.10 to 7.73 at 2048 context | Quality cost of 2-bit background storage | Accepted tradeoff, tunable via `R_size` and `H_ratio` |

---

## 16. Extension points

**AVX-VNNI integer attention.** `_mm256_dpbusd_epi32` computes `Q · K_quant` in the integer domain with no FP32 reconstruction. Held, not abandoned. The blockers are real: there is no Python reference implementation to verify against, the Q-quantization scheme and the rescale math are open design questions, and for a `batch=1` decode workload the kernel is probably memory-bound rather than compute-bound, which means the payoff may be small. Build the Python reference first, then the kernel.

**Xe-LP iGPU offload.** DP4A on 80 to 96 EUs under a zero-copy UMA model, sharing host RAM with no PCIe transfer. Natural fit for the packed tier since the data is already INT8-shaped. Needs a SYCL or Level Zero path.

**Adding a kernel.** Header in `csrc/include/`, implementation in `csrc/cpu/`, binding in `csrc/bindings.cpp`, source entry in `setup.py`, numpy reference plus tie-boundary fixtures in `tests/`. Verify bit-exact against the reference before touching the Python side.

---

## Appendix A: Symbol reference

| Symbol | Meaning |
| :--- | :--- |
| `S` | Attention sink count, fixed at 4 |
| `R_size` | Recent window capacity, default 256 |
| `H_ratio` | Heavy-hitter budget as a fraction of `max_seq_len`, default 0.05 |
| `C`, `CHUNK_SIZE` | Tokens per packed block, fixed at 16 |
| `PBS` | Packed Block Storage, Tier 3 |
| `WR` | Waiting Room, the pre-quantization staging buffer |
| `Dq`, `quant_head_dim` | `ceil(head_dim / 16)`, int32 words per token per head on the V side |
| `group_size` | `num_attention_heads / num_key_value_heads` |

## Appendix B: Reserved

Space for future sections. Suggested additions as the work lands: iGPU kernel contracts, quantized-Q rescale derivation, batched decode layout, per-layer tier budgets.
