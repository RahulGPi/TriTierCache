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
10. [Integration layer](#10-integration-layer)
11. [Build](#11-build)
12. [Numerical contract](#12-numerical-contract)
13. [Testing methodology](#13-testing-methodology)
14. [Known limitations](#14-known-limitations)
15. [Extension points](#15-extension-points)

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

## 10. Integration layer

`src/tri_tier/integration/patch_llama.py` monkey-patches `LlamaAttention.forward`.

**Fused path** (extension present). Bypasses `reconstruct_full_cache()` entirely. Pulls Sink, RW and HH tiers plus the PBS buffers directly and hands pointers to the kernel. Valid because attention is order-invariant once RoPE has been applied upstream.

**Reference path** (`_reference_attention_path`, extension absent). Calls `reconstruct_full_cache()`, which returns token-major `[total_len, H, D]`, then applies the GQA repeat and `.transpose(0, 1).unsqueeze(0)` to reach the `[1, num_q_heads, total_len, D]` layout attention expects. Note the layout is token-major, not head-major. Getting this backwards produces plausible-looking output with silently scrambled heads.

**GQA.** Handled inside the kernel by index mapping, `kvh = qh / group_size`. It is deliberately not handled by `repeat_interleave` before the call, because that duplicates the entire KV tensor in memory and defeats the point of a streaming design.

---

## 11. Build

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

## 12. Numerical contract

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

## 13. Testing methodology

**Ground truth generation.** Two-script pipeline per kernel. `gen_ref_*.py` produces numpy ground truth. `gen_test_*.py` emits a compilable C++ test with expected values. For anything above toy scale, fixtures go to raw binary files read through ctypes rather than inline float arrays, because multi-megabyte literal arrays make the compiler unusable.

**Tie-breaking regression.** Test inputs deliberately include values landing exactly on `.5` quantization boundaries. This is not optional. Random uniform data almost never hits an exact tie in FP32, so a rounding-convention bug passes every randomized test and then shows up as a one-level error on real activations. Keep the `.5` cases in the suite permanently.

**Diagnostic order.** When a kernel test fails, read the hex pattern of the mismatched packed values before reading the code. The failure shape identifies the bug class directly: every lane wrong points at scale or zero computation, alternating lanes wrong points at shift amounts, one half of the register wrong points at a horizontal fold helper.

**Python suite.**

```bash
PYTHONPATH=src pytest -v
```

`test_cache_init.py` covers buffer sizing including the `num_blocks * CHUNK_SIZE` rule. `test_cache_methods.py` covers ingestion, tier routing, promotion and demotion, and score accumulation. `test_patch_llama.py` covers the patch and GQA head mapping.

---

## 14. Known limitations

| Issue | Impact | Status |
| :--- | :--- | :--- |
| `V_Grouped.reshape(..., quant_head_dim, 16)` assumes `head_dim % 16 == 0` | Silent wrong reshape for other head dims. Dormant at 64 and 128. | Open, needs an assert at minimum |
| No dtype, contiguity or device assertions before `.data_ptr()` | Memory corruption instead of an exception | Open |
| Fused path returns `attn_weights=None` | Breaks `output_attentions=True`, attention hooks, any logging of per-head weights | Open by design, needs documenting at the API level |
| Prefill is 1.22x slower than vanilla | Routing and quantization on the ingestion path | Open, not yet profiled |
| Batch size fixed at 1 | No batched serving | Out of scope currently |
| PPL rises 6.10 to 7.73 at 2048 context | Quality cost of 2-bit background storage | Accepted tradeoff, tunable via `R_size` and `H_ratio` |

---

## 15. Extension points

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
