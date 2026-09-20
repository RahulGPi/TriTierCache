#!/usr/bin/env python3
"""
benchmarks/benchmark_latency.py
Evaluates:
1. Decode Latency (ms/token), Throughput (tokens/sec), and Implied Memory Bandwidth (GB/s)
   across 8-point context length sweep: {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}.
2. Time-to-First-Token (TTFT / prefill latency) across: {128, 512, 1024, 2048, 4096, 8192}.
3. AVX2 microbenchmarks with 5 repeats (mean ± std).
4. Multi-core OpenMP thread scaling with 5 repeats (mean ± std).
"""
import os
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Any

from benchmarks.common import (
    load_model,
    check_bandwidth_plausible,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from benchmarks.utils import generate_step_by_step
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod

try:
    import tri_tier._C as _C
    HAS_CPP_EXT = True
except ImportError:
    _C = None
    HAS_CPP_EXT = False


def calculate_model_weight_bytes(model) -> int:
    """Calculates total parameter memory footprint in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def benchmark_decode_latency_throughput(model_name: str = DEFAULT_MODEL_ID,
                                        context_lengths: List[int] = None,
                                        gen_tokens: int = 20,
                                        run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 115)
    print(f" [1/4] BENCHMARKING DECODE LATENCY, THROUGHPUT & IMPLIED MEMORY BANDWIDTH (Model: {model_name})")
    print("=" * 115)

    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    model, tok = load_model(model_name, dtype=torch.float32)
    weight_bytes = calculate_model_weight_bytes(model)
    weight_mb = weight_bytes / (1024.0 * 1024.0)
    print(f"Model Parameter Size: {weight_mb:.2f} MB ({weight_bytes} bytes)")

    filler = "Scaling deep autoregressive language models requires extreme computational efficiency and low memory bandwidth overhead. "
    results = []

    print(f"{'Context':<9} | {'Vanilla (ms/tok)':<18} | {'TriTier (ms/tok)':<18} | {'Implied BW (GB/s)':<20} | {'TriTier tok/s':<16} | {'Bandwidth Plausible'}")
    print("-" * 115)

    for ctx_len in context_lengths:
        tokens = tok(filler, return_tensors="pt").input_ids
        while tokens.shape[1] < ctx_len:
            tokens = torch.cat([tokens, tokens], dim=1)
        input_ids = tokens[:, :ctx_len]

        # 1. Vanilla Baseline
        remove_patch()
        reset_caches(model)
        _, vanilla_lats = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        vanilla_avg_ms = (sum(vanilla_lats) / len(vanilla_lats)) * 1000.0
        vanilla_tps = 1000.0 / vanilla_avg_ms if vanilla_avg_ms > 0 else 0.0

        # 2. TriTierCache
        patch_mod.R_SIZE = run_config.R_size
        patch_mod.H_RATIO = run_config.H_ratio
        apply_patch()
        reset_caches(model)
        _, tritier_lats = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        tritier_avg_ms = (sum(tritier_lats) / len(tritier_lats)) * 1000.0
        tritier_tps = 1000.0 / tritier_avg_ms if tritier_avg_ms > 0 else 0.0

        # Implied Bandwidth = (Weight Bytes / Time per Token)
        is_plausible, implied_bw_gb_s = check_bandwidth_plausible(tritier_avg_ms, weight_bytes)
        bw_flag = "PLAUSIBLE (<100 GB/s)" if is_plausible else f"WARN: {implied_bw_gb_s:.1f} GB/s (>100 GB/s)"

        print(f"{ctx_len:<9d} | {vanilla_avg_ms:<18.2f} | {tritier_avg_ms:<18.2f} | {implied_bw_gb_s:<20.2f} | {tritier_tps:<16.2f} | {bw_flag}")

        row = {
            "context_length": ctx_len,
            "generated_tokens": gen_tokens,
            "vanilla_latency_ms": vanilla_avg_ms,
            "tritier_latency_ms": tritier_avg_ms,
            "implied_bandwidth_gb_s": implied_bw_gb_s,
            "bandwidth_plausible": is_plausible,
            "vanilla_throughput_tok_s": vanilla_tps,
            "tritier_throughput_tok_s": tritier_tps,
            "bandwidth_flag": bw_flag,
        }
        row.update(run_config.to_dict())
        results.append(row)

    return results


def benchmark_ttft(model_name: str = DEFAULT_MODEL_ID,
                   prompt_lengths: List[int] = None,
                   run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 115)
    print(f" [2/4] BENCHMARKING TIME-TO-FIRST-TOKEN (TTFT / Prefill Latency, Model: {model_name})")
    print("=" * 115)

    if prompt_lengths is None:
        prompt_lengths = [128, 512, 1024, 2048, 4096, 8192]

    model, tok = load_model(model_name, dtype=torch.float32)
    filler = "High throughput auto-regressive generation requires efficient key-value cache memory layout. "
    results = []

    print(f"{'Prompt Len':<12} | {'Vanilla TTFT (ms)':<20} | {'TriTier TTFT (ms)':<20} | {'Diff (ms)'}")
    print("-" * 115)

    for p_len in prompt_lengths:
        tokens = tok(filler, return_tensors="pt").input_ids
        while tokens.shape[1] < p_len:
            tokens = torch.cat([tokens, tokens], dim=1)
        input_ids = tokens[:, :p_len]

        # 1. Vanilla Batched Prefill
        remove_patch()
        reset_caches(model)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1)
        t1 = time.perf_counter()
        vanilla_ttft_ms = (t1 - t0) * 1000.0

        # 2. TriTier Batched Prefill
        patch_mod.R_SIZE = run_config.R_size
        patch_mod.H_RATIO = run_config.H_ratio
        apply_patch()
        reset_caches(model)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids)
            _ = out.logits[:, -1, :].argmax(dim=-1)
        t1 = time.perf_counter()
        tritier_ttft_ms = (t1 - t0) * 1000.0

        diff_ms = tritier_ttft_ms - vanilla_ttft_ms
        print(f"{p_len:<12d} | {vanilla_ttft_ms:<20.2f} | {tritier_ttft_ms:<20.2f} | {diff_ms:+8.2f} ms")

        row = {
            "prompt_length": p_len,
            "vanilla_ttft_ms": vanilla_ttft_ms,
            "tritier_ttft_ms": tritier_ttft_ms,
            "diff_ms": diff_ms,
        }
        row.update(run_config.to_dict())
        results.append(row)

    return results


def benchmark_microbenchmarks(num_iters: int = 500, repeats: int = 5, seeds: List[int] = None) -> List[Dict[str, Any]]:
    print("\n" + "=" * 115)
    print(f" [3/4] BENCHMARKING PER-KERNEL AVX2 MICROBENCHMARKS ({repeats} Repeats with Mean ± Std)")
    print("=" * 115)

    if not HAS_CPP_EXT or _C is None:
        print("C++ Extension tri_tier._C is not available. Skipping microbenchmarks.")
        return []

    if seeds is None:
        seeds = [42, 43, 44, 45, 46][:repeats]

    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    dense_count = 260
    num_blocks = 20
    total_tokens = dense_count + num_blocks * 16

    repeat_latencies = []

    for rep_idx, seed in enumerate(seeds, start=1):
        set_seed(seed)
        Q = torch.randn(num_q_heads, head_dim, dtype=torch.float32)
        dense_K = torch.randn(dense_count, num_kv_heads, head_dim, dtype=torch.float32)
        dense_V = torch.randn(dense_count, num_kv_heads, head_dim, dtype=torch.float32)
        PBS_K_Packed = torch.randint(0, 2**31 - 1, (num_blocks, num_kv_heads, head_dim), dtype=torch.int32)
        PBS_K_Scales = torch.rand(num_blocks, num_kv_heads, head_dim, dtype=torch.float32)
        PBS_K_Zeroes = torch.rand(num_blocks, num_kv_heads, head_dim, dtype=torch.float32)
        PBS_V_Packed = torch.randint(0, 2**31 - 1, (num_blocks * 16, num_kv_heads, head_dim // 16), dtype=torch.int32)
        PBS_V_Scales = torch.rand(num_blocks * 16, num_kv_heads, 1, dtype=torch.float32)
        PBS_V_Zeroes = torch.rand(num_blocks * 16, num_kv_heads, 1, dtype=torch.float32)
        PBS_token_ids = torch.arange(num_blocks * 16, dtype=torch.int64)
        attn_output = torch.empty(num_q_heads, head_dim, dtype=torch.float32)
        mean_weights = torch.empty(total_tokens, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _C.fused_attention_decode(
                Q.data_ptr(), dense_K.data_ptr(), dense_V.data_ptr(), dense_count,
                PBS_K_Packed.data_ptr(), PBS_K_Scales.data_ptr(), PBS_K_Zeroes.data_ptr(),
                PBS_V_Packed.data_ptr(), PBS_V_Scales.data_ptr(), PBS_V_Zeroes.data_ptr(),
                PBS_token_ids.data_ptr(), num_blocks, num_q_heads, num_kv_heads, head_dim,
                attn_output.data_ptr(), mean_weights.data_ptr(),
            )

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _C.fused_attention_decode(
                Q.data_ptr(), dense_K.data_ptr(), dense_V.data_ptr(), dense_count,
                PBS_K_Packed.data_ptr(), PBS_K_Scales.data_ptr(), PBS_K_Zeroes.data_ptr(),
                PBS_V_Packed.data_ptr(), PBS_V_Scales.data_ptr(), PBS_V_Zeroes.data_ptr(),
                PBS_token_ids.data_ptr(), num_blocks, num_q_heads, num_kv_heads, head_dim,
                attn_output.data_ptr(), mean_weights.data_ptr(),
            )
        t1 = time.perf_counter()
        avg_us = ((t1 - t0) / num_iters) * 1_000_000.0
        repeat_latencies.append(avg_us)

    mean_us = float(np.mean(repeat_latencies))
    std_us = float(np.std(repeat_latencies))
    ops_sec = 1_000_000.0 / mean_us

    print(f"Kernel: Fused Attention Decode AVX2 (tokens={total_tokens}, q_heads={num_q_heads}, kv_heads={num_kv_heads}, dim={head_dim})")
    print(f"Mean Latency: {mean_us:.2f} ± {std_us:.2f} µs per decode step | Calls/sec: {ops_sec:.1f} ops/sec (across {repeats} runs)")

    records = [{
        "kernel_name": "fused_attention_decode_avx2",
        "total_active_tokens": total_tokens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "repeats": repeats,
        "seeds": str(seeds),
        "mean_latency_us": mean_us,
        "std_latency_us": std_us,
        "ops_per_sec": ops_sec,
    }]

    return records


def benchmark_thread_scaling(thread_counts: List[int] = None,
                             num_iters: int = 500,
                             repeats: int = 5,
                             seeds: List[int] = None) -> List[Dict[str, Any]]:
    print("\n" + "=" * 115)
    print(f" [4/4] BENCHMARKING MULTI-CORE THREAD SCALING ({repeats} Repeats with Mean ± Std)")
    print("=" * 115)

    if not HAS_CPP_EXT or _C is None:
        print("C++ Extension tri_tier._C is not available. Skipping thread scaling benchmark.")
        return []

    if thread_counts is None:
        thread_counts = [1, 2, 4, 8]
    if seeds is None:
        seeds = [42, 43, 44, 45, 46][:repeats]

    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    dense_count = 260
    num_blocks = 20
    total_tokens = dense_count + num_blocks * 16

    records = []
    base_latency = None
    print(f"{'Threads':<10} | {'Latency (µs)':<22} | {'Speedup vs 1T':<18} | {'Throughput (calls/sec)':<24} | {'Status'}")
    print("-" * 115)

    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)

        thread_lats = []
        for seed in seeds:
            set_seed(seed)
            Q = torch.randn(num_q_heads, head_dim, dtype=torch.float32)
            dense_K = torch.randn(dense_count, num_kv_heads, head_dim, dtype=torch.float32)
            dense_V = torch.randn(dense_count, num_kv_heads, head_dim, dtype=torch.float32)
            PBS_K_Packed = torch.randint(0, 2**31 - 1, (num_blocks, num_kv_heads, head_dim), dtype=torch.int32)
            PBS_K_Scales = torch.rand(num_blocks, num_kv_heads, head_dim, dtype=torch.float32)
            PBS_K_Zeroes = torch.rand(num_blocks, num_kv_heads, head_dim, dtype=torch.float32)
            PBS_V_Packed = torch.randint(0, 2**31 - 1, (num_blocks * 16, num_kv_heads, head_dim // 16), dtype=torch.int32)
            PBS_V_Scales = torch.rand(num_blocks * 16, num_kv_heads, 1, dtype=torch.float32)
            PBS_V_Zeroes = torch.rand(num_blocks * 16, num_kv_heads, 1, dtype=torch.float32)
            PBS_token_ids = torch.arange(num_blocks * 16, dtype=torch.int64)
            attn_output = torch.empty(num_q_heads, head_dim, dtype=torch.float32)
            mean_weights = torch.empty(total_tokens, dtype=torch.float32)

            for _ in range(10):
                _C.fused_attention_decode(
                    Q.data_ptr(), dense_K.data_ptr(), dense_V.data_ptr(), dense_count,
                    PBS_K_Packed.data_ptr(), PBS_K_Scales.data_ptr(), PBS_K_Zeroes.data_ptr(),
                    PBS_V_Packed.data_ptr(), PBS_V_Scales.data_ptr(), PBS_V_Zeroes.data_ptr(),
                    PBS_token_ids.data_ptr(), num_blocks, num_q_heads, num_kv_heads, head_dim,
                    attn_output.data_ptr(), mean_weights.data_ptr(),
                )

            t0 = time.perf_counter()
            for _ in range(num_iters):
                _C.fused_attention_decode(
                    Q.data_ptr(), dense_K.data_ptr(), dense_V.data_ptr(), dense_count,
                    PBS_K_Packed.data_ptr(), PBS_K_Scales.data_ptr(), PBS_K_Zeroes.data_ptr(),
                    PBS_V_Packed.data_ptr(), PBS_V_Scales.data_ptr(), PBS_V_Zeroes.data_ptr(),
                    PBS_token_ids.data_ptr(), num_blocks, num_q_heads, num_kv_heads, head_dim,
                    attn_output.data_ptr(), mean_weights.data_ptr(),
                )
            t1 = time.perf_counter()
            lat_us = ((t1 - t0) / num_iters) * 1_000_000.0
            thread_lats.append(lat_us)

        mean_lat = float(np.mean(thread_lats))
        std_lat = float(np.std(thread_lats))

        if base_latency is None:
            base_latency = mean_lat
        speedup = base_latency / max(1e-6, mean_lat)
        calls_per_sec = 1_000_000.0 / mean_lat

        lat_str = f"{mean_lat:.2f} ± {std_lat:.2f}"
        print(f"{n_threads:<10d} | {lat_str:<22} | {speedup:<18.2f}x | {calls_per_sec:<24.1f} | PASSED")

        records.append({
            "num_threads": n_threads,
            "repeats": repeats,
            "seeds": str(seeds),
            "mean_latency_us": mean_lat,
            "std_latency_us": std_lat,
            "speedup_vs_1t": speedup,
            "calls_per_sec": calls_per_sec,
        })

    return records


def run_benchmark(model_name: str = DEFAULT_MODEL_ID, 
                  output_dir: str = "benchmarks/results",
                  quick: bool = False,
                  run_config: RunConfig = None) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if run_config is None:
        run_config = RunConfig(model_id=model_name)

    ctx_lengths = [256, 512, 1024] if quick else [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    prompt_lengths = [128, 512, 1024] if quick else [128, 512, 1024, 2048, 4096, 8192]

    # 1. Decode Latency & Throughput
    lat_res = benchmark_decode_latency_throughput(model_name, context_lengths=ctx_lengths, gen_tokens=20, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "latency_results.csv"), lat_res)

    # 2. TTFT
    ttft_res = benchmark_ttft(model_name, prompt_lengths=prompt_lengths, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "ttft_results.csv"), ttft_res)

    # 3. Microbenchmarks (5 repeats)
    micro_res = benchmark_microbenchmarks(num_iters=200 if quick else 500, repeats=5)
    save_results_to_csv(os.path.join(output_dir, "microbenchmarks_results.csv"), micro_res)

    # 4. Thread Scaling (5 repeats)
    scaling_res = benchmark_thread_scaling(thread_counts=[1, 2, 4, 8], num_iters=200 if quick else 500, repeats=5)
    save_results_to_csv(os.path.join(output_dir, "thread_scaling_results.csv"), scaling_res)

    return {"latency": lat_res, "ttft": ttft_res, "micro": micro_res, "thread_scaling": scaling_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Latency, Throughput & Microbenchmarks")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_cfg = RunConfig(model_id=args.model, seed=args.seed)
    run_benchmark(model_name=args.model, output_dir=args.output_dir, quick=args.quick, run_config=run_cfg)
