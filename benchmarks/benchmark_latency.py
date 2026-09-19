#!/usr/bin/env python3
"""
benchmarks/benchmark_latency.py
Evaluates:
1. Decode Latency (ms/token) & Throughput (tokens/sec) at increasing context lengths.
2. Time-to-First-Token (TTFT / prefill latency).
3. Per-kernel AVX2 microbenchmarks (quantize, dequantize, fused attention).
"""
import os
import time
import argparse
import torch
from typing import List, Dict, Any
from benchmarks.utils import load_model_and_tokenizer, generate_step_by_step, save_results_to_csv
from tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches

try:
    import tri_tier._C as _C
    HAS_CPP_EXT = True
except ImportError:
    _C = None
    HAS_CPP_EXT = False


def benchmark_decode_latency_throughput(model, tok, context_lengths: List[int] = None, gen_tokens: int = 30) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(" [1/3] BENCHMARKING DECODE LATENCY & THROUGHPUT (batch=1)")
    print("=" * 80)

    if context_lengths is None:
        context_lengths = [128, 256, 512, 1024]

    filler = "Scaling large language models requires extreme computational efficiency and memory bandwidth optimizations. "
    results = []

    print(f"{'Context':<9} | {'Vanilla (ms/tok)':<18} | {'TriTier (ms/tok)':<18} | {'Vanilla (tok/s)':<16} | {'TriTier (tok/s)':<16} | {'Speedup'}")
    print("-" * 80)

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
        apply_patch()
        reset_caches(model)
        _, tritier_lats = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        tritier_avg_ms = (sum(tritier_lats) / len(tritier_lats)) * 1000.0
        tritier_tps = 1000.0 / tritier_avg_ms if tritier_avg_ms > 0 else 0.0

        speedup = vanilla_avg_ms / max(0.001, tritier_avg_ms)

        print(f"{ctx_len:<9d} | {vanilla_avg_ms:<18.2f} | {tritier_avg_ms:<18.2f} | {vanilla_tps:<16.2f} | {tritier_tps:<16.2f} | {speedup:5.2f}x")

        results.append({
            "context_length": ctx_len,
            "generated_tokens": gen_tokens,
            "vanilla_latency_ms": vanilla_avg_ms,
            "tritier_latency_ms": tritier_avg_ms,
            "vanilla_throughput_tokens_per_sec": vanilla_tps,
            "tritier_throughput_tokens_per_sec": tritier_tps,
            "speedup": speedup,
        })

    return results


def benchmark_ttft(model, tok, prompt_lengths: List[int] = None) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(" [2/3] BENCHMARKING TIME-TO-FIRST-TOKEN (TTFT / Prefill Latency)")
    print("=" * 80)

    if prompt_lengths is None:
        prompt_lengths = [64, 128, 256, 512]

    filler = "High throughput auto-regressive generation requires efficient key-value cache memory layout. "
    results = []

    print(f"{'Prompt Len':<12} | {'Vanilla TTFT (ms)':<20} | {'TriTier TTFT (ms)':<20} | {'Diff (ms)'}")
    print("-" * 80)

    for p_len in prompt_lengths:
        tokens = tok(filler, return_tensors="pt").input_ids
        while tokens.shape[1] < p_len:
            tokens = torch.cat([tokens, tokens], dim=1)
        input_ids = tokens[:, :p_len]

        # 1. Vanilla
        remove_patch()
        reset_caches(model)
        t0 = time.perf_counter()
        past_kv = None
        with torch.no_grad():
            for pos in range(p_len):
                out = model(input_ids[:, pos:pos+1], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            _ = out.logits[:, -1, :].argmax(dim=-1)
        t1 = time.perf_counter()
        vanilla_ttft_ms = (t1 - t0) * 1000.0

        # 2. TriTierCache
        apply_patch()
        reset_caches(model)
        t0 = time.perf_counter()
        with torch.no_grad():
            for pos in range(p_len):
                out = model(input_ids[:, pos:pos+1])
            _ = out.logits[:, -1, :].argmax(dim=-1)
        t1 = time.perf_counter()
        tritier_ttft_ms = (t1 - t0) * 1000.0

        diff_ms = tritier_ttft_ms - vanilla_ttft_ms
        print(f"{p_len:<12d} | {vanilla_ttft_ms:<20.2f} | {tritier_ttft_ms:<20.2f} | {diff_ms:+8.2f} ms")

        results.append({
            "prompt_length": p_len,
            "vanilla_ttft_ms": vanilla_ttft_ms,
            "tritier_ttft_ms": tritier_ttft_ms,
            "diff_ms": diff_ms,
        })

    return results


def benchmark_microbenchmarks(num_iters: int = 500) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(" [3/3] BENCHMARKING PER-KERNEL AVX2 MICROBENCHMARKS")
    print("=" * 80)

    if not HAS_CPP_EXT or _C is None:
        print("C++ Extension tri_tier._C is not available. Skipping microbenchmarks.")
        return []

    num_q_heads = 32
    num_kv_heads = 8
    head_dim = 128
    dense_count = 260
    num_blocks = 20

    # Allocate tensors
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
    total_tokens = dense_count + num_blocks * 16
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

    # Benchmark fused attention decode
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
    avg_fused_us = ((t1 - t0) / num_iters) * 1_000_000.0

    print(f"Kernel: Fused Attention Decode AVX2 (tokens={total_tokens}, q_heads={num_q_heads}, kv_heads={num_kv_heads}, dim={head_dim})")
    print(f"Mean Latency: {avg_fused_us:.2f} µs per decode step | Calls/sec: {1_000_000.0 / avg_fused_us:.1f} ops/sec")

    records = [{
        "kernel_name": "fused_attention_decode_avx2",
        "total_active_tokens": total_tokens,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "mean_latency_us": avg_fused_us,
        "ops_per_sec": 1_000_000.0 / avg_fused_us,
    }]

    return records


def run_benchmark(model_name: str = "meta-llama/Llama-3.2-1B", 
                  output_dir: str = "benchmarks/results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_name)

    # 1. Decode Latency & Throughput
    lat_res = benchmark_decode_latency_throughput(model, tok, context_lengths=[128, 256, 512], gen_tokens=25)
    save_results_to_csv(os.path.join(output_dir, "latency_results.csv"), lat_res)

    # 2. TTFT
    ttft_res = benchmark_ttft(model, tok, prompt_lengths=[64, 128, 256])
    save_results_to_csv(os.path.join(output_dir, "ttft_results.csv"), ttft_res)

    # 3. Microbenchmarks
    micro_res = benchmark_microbenchmarks(num_iters=1000)
    save_results_to_csv(os.path.join(output_dir, "microbenchmarks_results.csv"), micro_res)

    return {"latency": lat_res, "ttft": ttft_res, "micro": micro_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Latency, Throughput & Microbenchmarks")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    args = parser.parse_args()

    run_benchmark(model_name=args.model, output_dir=args.output_dir)
