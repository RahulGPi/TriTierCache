#!/usr/bin/env python3
"""
benchmarks/benchmark_memory.py
Evaluates:
1. Peak RSS Process Memory (MB) growth curve: Vanilla vs TriTierCache.
2. Actual Achieved Compression Ratios across context lengths (512 up to 32,768 tokens).
"""
import os
import gc
import argparse
import torch
from typing import List, Dict, Any
from benchmarks.utils import get_process_rss_mb, calculate_tri_tier_cache_bytes, save_results_to_csv
from tri_tier.cache import TriTierCache


def benchmark_compression_curve(seq_lengths: List[int] = None,
                                num_kv_heads: int = 8,
                                head_dim: int = 128,
                                r_size: int = 256,
                                h_ratio: float = 0.05) -> List[Dict[str, Any]]:
    print("\n" + "=" * 85)
    print(" [1/2] BENCHMARKING ACHIEVED COMPRESSION RATIO CURVE")
    print("=" * 85)

    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 3200, 4096, 8192, 16384, 32768]

    records = []
    print(f"{'Context (tokens)':<18} | {'Vanilla FP32 (MB)':<18} | {'Vanilla FP16 (MB)':<18} | {'TriTier (MB)':<14} | {'Comp Ratio (vs FP32)':<20} | {'Comp Ratio (vs FP16)'}")
    print("-" * 110)

    for seq_len in seq_lengths:
        comp = calculate_tri_tier_cache_bytes(
            total_tokens=seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sink_size=4,
            r_size=r_size,
            h_ratio=h_ratio,
        )

        fp32_mb = comp["vanilla_fp32_bytes"] / (1024.0 * 1024.0)
        fp16_mb = comp["vanilla_fp16_bytes"] / (1024.0 * 1024.0)
        tritier_mb = comp["tri_tier_bytes"] / (1024.0 * 1024.0)
        ratio_fp32 = comp["compression_ratio_vs_fp32"]
        ratio_fp16 = comp["compression_ratio_vs_fp16"]

        print(f"{seq_len:<18d} | {fp32_mb:<18.2f} | {fp16_mb:<18.2f} | {tritier_mb:<14.2f} | {ratio_fp32:<20.2f}x | {ratio_fp16:.2f}x")

        records.append({
            "context_length": seq_len,
            "vanilla_fp32_mb": fp32_mb,
            "vanilla_fp16_mb": fp16_mb,
            "tritier_mb": tritier_mb,
            "compression_ratio_vs_fp32": ratio_fp32,
            "compression_ratio_vs_fp16": ratio_fp16,
        })

    return records


def benchmark_rss_memory_footprint(seq_lengths: List[int] = None,
                                   num_kv_heads: int = 8,
                                   head_dim: int = 128) -> List[Dict[str, Any]]:
    print("\n" + "=" * 85)
    print(" [2/2] BENCHMARKING PHYSICAL RSS MEMORY ALLOCATION (Flat Growth Curve)")
    print("=" * 85)

    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    records = []
    print(f"{'Context (tokens)':<18} | {'Vanilla Cache RSS (MB)':<25} | {'TriTierCache RSS (MB)':<25}")
    print("-" * 75)

    for seq_len in seq_lengths:
        # 1. Vanilla simulation: full FP32 tensors
        gc.collect()
        rss_before = get_process_rss_mb()
        vanilla_k = torch.empty((seq_len, num_kv_heads, head_dim), dtype=torch.float32)
        vanilla_v = torch.empty((seq_len, num_kv_heads, head_dim), dtype=torch.float32)
        vanilla_rss = get_process_rss_mb() - rss_before
        del vanilla_k, vanilla_v
        gc.collect()

        # 2. TriTierCache instance
        rss_before_tt = get_process_rss_mb()
        cache = TriTierCache(
            max_seq_len=seq_len,
            head_dim=head_dim,
            num_heads=num_kv_heads,
            R_size=256,
            H_ratio=0.05,
        )
        tritier_rss = get_process_rss_mb() - rss_before_tt
        del cache
        gc.collect()

        print(f"{seq_len:<18d} | {max(0.0, vanilla_rss):<25.2f} | {max(0.0, tritier_rss):<25.2f}")

        records.append({
            "context_length": seq_len,
            "vanilla_allocated_rss_mb": max(0.0, vanilla_rss),
            "tritier_allocated_rss_mb": max(0.0, tritier_rss),
        })

    return records


def run_benchmark(output_dir: str = "benchmarks/results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compression ratio scaling curve
    comp_res = benchmark_compression_curve()
    save_results_to_csv(os.path.join(output_dir, "compression_ratio_results.csv"), comp_res)

    # 2. Physical RSS footprint
    rss_res = benchmark_rss_memory_footprint()
    save_results_to_csv(os.path.join(output_dir, "memory_rss_results.csv"), rss_res)

    return {"compression": comp_res, "rss": rss_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Memory Footprint & Compression Benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    args = parser.parse_args()

    run_benchmark(output_dir=args.output_dir)
