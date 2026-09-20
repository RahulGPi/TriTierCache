#!/usr/bin/env python3
"""
benchmarks/run_all_benchmarks.py
Master runner script that executes the complete TriTierCache benchmark suite (Phase 0 Compliant):
- Correctness (Top-1 Token Agreement & Numerical Drift vs Vanilla FP16)
- Perplexity / Quality (PPL >= 4x R_size, Ablation, NIAH Retrieval with Eviction Assertions)
- Latency & Bandwidth (ms/tok, TTFT, Implied Bandwidth, AVX2 Microbenchmarks)
- Memory & Live Buffer Accounting (sum element_size * nelement, Comp Ratios vs FP16/FP32)
"""
import os
import sys
import time
import argparse
from typing import Dict, Any, List

from benchmarks import (
    benchmark_correctness,
    benchmark_perplexity,
    benchmark_latency,
    benchmark_memory,
)
from benchmarks.utils import save_results_to_csv, DEFAULT_MODEL_ID


def print_banner(title: str):
    print("\n" + "=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="TriTierCache Master Benchmark Suite (Phase 0 Compliant)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or checkpoint path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Directory to save CSV results")
    parser.add_argument("--suite", type=str, choices=["all", "correctness", "perplexity", "latency", "memory"], default="all",
                        help="Specific benchmark suite to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="Run in fast mode with smaller sequence counts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.perf_counter()

    print_banner(f"Starting TriTierCache Benchmark Suite: Model={args.model}")
    summary_rows: List[Dict[str, Any]] = []

    # 1. Correctness
    if args.suite in ["all", "correctness"]:
        print_banner("Running Suite 1: Correctness & Verification (vs FP16 Baseline)")
        corr_out = benchmark_correctness.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
            quick=args.quick,
        )
        avg_match = sum(r["agreement_percentage"] for r in corr_out["token_match"]) / len(corr_out["token_match"])
        avg_cos_sim = sum(r["cosine_similarity"] for r in corr_out["drift"]) / len(corr_out["drift"])
        summary_rows.append({
            "category": "Correctness",
            "metric": "Avg Top-1 Token Agreement (%)",
            "value": f"{avg_match:.2f}%",
        })
        summary_rows.append({
            "category": "Correctness",
            "metric": "Avg Logit Cosine Similarity",
            "value": f"{avg_cos_sim:.6f}",
        })

    # 2. Perplexity & Quality
    if args.suite in ["all", "perplexity"]:
        print_banner("Running Suite 2: Perplexity, Ablation & NIAH Retrieval (Context >= 4x R_size)")
        ppl_out = benchmark_perplexity.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
            quick=args.quick,
        )
        p_row = ppl_out["ppl"][0]
        summary_rows.append({
            "category": "Quality",
            "metric": f"Vanilla FP16 Baseline PPL @ {p_row['eval_tokens']} toks",
            "value": f"{p_row['vanilla_fp16_ppl']:.4f}",
        })
        summary_rows.append({
            "category": "Quality",
            "metric": f"TriTierCache PPL @ {p_row['eval_tokens']} toks",
            "value": f"{p_row['tritier_ppl']:.4f}",
        })
        summary_rows.append({
            "category": "Quality",
            "metric": "PPL Difference vs FP16 (%)",
            "value": f"{p_row['pct_diff_vs_fp16']:+.2f}%",
        })

    # 3. Latency & Throughput
    if args.suite in ["all", "latency"]:
        print_banner("Running Suite 3: Latency, TTFT & Implied Bandwidth")
        lat_out = benchmark_latency.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
            quick=args.quick,
        )
        mean_tt_lat = sum(r["tritier_latency_ms"] for r in lat_out["latency"]) / len(lat_out["latency"])
        mean_vn_lat = sum(r["vanilla_latency_ms"] for r in lat_out["latency"]) / len(lat_out["latency"])
        summary_rows.append({
            "category": "Latency",
            "metric": "Mean Vanilla Decode Latency",
            "value": f"{mean_vn_lat:.2f} ms/token",
        })
        summary_rows.append({
            "category": "Latency",
            "metric": "Mean TriTier Decode Latency",
            "value": f"{mean_tt_lat:.2f} ms/token",
        })
        if lat_out["micro"]:
            summary_rows.append({
                "category": "Latency",
                "metric": "AVX2 Fused Kernel Decode Latency",
                "value": f"{lat_out['micro'][0]['mean_latency_us']:.2f} µs/step",
            })

    # 4. Memory & Compression
    if args.suite in ["all", "memory"]:
        print_banner("Running Suite 4: Memory Footprint & Compression Curves")
        mem_out = benchmark_memory.run_benchmark(
            output_dir=args.output_dir,
        )
        r256 = next((r for r in mem_out["compression"] if r["context_length"] == 256), mem_out["compression"][0])
        r32k = next((r for r in mem_out["compression"] if r["context_length"] == 32768), mem_out["compression"][-1])
        summary_rows.append({
            "category": "Compression",
            "metric": "Compression Ratio @ 256 tokens (vs FP16)",
            "value": f"{r256['compression_ratio_vs_fp16']:.2f}x",
        })
        summary_rows.append({
            "category": "Compression",
            "metric": "Compression Ratio @ 32,768 tokens (vs FP16)",
            "value": f"{r32k['compression_ratio_vs_fp16']:.2f}x",
        })

    # Save summary CSV
    save_results_to_csv(os.path.join(args.output_dir, "summary_results.csv"), summary_rows)
    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 90)
    print(" BENCHMARK SUITE COMPLETE - SUMMARY")
    print("=" * 90)
    print(f"{'Category':<16} | {'Metric':<50} | {'Result':<15}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['category']:<16} | {row['metric']:<50} | {row['value']:<15}")
    print("=" * 90)
    print(f"Total Execution Time: {t_total:.2f} seconds")
    print(f"All CSV results saved to: '{os.path.abspath(args.output_dir)}/'")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
