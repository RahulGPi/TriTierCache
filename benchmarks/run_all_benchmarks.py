#!/usr/bin/env python3
"""
benchmarks/run_all_benchmarks.py
Master runner script that executes the complete TriTierCache benchmark suite:
- Correctness (Output Token Match & Numerical Drift)
- Perplexity / Quality (PPL, Ablation, NIAH Retrieval)
- Latency & Throughput (ms/token, tokens/sec, TTFT, AVX2 Microbenchmarks)
- Memory & Compression (RSS Growth Curve, Compression Ratio Curve)
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
from benchmarks.utils import save_results_to_csv


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="TriTierCache Master Benchmark Suite")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or checkpoint path")
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
        print_banner("Running Suite 1: Correctness & Verification")
        prompt_lens = [16, 64, 128] if args.quick else [16, 64, 128, 280]
        corr_out = benchmark_correctness.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
            prompt_lens=prompt_lens,
        )
        avg_match = sum(r["match_percentage"] for r in corr_out["token_match"]) / len(corr_out["token_match"])
        avg_cos_sim = sum(r["cosine_similarity"] for r in corr_out["drift"]) / len(corr_out["drift"])
        summary_rows.append({
            "category": "Correctness",
            "metric": "Avg Greedy Token Match (%)",
            "value": f"{avg_match:.2f}%",
        })
        summary_rows.append({
            "category": "Correctness",
            "metric": "Avg Logit Cosine Similarity",
            "value": f"{avg_cos_sim:.6f}",
        })

    # 2. Perplexity & Quality
    if args.suite in ["all", "perplexity"]:
        print_banner("Running Suite 2: Perplexity, Ablation & NIAH Retrieval")
        ppl_out = benchmark_perplexity.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
        )
        p_row = ppl_out["ppl"][0]
        summary_rows.append({
            "category": "Quality",
            "metric": "Vanilla Baseline PPL",
            "value": f"{p_row['vanilla_ppl']:.4f}",
        })
        summary_rows.append({
            "category": "Quality",
            "metric": "TriTierCache PPL",
            "value": f"{p_row['tritier_ppl']:.4f}",
        })
        summary_rows.append({
            "category": "Quality",
            "metric": "PPL Difference (%)",
            "value": f"{p_row['pct_diff']:+.2f}%",
        })

    # 3. Latency & Throughput
    if args.suite in ["all", "latency"]:
        print_banner("Running Suite 3: Latency, Throughput & Microbenchmarks")
        lat_out = benchmark_latency.run_benchmark(
            model_name=args.model,
            output_dir=args.output_dir,
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
        r3200 = next((r for r in mem_out["compression"] if r["context_length"] == 3200), mem_out["compression"][-1])
        r32k = next((r for r in mem_out["compression"] if r["context_length"] == 32768), mem_out["compression"][-1])
        summary_rows.append({
            "category": "Compression",
            "metric": "Compression Ratio @ 3,200 tokens (vs FP32)",
            "value": f"{r3200['compression_ratio_vs_fp32']:.2f}x",
        })
        summary_rows.append({
            "category": "Compression",
            "metric": "Compression Ratio @ 32,768 tokens (vs FP32)",
            "value": f"{r32k['compression_ratio_vs_fp32']:.2f}x",
        })

    # Save summary CSV
    save_results_to_csv(os.path.join(args.output_dir, "summary_results.csv"), summary_rows)
    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print(" BENCHMARK SUITE COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"{'Category':<16} | {'Metric':<45} | {'Result':<15}")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['category']:<16} | {row['metric']:<45} | {row['value']:<15}")
    print("=" * 80)
    print(f"Total Execution Time: {t_total:.2f} seconds")
    print(f"All CSV results saved to: '{os.path.abspath(args.output_dir)}/'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
