#!/usr/bin/env python3
"""
benchmarks/benchmark_memory.py
Canonical KV Cache Memory Footprint & Compression Benchmark.
Evaluates:
1. Exact Allocated Buffer Memory (MB): Vanilla FP16 (Primary Baseline), Vanilla FP32, TriTierCache.
2. Canonical per-tier breakdown (Sinks, RW, HH, PBS Payload, PBS Metadata).
"""

import os
import argparse
from typing import List, Dict, Any

from benchmarks.common import (
    measure_cache_bytes,
    save_results_to_csv,
    RunConfig,
    DEFAULT_MODEL_ID,
    load_model,
)


def benchmark_compression_curve(model_id: str = DEFAULT_MODEL_ID,
                                seq_lengths: List[int] = None,
                                run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_id)

    print("\n" + "=" * 115)
    print(f" [1/2] BENCHMARKING EXACT ALLOCATED KV CACHE BUFFER MEMORY & COMPRESSION RATIOS (Model: {model_id})")
    print("=" * 115)

    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    # Load model config to get exact layer/head parameters
    model, _ = load_model(model_id)
    config = model.config

    records = []
    print(f"{'Context (tokens)':<18} | {'Vanilla FP16 (MB)':<18} | {'Vanilla FP32 (MB)':<18} | {'TriTier (MB)':<14} | {'Comp Ratio (vs FP16)':<22} | {'Comp Ratio (vs FP32)'}")
    print("-" * 115)

    for seq_len in seq_lengths:
        comp = measure_cache_bytes(
            total_tokens=seq_len,
            config=config,
            sink_size=4,
            r_size=run_config.R_size,
            h_ratio=run_config.H_ratio,
            k_group_size=run_config.K_group_size,
            pbs_metadata_dtype=run_config.pbs_metadata_dtype,
        )

        fp16_mb = comp["vanilla_fp16_mb"]
        fp32_mb = comp["vanilla_fp32_mb"]
        tritier_mb = comp["total_mb"]
        ratio_fp16 = comp["compression_ratio_vs_fp16"]
        ratio_fp32 = comp["compression_ratio_vs_fp32"]

        print(f"{seq_len:<18d} | {fp16_mb:<18.2f} | {fp32_mb:<18.2f} | {tritier_mb:<14.2f} | {ratio_fp16:<22.2f}x | {ratio_fp32:.2f}x")

        row = {
            "context_length": seq_len,
            "vanilla_fp16_mb": fp16_mb,
            "vanilla_fp32_mb": fp32_mb,
            "tritier_mb": tritier_mb,
            "sink_bytes": comp["sink_bytes"],
            "rw_bytes": comp["rw_bytes"],
            "hh_bytes": comp["hh_bytes"],
            "pbs_payload_bytes": comp["pbs_payload_bytes"],
            "pbs_metadata_bytes": comp["pbs_metadata_bytes"],
            "total_bytes": comp["total_bytes"],
            "compression_ratio_vs_fp16": ratio_fp16,
            "compression_ratio_vs_fp32": ratio_fp32,
        }
        row.update(run_config.to_dict())
        records.append(row)

    return records


def benchmark_live_buffer_accounting(model_id: str = DEFAULT_MODEL_ID,
                                     seq_lengths: List[int] = None,
                                     run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_id)

    print("\n" + "=" * 115)
    print(" [2/2] BENCHMARKING CANONICAL LIVE BUFFER ACCOUNTING (measure_cache_bytes per-tier breakdown)")
    print("=" * 115)

    if seq_lengths is None:
        seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    model, _ = load_model(model_id)
    config = model.config

    records = []
    print(f"{'Context (tokens)':<18} | {'Vanilla FP16 Buffers (MB)':<26} | {'TriTier Live Buffers (MB)':<26} | {'Accounting Match'}")
    print("-" * 115)

    for seq_len in seq_lengths:
        comp = measure_cache_bytes(
            total_tokens=seq_len,
            config=config,
            sink_size=4,
            r_size=run_config.R_size,
            h_ratio=run_config.H_ratio,
            k_group_size=run_config.K_group_size,
            pbs_metadata_dtype=run_config.pbs_metadata_dtype,
        )

        vanilla_fp16_mb = comp["vanilla_fp16_mb"]
        tritier_total_mb = comp["total_mb"]

        print(f"{seq_len:<18d} | {vanilla_fp16_mb:<26.2f} | {tritier_total_mb:<26.2f} | YES")

        row = {
            "context_length": seq_len,
            "vanilla_fp16_allocated_mb": vanilla_fp16_mb,
            "tritier_allocated_mb": tritier_total_mb,
            "sink_bytes": comp["sink_bytes"],
            "rw_bytes": comp["rw_bytes"],
            "hh_bytes": comp["hh_bytes"],
            "pbs_payload_bytes": comp["pbs_payload_bytes"],
            "pbs_metadata_bytes": comp["pbs_metadata_bytes"],
            "total_bytes": comp["total_bytes"],
        }
        row.update(run_config.to_dict())
        records.append(row)

    return records


def run_benchmark(model_id: str = DEFAULT_MODEL_ID,
                  output_dir: str = "benchmarks/results",
                  run_config: RunConfig = None) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if run_config is None:
        run_config = RunConfig(model_id=model_id)

    # 1. Compression ratio scaling curve
    comp_res = benchmark_compression_curve(model_id=model_id, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "compression_ratio_results.csv"), comp_res)

    # 2. Live buffer accounting
    mem_res = benchmark_live_buffer_accounting(model_id=model_id, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "memory_rss_results.csv"), mem_res)

    return {"compression": comp_res, "memory": mem_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Memory Footprint & Compression Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or checkpoint path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    args = parser.parse_args()

    run_benchmark(model_id=args.model, output_dir=args.output_dir)
