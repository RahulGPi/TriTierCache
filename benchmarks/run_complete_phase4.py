#!/usr/bin/env python3
"""
benchmarks/run_complete_phase4.py
Phase 4: Complete NIAH Dataset with fine-grained intervals [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5].
Broken out by:
  - rope_mode: 'a' vs 'b'
  - h_ratio: 0.05 vs 0.15
  - depth: 0.20, 0.70, 0.95

Emits raw CSV:
  benchmarks/results/niah_complete_graduated_dataset.csv
"""

import os
import argparse
import csv
import numpy as np
import torch

from typing import List, Dict, Any
from benchmarks.common import (
    load_model,
    save_results_to_csv,
    set_seed,
    DEFAULT_MODEL_ID,
)
from benchmarks.analyze_niah_failure import analyze_needle_storage_fidelity


def run_complete_phase4(model_name: str = DEFAULT_MODEL_ID,
                        output_csv: str = "benchmarks/results/niah_complete_graduated_dataset.csv",
                        seed: int = 42) -> List[Dict[str, Any]]:
    set_seed(seed)
    model, tok = load_model(model_name, dtype=torch.float32)
    base_trained_len = 2048

    # Load existing 24 rows if available from niah_graduated_curve.csv
    existing_rows = []
    curve_csv = "benchmarks/results/niah_graduated_curve.csv"
    if os.path.exists(curve_csv):
        with open(curve_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append({
                    "length_ratio": float(r["length_ratio"]),
                    "context_length": int(r["context_length"]),
                    "depth": float(r["depth"]),
                    "rope_mode": r["rope_mode"],
                    "h_ratio": float(r["h_ratio"]),
                    "needle_tier": r["needle_tier"],
                    "mean_k_cosine_sim": float(r["mean_k_cosine_sim"]),
                    "mean_v_cosine_sim": float(r["mean_v_cosine_sim"]),
                    "storage_intact": (r["storage_intact"].lower() == "true"),
                    "generated_text": r["generated_text"],
                    "success": (r["success"].lower() == "true"),
                })
        print(f"Loaded {len(existing_rows)} existing rows from {curve_csv}")

    # Missing fine-grained intervals
    fine_grained_ratios = [1.1, 1.2, 1.3, 1.4]
    rope_modes = ["a", "b"]
    depths = [0.20, 0.70, 0.95]
    h_ratios = [0.05, 0.15]

    new_rows = []

    print("\n" + "=" * 135)
    print(" [PHASE 4] EVALUATING FINE-GRAINED INTERVALS (1.1x to 1.4x)")
    print("=" * 135)
    print(f"{'Ratio':<8} | {'Context':<8} | {'Depth':<8} | {'RoPE':<6} | {'H_ratio':<8} | {'Tier':<8} | {'K Cos':<10} | {'V Cos':<10} | {'Output':<18} | {'Success'}")
    print("-" * 135)

    for ratio in fine_grained_ratios:
        total_len = int(ratio * base_trained_len)
        for h_ratio in h_ratios:
            for mode in rope_modes:
                for depth in depths:
                    res = analyze_needle_storage_fidelity(
                        model=model,
                        tok=tok,
                        total_len=total_len,
                        depth=depth,
                        needle_key="94821",
                        rope_mode=mode,
                        trained_len=base_trained_len,
                        h_ratio=h_ratio,
                    )

                    succ_str = "YES" if res["success"] else "NO"
                    print(f"{ratio:<8.2f} | {res['context_length']:<8d} | {depth:<8.2f} | {mode:<6} | {h_ratio:<8.2f} | {res['dominant_tier']:<8} | {res['mean_k_cosine_sim']:<10.4f} | {res['mean_v_cosine_sim']:<10.4f} | {res['generated_text'][:16]:<18} | {succ_str}")

                    row = {
                        "length_ratio": ratio,
                        "context_length": res["context_length"],
                        "depth": depth,
                        "rope_mode": mode,
                        "h_ratio": h_ratio,
                        "needle_tier": res["dominant_tier"],
                        "mean_k_cosine_sim": res["mean_k_cosine_sim"],
                        "mean_v_cosine_sim": res["mean_v_cosine_sim"],
                        "storage_intact": res["storage_intact"],
                        "generated_text": res["generated_text"],
                        "success": res["success"],
                    }
                    new_rows.append(row)

    # Combine existing and new rows, then sort by (length_ratio, h_ratio, rope_mode, depth)
    all_rows = existing_rows + new_rows
    all_rows.sort(key=lambda x: (x["length_ratio"], x["h_ratio"], x["rope_mode"], x["depth"]))

    save_results_to_csv(output_csv, all_rows)
    print(f"\n[Phase 4 Complete Dataset Saved] -> {output_csv} ({len(all_rows)} total rows)")
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-csv", type=str, default="benchmarks/results/niah_complete_graduated_dataset.csv")
    args = parser.parse_args()

    run_complete_phase4(model_name=args.model, output_csv=args.output_csv)
