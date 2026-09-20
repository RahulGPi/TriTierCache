#!/usr/bin/env python3
"""
benchmarks/run_isolation_grid.py
Phase 1.3: Root-Cause Isolation Grid.
Evaluates 16 orthogonal combinations of:
- rope_mode: 'a' vs 'b'
- K_group_size: 16 vs 32
- pbs_metadata_dtype: 'fp16' vs 'fp32'
- hh_decay: None (off) vs 0.999 (on)
Using fast numerical drift diagnostic (20 steps @ 512 prompt).
Outputs:
  benchmarks/results/isolation_grid_results.csv
"""

import os
import itertools
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def evaluate_config_numerical_drift(model, 
                                    input_ids: torch.Tensor, 
                                    run_cfg: RunConfig, 
                                    decode_steps: int = 20) -> Dict[str, Any]:
    """
    Evaluates step-by-step numerical drift (Cosine Sim, MSE) for a single RunConfig.
    """
    seq_len = input_ids.shape[1]

    # 1. Run Vanilla Baseline
    remove_patch()
    reset_caches(model)
    vanilla_logits = []
    curr_ids = input_ids.clone()
    past_kv = None

    with torch.no_grad():
        out = model(curr_ids, use_cache=True)
        past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
        for step in range(decode_steps):
            last_logits = out.logits[:, -1, :].clone().float()
            vanilla_logits.append(last_logits)
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            out = model(next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values if hasattr(out, "past_key_values") else None

    # 2. Configure and Run TriTierCache
    patch_mod.R_SIZE = run_cfg.R_size
    patch_mod.H_RATIO = run_cfg.H_ratio
    apply_patch()
    reset_caches(model)

    tritier_logits = []
    curr_ids = input_ids.clone()

    with torch.no_grad():
        out = model(curr_ids)
        for step in range(decode_steps):
            last_logits = out.logits[:, -1, :].clone().float()
            tritier_logits.append(last_logits)
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            pos = seq_len + step
            
            # rope_mode 'a' = absolute position; 'b' = relative / window position
            if run_cfg.rope_mode == "a":
                pos_tensor = torch.tensor([[pos]], dtype=torch.int64)
            else:
                # StreamingLLM style: relative position capped within recent window + sinks
                rel_pos = min(pos, run_cfg.R_size + 4)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)

            out = model(next_id, position_ids=pos_tensor)

    # 3. Compute drift metrics across steps
    step_cos_sims = []
    step_mses = []
    step_matches = []

    for v_log, t_log in zip(vanilla_logits, tritier_logits):
        cos_sim = F.cosine_similarity(v_log, t_log, dim=-1).mean().item()
        mse = F.mse_loss(v_log, t_log).item()
        top1_match = bool(v_log.argmax(dim=-1) == t_log.argmax(dim=-1))

        step_cos_sims.append(cos_sim)
        step_mses.append(mse)
        step_matches.append(1 if top1_match else 0)

    mean_cos = float(np.mean(step_cos_sims))
    min_cos = float(np.min(step_cos_sims))
    std_cos = float(np.std(step_cos_sims))
    mean_mse = float(np.mean(step_mses))
    match_pct = (sum(step_matches) / len(step_matches)) * 100.0

    res = {
        "mean_cosine_sim": mean_cos,
        "min_cosine_sim": min_cos,
        "std_cosine_sim": std_cos,
        "mean_mse": mean_mse,
        "top1_match_pct_20steps": match_pct,
        "step_cos_sims_str": ",".join([f"{s:.4f}" for s in step_cos_sims]),
    }
    res.update(run_cfg.to_dict())
    return res


def run_isolation_grid(model_id: str = DEFAULT_MODEL_ID,
                       prompt_len: int = 512,
                       output_dir: str = "benchmarks/results",
                       seed: int = 42) -> List[Dict[str, Any]]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 105)
    print(f" [PHASE 1.3] RUNNING 16-COMBINATION ROOT-CAUSE ISOLATION GRID (Model: {model_id})")
    print("=" * 105)

    model, tok = load_model(model_id)
    corpus = get_evaluation_corpus(min_tokens=prompt_len + 50, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :prompt_len]

    # Grid options: 2 x 2 x 2 x 2 = 16 combos
    rope_modes = ["a", "b"]
    k_group_sizes = [16, 32]
    pbs_dtypes = ["fp16", "fp32"]
    hh_decays = [None, 0.999]

    combos = list(itertools.product(rope_modes, k_group_sizes, pbs_dtypes, hh_decays))
    results = []

    print(f"{'Combo':<6} | {'RoPE':<6} | {'K Group':<8} | {'PBS Meta':<9} | {'HH Decay':<9} | {'Mean Cos Sim':<14} | {'Min Cos Sim':<14} | {'Top-1 (%)'}")
    print("-" * 105)

    for idx, (rope_m, k_grp, pbs_dt, hh_dec) in enumerate(combos, start=1):
        cfg = RunConfig(
            model_id=model_id,
            rope_mode=rope_m,
            K_group_size=k_grp,
            pbs_metadata_dtype=pbs_dt,
            hh_decay=hh_dec,
            seed=seed,
            R_size=256,
            H_ratio=0.05
        )

        res = evaluate_config_numerical_drift(model, input_ids, cfg, decode_steps=20)
        results.append(res)

        decay_str = "None" if hh_dec is None else f"{hh_dec:.3f}"
        print(f"{idx:<6d} | {rope_m:<6} | {k_grp:<8d} | {pbs_dt:<9} | {decay_str:<9} | {res['mean_cosine_sim']:<14.6f} | {res['min_cosine_sim']:<14.6f} | {res['top1_match_pct_20steps']:.1f}%")

    # Sort results by min_cosine_sim (healthiest configs first)
    sorted_results = sorted(results, key=lambda r: (r["min_cosine_sim"], r["mean_cosine_sim"]), reverse=True)

    print("\n" + "=" * 105)
    print(" TOP 3 HEALTHIEST CONFIGURATIONS IN ISOLATION GRID")
    print("=" * 105)
    for rank, r in enumerate(sorted_results[:3], start=1):
        print(f"Rank {rank}: RoPE={r['rope_mode']}, K_group={r['K_group_size']}, Meta={r['pbs_metadata_dtype']}, Decay={r['hh_decay_str']} -> Mean Cos={r['mean_cosine_sim']:.4f}, Min Cos={r['min_cosine_sim']:.4f}")
    print("=" * 105)

    # Save to CSV
    csv_path = os.path.join(output_dir, "isolation_grid_results.csv")
    save_results_to_csv(csv_path, results)
    print(f"Full isolation grid results saved -> {csv_path}")

    return sorted_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Root-Cause Isolation Grid Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or checkpoint path")
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt length for drift test")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_isolation_grid(model_id=args.model, prompt_len=args.prompt_len, output_dir=args.output_dir, seed=args.seed)
