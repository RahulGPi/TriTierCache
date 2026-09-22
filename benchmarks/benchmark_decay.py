#!/usr/bin/env python3
"""
benchmarks/benchmark_decay.py
Phase 2: Heavy Hitter Decay vs Undecayed Accumulation over >= 2,000 decode steps.
Tracks:
1. Composition of Heavy-Hitter tier (which token IDs are retained) at steps 100, 500, 1000, 2000.
2. Jaccard similarity of retained token ID sets between decay=0.999 vs decay=None.
3. Logit cosine similarity and Top-1 token agreement between the two settings.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Set, Tuple

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


def extract_hh_token_ids(model) -> Set[int]:
    """Collects the union of all active Heavy-Hitter token IDs across all model layers."""
    all_hh_ids = set()
    for name, module in model.named_modules():
        if hasattr(module, "tri_tier_cache") and module.tri_tier_cache is not None:
            cache = module.tri_tier_cache
            if hasattr(cache, "HH_token_ids") and cache.HH_token_ids is not None:
                t_ids = cache.HH_token_ids
                if isinstance(t_ids, torch.Tensor):
                    valid = t_ids[t_ids >= 0].tolist()
                    all_hh_ids.update(valid)
    return all_hh_ids


def run_decay_comparison(model_name: str = DEFAULT_MODEL_ID,
                         prompt_len: int = 256,
                         decode_steps: int = 2000,
                         checkpoints: List[int] = None,
                         output_dir: str = "benchmarks/results",
                         seed: int = 42) -> Dict[str, Any]:
    if checkpoints is None:
        checkpoints = [100, 500, 1000, 2000]

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 115)
    print(f" [PHASE 2] LONG-SEQUENCE HEAVY-HITTER DECAY VALIDATION ({decode_steps} Steps, Model: {model_name})")
    print(f" Checkpoints: {checkpoints} | Decay: 0.999 vs None (1.000) | Seed: {seed}")
    print("=" * 115)

    model, tok = load_model(model_name, dtype=torch.float32)
    corpus = get_evaluation_corpus(min_tokens=prompt_len + 50, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :prompt_len]

    def execute_run(score_decay: float, label: str):
        print(f"\nExecuting {label} (decay={score_decay})...")
        patch_mod.R_SIZE = 256
        patch_mod.H_RATIO = 0.05
        patch_mod.SCORE_DECAY = score_decay
        patch_mod.K_GROUP_SIZE = 16
        patch_mod.PBS_METADATA_DTYPE = "fp16"
        apply_patch()
        reset_caches(model)

        hh_snapshots: Dict[int, Set[int]] = {}
        logits_history: List[torch.Tensor] = []
        curr_ids = input_ids.clone()

        with torch.no_grad():
            out = model(curr_ids)
            for step in range(1, decode_steps + 1):
                last_logits = out.logits[:, -1, :].clone().float()
                logits_history.append(last_logits)
                next_id = last_logits.argmax(dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_id], dim=-1)
                pos = prompt_len + step - 1

                # Absolute position indexing
                pos_tensor = torch.tensor([[pos]], dtype=torch.int64)
                out = model(next_id, position_ids=pos_tensor)

                if step in checkpoints:
                    hh_snapshots[step] = extract_hh_token_ids(model)
                    print(f"  [{label}] Step {step:4d}: {len(hh_snapshots[step])} unique HH tokens retained")

        return hh_snapshots, logits_history

    # Run 1: Undecayed (decay = 1.0)
    hh_undecayed, logits_undecayed = execute_run(score_decay=1.0, label="Undecayed (hh_decay=None)")

    # Run 2: Decayed (decay = 0.999)
    hh_decayed, logits_decayed = execute_run(score_decay=0.999, label="Decayed (hh_decay=0.999)")

    # Compare checkpoints
    comparison_rows = []
    print("\n" + "=" * 115)
    print(" HEAVY-HITTER TIER COMPOSITION DIVERGENCE ANALYSIS ACROSS DECODE CHECKPOINTS")
    print("=" * 115)
    print(f"{'Checkpoint':<12} | {'Undecayed HH Count':<20} | {'Decayed HH Count':<18} | {'Overlap Count':<16} | {'Jaccard Sim':<14} | {'Retained Sets Match'}")
    print("-" * 115)

    for step in sorted(checkpoints):
        if step > decode_steps:
            continue
        set_u = hh_undecayed.get(step, set())
        set_d = hh_decayed.get(step, set())

        intersection = set_u.intersection(set_d)
        union = set_u.union(set_d)
        jaccard = len(intersection) / len(union) if union else 1.0
        match_str = "YES" if set_u == set_d else "NO (Diverged)"

        print(f"{step:<12d} | {len(set_u):<20d} | {len(set_d):<18d} | {len(intersection):<16d} | {jaccard:<14.4f} | {match_str}")

        # Sample divergent token IDs if sets differ
        only_in_undecayed = sorted(list(set_u - set_d))[:5]
        only_in_decayed = sorted(list(set_d - set_u))[:5]

        row = {
            "checkpoint_step": step,
            "undecayed_hh_count": len(set_u),
            "decayed_hh_count": len(set_d),
            "overlap_count": len(intersection),
            "jaccard_similarity": jaccard,
            "sets_identical": (set_u == set_d),
            "only_in_undecayed_sample": str(only_in_undecayed),
            "only_in_decayed_sample": str(only_in_decayed),
            "model_id": model_name,
            "prompt_len": prompt_len,
            "seed": seed,
        }
        comparison_rows.append(row)

    # Compute logit drift at the checkpoints
    print("\n" + "=" * 115)
    print(" LOGIT DRIFT & TOP-1 TOKEN AGREEMENT: DECAYED vs UNDECAYED")
    print("=" * 115)
    print(f"{'Step Range':<16} | {'Mean Cosine Sim':<18} | {'Min Cosine Sim':<16} | {'Top-1 Agreement (%)'}")
    print("-" * 115)

    step_cos_sims = []
    step_matches = []
    for step_idx, (l_u, l_d) in enumerate(zip(logits_undecayed, logits_decayed), start=1):
        cos = F.cosine_similarity(l_u, l_d, dim=-1).item()
        match = bool(l_u.argmax(dim=-1) == l_d.argmax(dim=-1))
        step_cos_sims.append(cos)
        step_matches.append(1 if match else 0)

        if step_idx in checkpoints:
            slice_cos = step_cos_sims[max(0, step_idx - 100):step_idx]
            slice_match = step_matches[max(0, step_idx - 100):step_idx]
            mean_c = float(np.mean(slice_cos))
            min_c = float(np.min(slice_cos))
            agr_pct = (sum(slice_match) / len(slice_match)) * 100.0
            print(f"Step 1 - {step_idx:<9d} | {mean_c:<18.6f} | {min_c:<16.6f} | {agr_pct:.1f}%")

    # Save to CSV
    csv_path = os.path.join(output_dir, "hh_decay_comparison_results.csv")
    save_results_to_csv(csv_path, comparison_rows)
    print(f"\n[CSV Saved] -> {csv_path}")

    # Reset patch defaults
    patch_mod.SCORE_DECAY = 0.999
    return {"checkpoints": comparison_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heavy-Hitter Decay Validation Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or checkpoint path")
    parser.add_argument("--prompt-len", type=int, default=256, help="Prompt length")
    parser.add_argument("--decode-steps", type=int, default=2000, help="Total decode steps (>=2000)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_decay_comparison(
        model_name=args.model,
        prompt_len=args.prompt_len,
        decode_steps=args.decode_steps,
        output_dir=args.output_dir,
        seed=args.seed
    )
