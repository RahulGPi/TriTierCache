#!/usr/bin/env python3
"""
benchmarks/run_attribution_matrix.py
Phase 2 / Correctness Root-Cause Attribution Matrix (500-token test).
Isolates the step 24 -> 14 behavioral shift across four conditions:
  a) Neither fix (original baseline: old clamped RoPE, old kernel grp-16/fp32)
  b) RoPE fix only (corrected RoPE, old kernel grp-16/fp32)
  c) Kernel wiring fix only (old clamped RoPE, new kernel grp-32/fp16)
  d) Both fixes applied (corrected RoPE, new kernel grp-32/fp16)

Outputs:
  benchmarks/results/attribution_matrix_summary.csv
  benchmarks/results/attribution_matrix_drift.csv
"""

import os
import argparse
import time
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


def generate_with_condition(model,
                            input_ids: torch.Tensor,
                            gen_tokens: int,
                            condition_name: str,
                            rope_mode_type: str, # "old_clamped" vs "new_corrected"
                            k_group_size: int,
                            pbs_metadata_dtype: str,
                            vanilla_tokens: List[int],
                            vanilla_logits: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Runs 500-token generation under a specific experimental condition and compares with Vanilla.
    """
    seq_len = input_ids.shape[1]
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = k_group_size
    patch_mod.PBS_METADATA_DTYPE = pbs_metadata_dtype
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()
    reset_caches(model)

    gen_ids = []
    cos_sims = []
    mses = []
    matches = []

    curr_ids = input_ids.clone()
    first_div_step = None

    with torch.no_grad():
        # Prefill prompt
        out = model(curr_ids)

        for step in range(gen_tokens):
            last_logits = out.logits[:, -1, :].clone().float()
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            tok_val = next_id.item()
            gen_ids.append(tok_val)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)

            v_tok = vanilla_tokens[step]
            v_logit = vanilla_logits[step]

            is_match = (tok_val == v_tok)
            matches.append(1 if is_match else 0)

            cos = F.cosine_similarity(v_logit, last_logits, dim=-1).item()
            mse = F.mse_loss(v_logit, last_logits).item()
            cos_sims.append(cos)
            mses.append(mse)

            if not is_match and first_div_step is None:
                first_div_step = step + 1 # 1-indexed

            # Position calculation for next token
            pos = seq_len + step
            if rope_mode_type == "old_clamped":
                # Old bugged RoPE: clamped to 256 + 4
                rel_pos = min(pos, 256 + 4)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)
            else:
                # New corrected RoPE: standard monotonic position within trained context
                pos_tensor = torch.tensor([[pos]], dtype=torch.int64)

            out = model(next_id, position_ids=pos_tensor)

    match_count = sum(matches)
    match_pct = (match_count / gen_tokens) * 100.0

    print(f"Condition: {condition_name:<30} | First Div: Step {str(first_div_step):<5} | Agreement: {match_pct:6.2f}% ({match_count}/{gen_tokens}) | Mean Cos: {np.mean(cos_sims):.4f}")

    return {
        "condition": condition_name,
        "rope_type": rope_mode_type,
        "k_group_size": k_group_size,
        "pbs_metadata_dtype": pbs_metadata_dtype,
        "first_divergence_step": -1 if first_div_step is None else first_div_step,
        "agreement_pct": match_pct,
        "matching_tokens": match_count,
        "total_tokens": gen_tokens,
        "mean_cosine_sim": float(np.mean(cos_sims)),
        "min_cosine_sim": float(np.min(cos_sims)),
        "mean_mse": float(np.mean(mses)),
        "div_step_cos": cos_sims[first_div_step - 1] if first_div_step else 1.0,
        "div_step_mse": mses[first_div_step - 1] if first_div_step else 0.0,
        "cos_sims": cos_sims,
        "mses": mses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 115)
    print(f" [PHASE 2] ROOT-CAUSE ATTRIBUTION MATRIX ({args.gen_tokens}-Token Test, Model: {args.model})")
    print(f" Prompt Length: {args.prompt_len} tokens")
    print("=" * 115)

    model, tok = load_model(args.model, dtype=torch.float32)
    corpus = get_evaluation_corpus(min_tokens=args.prompt_len + 100, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :args.prompt_len]

    # 1. Run Vanilla Baseline ONCE to obtain reference trajectory
    cache_ref_path = os.path.join(args.output_dir, "vanilla_ref_4096_500.pt")
    if os.path.exists(cache_ref_path):
        print(f"\n--- Loading cached Vanilla FP32 reference from {cache_ref_path} ---")
        cached_data = torch.load(cache_ref_path, weights_only=False)
        vanilla_tokens = cached_data["tokens"]
        vanilla_logits = cached_data["logits"]
        print(f"Loaded {len(vanilla_tokens)} reference tokens.")
    else:
        print("\n--- Running Vanilla FP32 Baseline (500 tokens) ---")
        remove_patch()
        reset_caches(model)
        vanilla_tokens = []
        vanilla_logits = []
        curr_ids = input_ids.clone()
        past_kv = None

        t0 = time.time()
        with torch.no_grad():
            out = model(curr_ids, use_cache=True)
            past_kv = out.past_key_values
            for step in range(args.gen_tokens):
                last_logits = out.logits[:, -1, :].clone().float()
                vanilla_logits.append(last_logits)
                next_id = last_logits.argmax(dim=-1, keepdim=True)
                vanilla_tokens.append(next_id.item())
                curr_ids = torch.cat([curr_ids, next_id], dim=-1)
                out = model(next_id, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
        print(f"Vanilla decode complete in {time.time() - t0:.2f}s")
        torch.save({"tokens": vanilla_tokens, "logits": vanilla_logits}, cache_ref_path)
        print(f"Saved Vanilla reference to {cache_ref_path}")


    # 2. Run the 4 conditions
    conditions = [
        ("a_neither_fix", "old_clamped", 16, "fp32"),
        ("b_rope_fix_only", "new_corrected", 16, "fp32"),
        ("c_kernel_wiring_only", "old_clamped", 32, "fp16"),
        ("d_both_fixes", "new_corrected", 32, "fp16"),
    ]

    summary_rows = []
    drift_rows = []

    print("\n" + "-" * 115)
    print(f"{'Condition':<25} | {'RoPE Type':<14} | {'Kernel Config':<15} | {'First Div':<10} | {'Agreement':<14} | {'Mean Cos':<10} | {'Div Step MSE'}")
    print("-" * 115)

    for cond_name, rope_type, k_grp, meta_dtype in conditions:
        res = generate_with_condition(
            model=model,
            input_ids=input_ids,
            gen_tokens=args.gen_tokens,
            condition_name=cond_name,
            rope_mode_type=rope_type,
            k_group_size=k_grp,
            pbs_metadata_dtype=meta_dtype,
            vanilla_tokens=vanilla_tokens,
            vanilla_logits=vanilla_logits,
        )

        kernel_str = f"grp{k_grp}_{meta_dtype}"
        div_str = f"Step {res['first_divergence_step']}" if res['first_divergence_step'] > 0 else "None"
        print(f"{cond_name:<25} | {rope_type:<14} | {kernel_str:<15} | {div_str:<10} | {res['agreement_pct']:6.2f}%       | {res['mean_cosine_sim']:<10.4f} | {res['div_step_mse']:.6f}")

        summary_rows.append({
            "condition": cond_name,
            "rope_type": rope_type,
            "k_group_size": k_grp,
            "pbs_metadata_dtype": meta_dtype,
            "first_divergence_step": res["first_divergence_step"],
            "agreement_percentage": res["agreement_pct"],
            "matching_tokens": res["matching_tokens"],
            "total_tokens": res["total_tokens"],
            "mean_cosine_sim": res["mean_cosine_sim"],
            "min_cosine_sim": res["min_cosine_sim"],
            "mean_mse": res["mean_mse"],
            "divergence_step_cosine_sim": res["div_step_cos"],
            "divergence_step_mse": res["div_step_mse"],
        })

        for s in range(min(50, args.gen_tokens)):
            drift_rows.append({
                "condition": cond_name,
                "step": s + 1,
                "cosine_similarity": res["cos_sims"][s],
                "mse": res["mses"][s],
                "matches_vanilla": bool(res["cos_sims"][s] > 0.999), # approx
            })

    save_results_to_csv(os.path.join(args.output_dir, "attribution_matrix_summary.csv"), summary_rows)
    save_results_to_csv(os.path.join(args.output_dir, "attribution_matrix_drift.csv"), drift_rows)


if __name__ == "__main__":
    main()
