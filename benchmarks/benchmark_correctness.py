#!/usr/bin/env python3
"""
benchmarks/benchmark_correctness.py
Evaluates:
1. Top-1 Greedy Token Agreement Rate over >= 500 generated tokens at 4k and 16k context length
   comparing TriTierCache vs Vanilla FP16 baseline.
2. Step-by-step Logit Numerical Drift (Cosine Similarity, MSE, Max Abs Diff).
"""
import os
import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from benchmarks.utils import generate_step_by_step
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def benchmark_top1_token_agreement(model_name: str,
                                   prompt_lengths: List[int] = None,
                                   gen_tokens: int = 500,
                                   run_config: RunConfig = None) -> List[Dict[str, Any]]:
    """Evaluates top-1 greedy decode token agreement between TriTierCache and Vanilla FP16 Cache."""
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 95)
    print(f" [1/2] BENCHMARKING TOP-1 GREEDY TOKEN AGREEMENT (gen_tokens={gen_tokens}, Model={model_name})")
    print("=" * 95)

    if prompt_lengths is None:
        prompt_lengths = [4096]

    model, tok = load_model(model_name, dtype=torch.float32)
    max_target = max(prompt_lengths)
    text_corpus = get_evaluation_corpus(min_tokens=max_target + 100, tokenizer=tok)
    input_tokens = tok(text_corpus, return_tensors="pt").input_ids

    results = []
    print(f"{'Context Length':<16} | {'Generated Tokens':<18} | {'Matching Tokens':<18} | {'Agreement (%)':<16} | {'First Divergence'}")
    print("-" * 95)

    for target_len in prompt_lengths:
        input_ids = input_tokens[:, :target_len]
        actual_len = input_ids.shape[1]

        # 1. Vanilla Baseline (Simulates standard FP16 decoding)
        remove_patch()
        reset_caches(model)
        vanilla_ids, _ = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        gen_vanilla = vanilla_ids[0, actual_len:].tolist()

        # 2. TriTierCache
        patch_mod.R_SIZE = run_config.R_size
        patch_mod.H_RATIO = run_config.H_ratio
        apply_patch()
        reset_caches(model)
        tritier_ids, _ = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        gen_tritier = tritier_ids[0, actual_len:].tolist()

        # Metrics
        matches = [1 if v == t else 0 for v, t in zip(gen_vanilla, gen_tritier)]
        match_count = sum(matches)
        match_pct = (match_count / max(1, len(matches))) * 100.0

        divergence_step = None
        for step_idx, m in enumerate(matches):
            if m == 0:
                divergence_step = step_idx + 1
                break

        div_str = "None (100% Exact)" if divergence_step is None else f"Step {divergence_step}"
        print(f"{actual_len:<16d} | {len(gen_vanilla):<18d} | {match_count:<18d} | {match_pct:<16.2f} | {div_str}")

        row = {
            "prompt_length": actual_len,
            "generated_tokens": len(gen_vanilla),
            "matching_tokens": match_count,
            "agreement_percentage": match_pct,
            "first_divergence_step": -1 if divergence_step is None else divergence_step,
        }
        row.update(run_config.to_dict())
        results.append(row)

    return results


def benchmark_numerical_drift(model_name: str,
                              prompt_len: int = 512,
                              decode_steps: int = 25,
                              run_config: RunConfig = None) -> List[Dict[str, Any]]:
    """Evaluates Cosine Similarity, MSE, and Max Abs Diff between Vanilla and TriTier logits."""
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 95)
    print(" [2/2] BENCHMARKING NUMERICAL DRIFT & COSINE SIMILARITY")
    print("=" * 95)

    model, tok = load_model(model_name, dtype=torch.float32)
    text_corpus = get_evaluation_corpus(min_tokens=prompt_len + 50, tokenizer=tok)
    input_ids = tok(text_corpus, return_tensors="pt").input_ids[:, :prompt_len]

    def run_and_collect_logits(use_tritier: bool):
        if use_tritier:
            patch_mod.R_SIZE = run_config.R_size
            patch_mod.H_RATIO = run_config.H_ratio
            apply_patch()
        else:
            remove_patch()
        reset_caches(model)

        logits_history = []
        curr_ids = input_ids.clone()
        past_kv = None

        with torch.no_grad():
            if not use_tritier:
                out = model(input_ids, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            else:
                out = model(input_ids)
                past_kv = None

            for step in range(decode_steps):
                last_logits = out.logits[:, -1, :].clone()
                logits_history.append(last_logits)
                next_id = last_logits.argmax(dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_id], dim=-1)
                pos = input_ids.shape[1] + step
                if not use_tritier:
                    out = model(next_id, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
                else:
                    out = model(next_id, position_ids=torch.tensor([[pos]], dtype=torch.int64))
                    past_kv = None

        return logits_history

    vanilla_logits = run_and_collect_logits(use_tritier=False)
    tritier_logits = run_and_collect_logits(use_tritier=True)

    drift_records = []
    print(f"{'Step':<6} | {'Cosine Sim':<14} | {'MSE':<14} | {'Max Abs Diff':<16} | {'Top-1 Match'}")
    print("-" * 95)

    for step, (v_log, t_log) in enumerate(zip(vanilla_logits, tritier_logits), start=1):
        cos_sim = F.cosine_similarity(v_log, t_log, dim=-1).item()
        mse = F.mse_loss(v_log, t_log).item()
        max_abs = torch.max(torch.abs(v_log - t_log)).item()
        v_top1 = v_log.argmax(dim=-1).item()
        t_top1 = t_log.argmax(dim=-1).item()
        match_str = "Match" if v_top1 == t_top1 else "Mismatch"

        print(f"{step:<6d} | {cos_sim:<14.6f} | {mse:<14.6f} | {max_abs:<16.6f} | {match_str}")

        row = {
            "step": step,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "max_abs_diff": max_abs,
            "top1_match": (v_top1 == t_top1),
        }
        row.update(run_config.to_dict())
        drift_records.append(row)

    return drift_records


def run_benchmark(model_name: str = DEFAULT_MODEL_ID, 
                  output_dir: str = "benchmarks/results",
                  quick: bool = False,
                  run_config: RunConfig = None) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    prompt_lens = [256] if quick else [4096, 16384]
    gen_tokens = 50 if quick else 500

    # 1. Top-1 Token Agreement
    match_results = benchmark_top1_token_agreement(model_name, prompt_lengths=prompt_lens, gen_tokens=gen_tokens, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "correctness_token_match.csv"), match_results)

    # 2. Numerical drift
    drift_results = benchmark_numerical_drift(model_name, prompt_len=prompt_lens[0], decode_steps=20, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "correctness_numerical_drift.csv"), drift_results)

    return {"token_match": match_results, "drift": drift_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Correctness & Verification Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_cfg = RunConfig(model_id=args.model, seed=args.seed)
    run_benchmark(model_name=args.model, output_dir=args.output_dir, quick=args.quick, run_config=run_cfg)
