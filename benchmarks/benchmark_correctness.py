#!/usr/bin/env python3
"""
benchmarks/benchmark_correctness.py
Evaluates output token match and layer-by-layer numerical drift (Cosine Similarity, MSE)
between TriTierCache and Vanilla HF Attention.
"""
import os
import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from benchmarks.utils import load_model_and_tokenizer, generate_step_by_step, save_results_to_csv
from tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches


def benchmark_token_match(model, tok, prompt_lengths: List[int], gen_tokens: int = 30) -> List[Dict[str, Any]]:
    """Evaluates greedy decode token match between TriTierCache and Vanilla HF Cache."""
    results = []
    base_text = ("The rapid evolution of artificial intelligence and deep learning algorithms "
                 "has transformed natural language processing, computer vision, and high-performance computing. "
                 "Memory hierarchy and KV cache optimizations are essential for scaling context windows. ")
    
    print("\n" + "=" * 75)
    print(" [1/2] BENCHMARKING OUTPUT TOKEN MATCH (Greedy Decode)")
    print("=" * 75)

    for target_len in prompt_lengths:
        # Build prompt of target token length
        tokens = tok(base_text, return_tensors="pt").input_ids
        while tokens.shape[1] < target_len:
            tokens = torch.cat([tokens, tokens], dim=1)
        input_ids = tokens[:, :target_len]
        actual_len = input_ids.shape[1]

        # 1. Vanilla Baseline
        remove_patch()
        reset_caches(model)
        vanilla_ids, _ = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        gen_vanilla = vanilla_ids[0, actual_len:].tolist()

        # 2. TriTierCache
        apply_patch()
        reset_caches(model)
        tritier_ids, _ = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
        gen_tritier = tritier_ids[0, actual_len:].tolist()

        # Metrics
        matches = [1 if v == t else 0 for v, t in zip(gen_vanilla, gen_tritier)]
        match_count = sum(matches)
        match_pct = (match_count / gen_tokens) * 100.0
        
        divergence_step = None
        for step_idx, m in enumerate(matches):
            if m == 0:
                divergence_step = step_idx + 1
                break

        print(f"Prompt Length: {actual_len:5d} tokens | Match: {match_count:2d}/{gen_tokens} ({match_pct:5.1f}%) | "
              f"Divergence: {'None' if divergence_step is None else f'Step {divergence_step}'}")

        results.append({
            "prompt_length": actual_len,
            "generated_tokens": gen_tokens,
            "matching_tokens": match_count,
            "match_percentage": match_pct,
            "first_divergence_step": -1 if divergence_step is None else divergence_step,
        })

    return results


def benchmark_numerical_drift(model, tok, prompt_len: int = 64, decode_steps: int = 25) -> List[Dict[str, Any]]:
    """Evaluates Cosine Similarity, MSE, and Max Abs Diff between Vanilla and TriTier logits."""
    print("\n" + "=" * 75)
    print(" [2/2] BENCHMARKING NUMERICAL DRIFT & COSINE SIMILARITY")
    print("=" * 75)

    base_text = "Transformer architectures rely heavily on self-attention mechanisms and key-value caching. "
    tokens = tok(base_text, return_tensors="pt").input_ids
    while tokens.shape[1] < prompt_len:
        tokens = torch.cat([tokens, tokens], dim=1)
    input_ids = tokens[:, :prompt_len]

    # Collect logits step-by-step
    def run_and_collect_logits(use_tritier: bool):
        if use_tritier:
            apply_patch()
        else:
            remove_patch()
        reset_caches(model)
        
        logits_history = []
        curr_ids = input_ids.clone()
        past_kv = None
        
        with torch.no_grad():
            for pos in range(prompt_len):
                tok_in = curr_ids[:, pos:pos+1]
                out = model(tok_in, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            
            for _ in range(decode_steps):
                last_logits = out.logits[:, -1, :].clone()
                logits_history.append(last_logits)
                next_id = last_logits.argmax(dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_id], dim=-1)
                out = model(next_id, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None

        return logits_history

    vanilla_logits = run_and_collect_logits(use_tritier=False)
    tritier_logits = run_and_collect_logits(use_tritier=True)

    drift_records = []
    print(f"{'Step':<6} | {'Cosine Sim':<12} | {'MSE':<12} | {'Max Abs Diff':<14} | {'Top-1 Token Match'}")
    print("-" * 75)

    for step, (v_log, t_log) in enumerate(zip(vanilla_logits, tritier_logits), start=1):
        cos_sim = F.cosine_similarity(v_log.float(), t_log.float(), dim=-1).mean().item()
        mse = F.mse_loss(v_log.float(), t_log.float()).item()
        max_diff = torch.max(torch.abs(v_log.float() - t_log.float())).item()
        top1_match = bool(v_log.argmax(dim=-1) == t_log.argmax(dim=-1))

        print(f"{step:<6d} | {cos_sim:<12.6f} | {mse:<12.6f} | {max_diff:<14.6f} | {'Match' if top1_match else 'Mismatch'}")

        drift_records.append({
            "decode_step": step,
            "cosine_similarity": cos_sim,
            "mse": mse,
            "max_abs_diff": max_diff,
            "top1_match": top1_match,
        })

    return drift_records


def run_benchmark(model_name: str = "meta-llama/Llama-3.2-1B", 
                  output_dir: str = "benchmarks/results",
                  prompt_lens: List[int] = None) -> Dict[str, Any]:
    if prompt_lens is None:
        prompt_lens = [16, 64, 128, 280]

    os.makedirs(output_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_name)

    # 1. Token match
    match_results = benchmark_token_match(model, tok, prompt_lengths=prompt_lens, gen_tokens=25)
    save_results_to_csv(os.path.join(output_dir, "correctness_token_match.csv"), match_results)

    # 2. Numerical drift
    drift_results = benchmark_numerical_drift(model, tok, prompt_len=prompt_lens[1], decode_steps=20)
    save_results_to_csv(os.path.join(output_dir, "correctness_numerical_drift.csv"), drift_results)

    return {"token_match": match_results, "drift": drift_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Correctness & Verification Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    args = parser.parse_args()

    run_benchmark(model_name=args.model, output_dir=args.output_dir)
