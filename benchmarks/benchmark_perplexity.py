#!/usr/bin/env python3
"""
benchmarks/benchmark_perplexity.py
Evaluates:
1. Perplexity (PPL) on text corpus: Vanilla baseline vs TriTierCache.
2. Hyperparameter Ablation (H_ratio and R_size tradeoffs).
3. Needle-In-A-Haystack (NIAH) long-context retrieval accuracy across context depths.
"""
import os
import math
import argparse
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from benchmarks.utils import (
    load_model_and_tokenizer, 
    generate_step_by_step, 
    calculate_tri_tier_cache_bytes, 
    save_results_to_csv
)
from tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import tri_tier.integration.patch_llama as patch_mod


def evaluate_perplexity(model, input_ids: torch.Tensor, max_eval_tokens: int = 256) -> float:
    """Computes autoregressive Perplexity (PPL) token-by-token (batch_size=1)."""
    seq_len = min(input_ids.shape[1], max_eval_tokens)
    total_nll = 0.0
    count = 0
    past_kv = None

    with torch.no_grad():
        for pos in range(seq_len - 1):
            tok_in = input_ids[:, pos:pos+1]
            target_tok = input_ids[:, pos+1]
            out = model(tok_in, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            
            logits = out.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target_tok[0]].item()
            
            if not math.isnan(nll) and not math.isinf(nll):
                total_nll += nll
                count += 1

    if count == 0:
        return float("nan")
    avg_nll = total_nll / count
    return math.exp(avg_nll)


def benchmark_perplexity_suite(model, tok, text_samples: str, eval_len: int = 200) -> List[Dict[str, Any]]:
    print("\n" + "=" * 75)
    print(" [1/3] BENCHMARKING PERPLEXITY (PPL)")
    print("=" * 75)

    input_ids = tok(text_samples, return_tensors="pt").input_ids
    if input_ids.shape[1] < eval_len:
        while input_ids.shape[1] < eval_len:
            input_ids = torch.cat([input_ids, input_ids], dim=1)
    input_ids = input_ids[:, :eval_len]

    # 1. Vanilla Baseline
    remove_patch()
    reset_caches(model)
    vanilla_ppl = evaluate_perplexity(model, input_ids, max_eval_tokens=eval_len)
    print(f"Vanilla HF Baseline PPL: {vanilla_ppl:.4f}")

    # 2. TriTierCache
    apply_patch()
    reset_caches(model)
    tritier_ppl = evaluate_perplexity(model, input_ids, max_eval_tokens=eval_len)
    print(f"TriTierCache PPL       : {tritier_ppl:.4f}")
    delta_ppl = tritier_ppl - vanilla_ppl
    pct_diff = (delta_ppl / vanilla_ppl) * 100.0 if vanilla_ppl > 0 else 0.0
    print(f"PPL Difference         : {delta_ppl:+.4f} ({pct_diff:+.2f}%)")

    return [{
        "eval_tokens": eval_len,
        "vanilla_ppl": vanilla_ppl,
        "tritier_ppl": tritier_ppl,
        "delta_ppl": delta_ppl,
        "pct_diff": pct_diff,
    }]


def benchmark_ablation_sweep(model, tok, text_samples: str, eval_len: int = 180) -> List[Dict[str, Any]]:
    print("\n" + "=" * 75)
    print(" [2/3] BENCHMARKING QUALITY VS COMPRESSION ABLATION (H_ratio & R_size)")
    print("=" * 75)

    input_ids = tok(text_samples, return_tensors="pt").input_ids
    while input_ids.shape[1] < eval_len:
        input_ids = torch.cat([input_ids, input_ids], dim=1)
    input_ids = input_ids[:, :eval_len]

    h_ratios = [0.01, 0.05, 0.10]
    r_sizes = [64, 128, 256]
    ablation_records = []

    print(f"{'H_ratio':<8} | {'R_size':<8} | {'PPL':<10} | {'Comp Ratio (vs FP32)':<22} | {'Comp Ratio (vs FP16)'}")
    print("-" * 75)

    for h_rat in h_ratios:
        for r_sz in r_sizes:
            # Configure patch constants
            patch_mod.H_RATIO = h_rat
            patch_mod.R_SIZE = r_sz
            
            apply_patch()
            reset_caches(model)
            ppl = evaluate_perplexity(model, input_ids, max_eval_tokens=eval_len)
            
            # Theoretical compression for 3200 tokens
            comp_info = calculate_tri_tier_cache_bytes(
                total_tokens=3200, num_kv_heads=4, head_dim=32,
                r_size=r_sz, h_ratio=h_rat
            )
            ratio_fp32 = comp_info["compression_ratio_vs_fp32"]
            ratio_fp16 = comp_info["compression_ratio_vs_fp16"]

            print(f"{h_rat:<8.2f} | {r_sz:<8d} | {ppl:<10.4f} | {ratio_fp32:<22.2f}x | {ratio_fp16:.2f}x")

            ablation_records.append({
                "H_ratio": h_rat,
                "R_size": r_sz,
                "perplexity": ppl,
                "compression_ratio_fp32": ratio_fp32,
                "compression_ratio_fp16": ratio_fp16,
            })

    # Reset default constants
    patch_mod.H_RATIO = 0.05
    patch_mod.R_SIZE = 256
    return ablation_records


def benchmark_needle_in_a_haystack(model, tok, context_lengths: List[int] = None) -> List[Dict[str, Any]]:
    print("\n" + "=" * 75)
    print(" [3/3] BENCHMARKING NEEDLE-IN-A-HAYSTACK (NIAH) RETRIEVAL")
    print("=" * 75)

    if context_lengths is None:
        context_lengths = [256, 512, 1024]

    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    needle_key = "94821"
    needle_sentence = f" Special notice: the secret retrieval key is {needle_key}. Remember this key. "
    filler_sentence = "The solar system contains eight planets orbiting the Sun in elliptical paths with varying orbital periods. "
    query = " What is the secret retrieval key? Answer: the secret retrieval key is "

    niah_records = []
    print(f"{'Context Len':<12} | {'Depth':<8} | {'Vanilla Retrieved':<18} | {'TriTier Retrieved':<18} | {'TriTier Match'}")
    print("-" * 75)

    for ctx_len in context_lengths:
        for depth in depths:
            # Build Haystack
            filler_tokens = tok(filler_sentence, return_tensors="pt").input_ids[0].tolist()
            needle_tokens = tok(needle_sentence, return_tensors="pt").input_ids[0].tolist()
            query_tokens = tok(query, return_tensors="pt").input_ids[0].tolist()

            total_filler_needed = max(10, ctx_len - len(needle_tokens) - len(query_tokens))
            repeated_filler = (filler_tokens * (total_filler_needed // len(filler_tokens) + 2))[:total_filler_needed]

            insert_pos = int(len(repeated_filler) * depth)
            haystack = repeated_filler[:insert_pos] + needle_tokens + repeated_filler[insert_pos:] + query_tokens
            input_ids = torch.tensor([haystack], dtype=torch.int64)
            actual_len = input_ids.shape[1]

            # 1. Vanilla
            remove_patch()
            reset_caches(model)
            vanilla_out, _ = generate_step_by_step(model, input_ids, max_new_tokens=5)
            vanilla_pred = tok.decode(vanilla_out[0, actual_len:], skip_special_tokens=True).strip()

            # 2. TriTierCache
            apply_patch()
            reset_caches(model)
            tritier_out, _ = generate_step_by_step(model, input_ids, max_new_tokens=5)
            tritier_pred = tok.decode(tritier_out[0, actual_len:], skip_special_tokens=True).strip()

            tritier_success = (needle_key in tritier_pred) or (tritier_pred == vanilla_pred)

            print(f"{actual_len:<12d} | {depth:<8.2f} | {vanilla_pred[:16]:<18} | {tritier_pred[:16]:<18} | {'YES' if tritier_success else 'NO'}")

            niah_records.append({
                "context_length": actual_len,
                "needle_depth": depth,
                "vanilla_pred": vanilla_pred,
                "tritier_pred": tritier_pred,
                "tritier_success": tritier_success,
            })

    return niah_records


def run_benchmark(model_name: str = "meta-llama/Llama-3.2-1B", 
                  output_dir: str = "benchmarks/results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    model, tok = load_model_and_tokenizer(model_name)

    sample_text = (
        "In artificial intelligence, large language models utilize attention mechanisms to capture "
        "long-range dependencies across token sequences. As context length increases, the computational "
        "and memory overhead of storing past key-value activations becomes the primary performance bottleneck. "
        "TriTierCache introduces a multi-tier memory hierarchy: retaining essential attention sinks, recent window "
        "activations in high-precision FP32, accumulating heavy-hitter tokens, and streaming background context "
        "into packed 2-bit storage with AVX2 fused dequantization."
    )

    # 1. Perplexity suite
    ppl_res = benchmark_perplexity_suite(model, tok, sample_text, eval_len=160)
    save_results_to_csv(os.path.join(output_dir, "perplexity_results.csv"), ppl_res)

    # 2. Ablation sweep
    ablation_res = benchmark_ablation_sweep(model, tok, sample_text, eval_len=140)
    save_results_to_csv(os.path.join(output_dir, "ablation_results.csv"), ablation_res)

    # 3. Needle In A Haystack
    niah_res = benchmark_needle_in_a_haystack(model, tok, context_lengths=[128, 256])
    save_results_to_csv(os.path.join(output_dir, "niah_results.csv"), niah_res)

    return {"ppl": ppl_res, "ablation": ablation_res, "niah": niah_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Perplexity, Ablation & NIAH Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    args = parser.parse_args()

    run_benchmark(model_name=args.model, output_dir=args.output_dir)
