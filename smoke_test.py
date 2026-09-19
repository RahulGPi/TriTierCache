#!/usr/bin/env python3
"""
smoke_test.py - Smoke test verifying generation with TriTierCache vs Vanilla HF Cache.
"""
import sys
import torch
from benchmarks.utils import load_model_and_tokenizer, generate_step_by_step, save_results_to_csv
from tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B"
    prompt = "The quick brown fox jumps over the lazy dog and"
    gen_tokens = 20

    print("=" * 70)
    print(f" TriTierCache Smoke Test: {model_name}")
    print("=" * 70)

    model, tok = load_model_and_tokenizer(model_name)
    input_ids = tok(prompt, return_tensors="pt").input_ids

    # 1. Vanilla HF Baseline (Unpatched)
    print("\n[1/2] Running Vanilla HF Attention (Baseline)...")
    remove_patch()
    reset_caches(model)
    vanilla_ids, vanilla_latencies = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
    vanilla_text = tok.decode(vanilla_ids[0], skip_special_tokens=True)
    avg_vanilla_lat = (sum(vanilla_latencies) / len(vanilla_latencies)) * 1000.0

    # 2. TriTierCache (Patched)
    print("\n[2/2] Running TriTierCache Patched Attention (AVX2 Fused)...")
    apply_patch()
    reset_caches(model)
    tritier_ids, tritier_latencies = generate_step_by_step(model, input_ids, max_new_tokens=gen_tokens)
    tritier_text = tok.decode(tritier_ids[0], skip_special_tokens=True)
    avg_tritier_lat = (sum(tritier_latencies) / len(tritier_latencies)) * 1000.0

    # 3. Verification & Comparison
    prompt_len = input_ids.shape[1]
    gen_vanilla = vanilla_ids[0, prompt_len:].tolist()
    gen_tritier = tritier_ids[0, prompt_len:].tolist()

    matching_tokens = sum(1 for v, t in zip(gen_vanilla, gen_tritier) if v == t)
    match_pct = (matching_tokens / gen_tokens) * 100.0

    divergence_step = None
    for idx, (v, t) in enumerate(zip(gen_vanilla, gen_tritier)):
        if v != t:
            divergence_step = idx + 1
            break

    print("\n" + "=" * 70)
    print(" GENERATION RESULTS & COMPARISON")
    print("=" * 70)
    print(f"Prompt: '{prompt}' (len={prompt_len})")
    print("-" * 70)
    print(f"Vanilla HF Output  : {vanilla_text}")
    print(f"TriTierCache Output: {tritier_text}")
    print("-" * 70)
    print(f"Token Match Count  : {matching_tokens}/{gen_tokens} ({match_pct:.1f}%)")
    print(f"First Divergence   : {'None (Exact Match)' if divergence_step is None else f'Token step {divergence_step}'}")
    print(f"Vanilla Decode Lat : {avg_vanilla_lat:.2f} ms/token")
    print(f"TriTier Decode Lat : {avg_tritier_lat:.2f} ms/token")
    print("=" * 70)

    # Save to CSV
    row = {
        "model": model_name,
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "generated_tokens": gen_tokens,
        "matching_tokens": matching_tokens,
        "match_percentage": match_pct,
        "first_divergence_step": -1 if divergence_step is None else divergence_step,
        "vanilla_avg_latency_ms": avg_vanilla_lat,
        "tritier_avg_latency_ms": avg_tritier_lat,
    }
    save_results_to_csv("benchmarks/results/smoke_test_results.csv", [row])


if __name__ == "__main__":
    main()
