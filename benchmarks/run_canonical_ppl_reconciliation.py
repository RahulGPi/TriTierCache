#!/usr/bin/env python3
"""
benchmarks/run_canonical_ppl_reconciliation.py
Phase 2 Step 7: Canonical PPL reconciliation (3-sample, 1024 eval_len).
Compares:
  - Config A: FP32 metadata / group-16
  - Config B: FP16 metadata / group-32
Alongside Vanilla Baseline, measuring compression vs quality.
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    measure_cache_bytes,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def evaluate_ppl(model, input_ids: torch.Tensor, max_eval_tokens: int = 1024) -> float:
    seq_len = min(input_ids.shape[1], max_eval_tokens)
    total_nll = 0.0
    count = 0
    past_kv = None

    from transformers.models.llama.modeling_llama import LlamaAttention
    from src.tri_tier.integration.patch_llama import patched_forward
    is_patched = (LlamaAttention.forward == patched_forward)

    with torch.no_grad():
        for pos in range(seq_len - 1):
            tok_in = input_ids[:, pos:pos+1]
            target_tok = input_ids[:, pos+1]
            if not is_patched:
                out = model(tok_in, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            else:
                out = model(tok_in, position_ids=torch.tensor([[pos]], dtype=torch.int64))
                past_kv = None
                if pos == 0:
                    for m in model.modules():
                        if hasattr(m, "tri_tier_cache") and m.tri_tier_cache is not None:
                            assert m.tri_tier_cache.k_group_size == patch_mod.K_GROUP_SIZE
                            assert m.tri_tier_cache.pbs_metadata_dtype == patch_mod.PBS_METADATA_DTYPE
                            if m.tri_tier_cache._engine is not None:
                                assert m.tri_tier_cache._engine.k_group_size == patch_mod.K_GROUP_SIZE
                                assert m.tri_tier_cache._engine.pbs_metadata_dtype == patch_mod.PBS_METADATA_DTYPE
                            break

            logits = out.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target_tok[0]].item()

            if not math.isnan(nll) and not math.isinf(nll):
                total_nll += nll
                count += 1

    if count == 0:
        return float("nan")
    return math.exp(total_nll / count)


def run_reconciliation(model_name: str = "meta-llama/Llama-3.2-1B",
                       eval_len: int = 1024,
                       num_samples: int = 3,
                       seed: int = 42,
                       output_csv: str = "benchmarks/results/canonical_ppl_reconciliation.csv") -> List[Dict[str, Any]]:
    set_seed(seed)
    print("=" * 115)
    print(f" [PHASE 2 STEP 7] CANONICAL PPL RECONCILIATION ({num_samples} Samples @ {eval_len} Tokens, Model: {model_name})")
    print("  Config A: FP32 Metadata / Group-16")
    print("  Config B: FP16 Metadata / Group-32")
    print("=" * 115)

    model_fp32, tok = load_model(model_name, dtype=torch.float32)
    corpus = get_evaluation_corpus(min_tokens=(eval_len * num_samples) + 500, tokenizer=tok)
    all_tokens = tok(corpus, return_tensors="pt").input_ids

    stride = max(200, (all_tokens.shape[1] - eval_len) // max(1, num_samples))
    sample_inputs = []
    for s_idx in range(num_samples):
        start_idx = s_idx * stride
        sample_inputs.append(all_tokens[:, start_idx:start_idx + eval_len])

    # 1. Vanilla FP32 baseline
    print("\n--- Running Vanilla FP32 Baseline ---")
    remove_patch()
    reset_caches(model_fp32)
    vanilla_ppls = []
    for s_idx, inp in enumerate(sample_inputs, start=1):
        ppl = evaluate_ppl(model_fp32, inp, max_eval_tokens=eval_len)
        vanilla_ppls.append(ppl)
        print(f"  Vanilla Sample {s_idx}: PPL = {ppl:.4f}")

    # 2. Config A: FP32 metadata / group-16
    print("\n--- Running Config A: FP32 Metadata / Group-16 ---")
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = 16
    patch_mod.PBS_METADATA_DTYPE = "fp32"
    patch_mod.SCORE_DECAY = 1.0
    apply_patch()
    reset_caches(model_fp32)

    config_a_ppls = []
    engine_bytes_a = None
    for s_idx, inp in enumerate(sample_inputs, start=1):
        reset_caches(model_fp32)
        ppl = evaluate_ppl(model_fp32, inp, max_eval_tokens=eval_len)
        config_a_ppls.append(ppl)
        if engine_bytes_a is None:
            for m in model_fp32.modules():
                if hasattr(m, "tri_tier_cache") and m.tri_tier_cache is not None and m.tri_tier_cache._engine is not None:
                    engine_bytes_a = m.tri_tier_cache._engine.get_buffer_bytes()
                    break
        print(f"  Config A Sample {s_idx}: PPL = {ppl:.4f}")

    # 3. Config B: FP16 metadata / group-32
    print("\n--- Running Config B: FP16 Metadata / Group-32 ---")
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = 32
    patch_mod.PBS_METADATA_DTYPE = "fp16"
    patch_mod.SCORE_DECAY = 1.0
    apply_patch()
    reset_caches(model_fp32)

    config_b_ppls = []
    engine_bytes_b = None
    for s_idx, inp in enumerate(sample_inputs, start=1):
        reset_caches(model_fp32)
        ppl = evaluate_ppl(model_fp32, inp, max_eval_tokens=eval_len)
        config_b_ppls.append(ppl)
        if engine_bytes_b is None:
            for m in model_fp32.modules():
                if hasattr(m, "tri_tier_cache") and m.tri_tier_cache is not None and m.tri_tier_cache._engine is not None:
                    engine_bytes_b = m.tri_tier_cache._engine.get_buffer_bytes()
                    break
        print(f"  Config B Sample {s_idx}: PPL = {ppl:.4f}")

    # Theoretical compression ratios
    comp_a_1024 = measure_cache_bytes(total_tokens=1024, config=model_fp32.config, r_size=256, h_ratio=0.05, k_group_size=16, pbs_metadata_dtype="fp32")
    comp_a_32k = measure_cache_bytes(total_tokens=32768, config=model_fp32.config, r_size=256, h_ratio=0.05, k_group_size=16, pbs_metadata_dtype="fp32")

    comp_b_1024 = measure_cache_bytes(total_tokens=1024, config=model_fp32.config, r_size=256, h_ratio=0.05, k_group_size=32, pbs_metadata_dtype="fp16")
    comp_b_32k = measure_cache_bytes(total_tokens=32768, config=model_fp32.config, r_size=256, h_ratio=0.05, k_group_size=32, pbs_metadata_dtype="fp16")

    # Build rows
    records = []
    for s_idx in range(num_samples):
        v = vanilla_ppls[s_idx]
        ca = config_a_ppls[s_idx]
        cb = config_b_ppls[s_idx]
        records.append({
            "sample_id": s_idx + 1,
            "eval_tokens": eval_len,
            "vanilla_ppl": v,
            "config_a_ppl_fp32_grp16": ca,
            "config_a_delta": ca - v,
            "config_a_pct_diff": ((ca - v) / v) * 100.0,
            "config_b_ppl_fp16_grp32": cb,
            "config_b_delta": cb - v,
            "config_b_pct_diff": ((cb - v) / v) * 100.0,
            "quality_delta_b_vs_a": cb - ca,
            "model_id": model_name,
            "engine_bytes_config_a": engine_bytes_a,
            "engine_bytes_config_b": engine_bytes_b,
            "comp_ratio_1024_a": comp_a_1024["compression_ratio_vs_fp16"],
            "comp_ratio_32k_a": comp_a_32k["compression_ratio_vs_fp16"],
            "comp_ratio_1024_b": comp_b_1024["compression_ratio_vs_fp16"],
            "comp_ratio_32k_b": comp_b_32k["compression_ratio_vs_fp16"],
        })

    # Summary row
    mean_v = float(np.mean(vanilla_ppls))
    std_v = float(np.std(vanilla_ppls))
    mean_ca = float(np.mean(config_a_ppls))
    std_ca = float(np.std(config_a_ppls))
    mean_cb = float(np.mean(config_b_ppls))
    std_cb = float(np.std(config_b_ppls))

    records.append({
        "sample_id": "SUMMARY_MEAN_STD",
        "eval_tokens": eval_len,
        "vanilla_ppl": mean_v,
        "vanilla_std": std_v,
        "config_a_ppl_fp32_grp16": mean_ca,
        "config_a_std": std_ca,
        "config_a_delta": mean_ca - mean_v,
        "config_a_pct_diff": ((mean_ca - mean_v) / mean_v) * 100.0,
        "config_b_ppl_fp16_grp32": mean_cb,
        "config_b_std": std_cb,
        "config_b_delta": mean_cb - mean_v,
        "config_b_pct_diff": ((mean_cb - mean_v) / mean_v) * 100.0,
        "quality_delta_b_vs_a": mean_cb - mean_ca,
        "model_id": model_name,
        "engine_bytes_config_a": engine_bytes_a,
        "engine_bytes_config_b": engine_bytes_b,
        "comp_ratio_1024_a": comp_a_1024["compression_ratio_vs_fp16"],
        "comp_ratio_32k_a": comp_a_32k["compression_ratio_vs_fp16"],
        "comp_ratio_1024_b": comp_b_1024["compression_ratio_vs_fp16"],
        "comp_ratio_32k_b": comp_b_32k["compression_ratio_vs_fp16"],
    })

    print("\n" + "=" * 125)
    print(f"{'Sample':<10} | {'Vanilla FP32':<14} | {'Config A (FP32/grp16)':<22} | {'Diff vs Van (%)':<16} | {'Config B (FP16/grp32)':<22} | {'Diff vs Van (%)':<16} | {'B vs A Delta'}")
    print("-" * 125)
    for r in records[:-1]:
        print(f"#{r['sample_id']:<9d} | {r['vanilla_ppl']:<14.4f} | {r['config_a_ppl_fp32_grp16']:<22.4f} | {r['config_a_pct_diff']:+15.2f}% | {r['config_b_ppl_fp16_grp32']:<22.4f} | {r['config_b_pct_diff']:+15.2f}% | {r['quality_delta_b_vs_a']:+.4f}")
    print("-" * 125)
    print(f"{'MEAN±STD':<10} | {mean_v:.4f}±{std_v:.4f}  | {mean_ca:.4f}±{std_ca:.4f}         | {((mean_ca - mean_v) / mean_v) * 100.0:+15.2f}% | {mean_cb:.4f}±{std_cb:.4f}         | {((mean_cb - mean_v) / mean_v) * 100.0:+15.2f}% | {mean_cb - mean_ca:+.4f}")
    print("=" * 125)

    print(f"\n[Memory & Compression Accounting]")
    print(f"  Config A Engine Buffer Memory: {engine_bytes_a:,} bytes ({engine_bytes_a / (1024*1024):.2f} MB)")
    print(f"  Config B Engine Buffer Memory: {engine_bytes_b:,} bytes ({engine_bytes_b / (1024*1024):.2f} MB)")
    mem_saved = (1.0 - engine_bytes_b / engine_bytes_a) * 100.0
    print(f"  Buffer Memory Reduction (B vs A): {mem_saved:.2f}%")
    print(f"  Config A Compression Ratio (1024 tok): {comp_a_1024['compression_ratio_vs_fp16']:.2f}x vs FP16 ({comp_a_1024['compression_ratio_vs_fp32']:.2f}x vs FP32)")
    print(f"  Config B Compression Ratio (1024 tok): {comp_b_1024['compression_ratio_vs_fp16']:.2f}x vs FP16 ({comp_b_1024['compression_ratio_vs_fp32']:.2f}x vs FP32)")
    print(f"  Config A Compression Ratio (32k tok) : {comp_a_32k['compression_ratio_vs_fp16']:.2f}x vs FP16 ({comp_a_32k['compression_ratio_vs_fp32']:.2f}x vs FP32)")
    print(f"  Config B Compression Ratio (32k tok) : {comp_b_32k['compression_ratio_vs_fp16']:.2f}x vs FP16 ({comp_b_32k['compression_ratio_vs_fp32']:.2f}x vs FP32)")

    save_results_to_csv(output_csv, records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical PPL Reconciliation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--eval-len", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default="benchmarks/results/canonical_ppl_reconciliation.csv")
    args = parser.parse_args()

    run_reconciliation(
        model_name=args.model,
        eval_len=args.eval_len,
        num_samples=args.num_samples,
        seed=args.seed,
        output_csv=args.output_csv
    )
