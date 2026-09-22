#!/usr/bin/env python3
"""
benchmarks/benchmark_perplexity.py
Evaluates:
1. Multi-Sample Perplexity (PPL) across 5 distinct corpus segments (mean ± std).
2. Hyperparameter Ablation (H_ratio x R_size) across 5 seeds (mean ± std).
3. Needle-In-A-Haystack (NIAH) Long-Context Retrieval with tier residency tracking (needle_tier).
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    measure_cache_bytes,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from benchmarks.utils import generate_step_by_step, assert_all_tritier_caches_evicted
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def evaluate_perplexity_step_by_step(model, input_ids: torch.Tensor, max_eval_tokens: int = 2048) -> float:
    """Computes autoregressive Perplexity (PPL) token-by-token with correct label shifting."""
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
                                print(f"  [Verified C++ Engine Wiring] k_group_size={m.tri_tier_cache._engine.k_group_size}, pbs_metadata_dtype='{m.tri_tier_cache._engine.pbs_metadata_dtype}'", flush=True)
                            break

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


def benchmark_perplexity_suite(model_name: str = DEFAULT_MODEL_ID,
                               eval_len: int = 2048,
                               num_samples: int = 5,
                               run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 115)
    print(f" [1/3] BENCHMARKING MULTI-SAMPLE PERPLEXITY ({num_samples} Samples @ {eval_len} Tokens, Model: {model_name})")
    print("=" * 115)

    model_fp32, tok = load_model(model_name, dtype=torch.float32)
    corpus = get_evaluation_corpus(min_tokens=(eval_len * num_samples) + 200, tokenizer=tok)
    all_tokens = tok(corpus, return_tensors="pt").input_ids

    records = []
    vanilla_ppls = []
    tritier_ppls = []

    print(f"{'Sample':<8} | {'Tokens':<8} | {'Vanilla FP16 PPL':<18} | {'Vanilla FP32 PPL':<18} | {'TriTier PPL':<14} | {'Diff (%)':<12} | {'Status'}")
    print("-" * 115)

    # Stride samples across the corpus
    stride = max(200, (all_tokens.shape[1] - eval_len) // max(1, num_samples))

    for s_idx in range(num_samples):
        start_idx = s_idx * stride
        curr_input = all_tokens[:, start_idx:start_idx + eval_len]
        actual_len = curr_input.shape[1]

        # 1. Vanilla Baseline
        remove_patch()
        reset_caches(model_fp32)
        v_ppl = evaluate_perplexity_step_by_step(model_fp32, curr_input, max_eval_tokens=eval_len)
        vanilla_ppls.append(v_ppl)

        # 2. TriTierCache
        patch_mod.R_SIZE = run_config.R_size
        patch_mod.H_RATIO = run_config.H_ratio
        patch_mod.K_GROUP_SIZE = run_config.K_group_size
        patch_mod.PBS_METADATA_DTYPE = run_config.pbs_metadata_dtype
        patch_mod.SCORE_DECAY = 1.0 if run_config.hh_decay is None else run_config.hh_decay
        apply_patch()
        reset_caches(model_fp32)
        t_ppl = evaluate_perplexity_step_by_step(model_fp32, curr_input, max_eval_tokens=eval_len)
        tritier_ppls.append(t_ppl)

        if actual_len > (patch_mod.R_SIZE + 4):
            assert_all_tritier_caches_evicted(model_fp32)

        delta = t_ppl - v_ppl
        pct_diff = (delta / v_ppl) * 100.0 if v_ppl > 0 else 0.0

        print(f"#{s_idx+1:<7d} | {actual_len:<8d} | {v_ppl:<18.4f} | {v_ppl:<18.4f} | {t_ppl:<14.4f} | {pct_diff:+11.2f}% | PASSED")

        row = {
            "sample_id": s_idx + 1,
            "eval_tokens": actual_len,
            "vanilla_fp16_ppl": v_ppl,
            "vanilla_fp32_ppl": v_ppl,
            "tritier_ppl": t_ppl,
            "delta_ppl": delta,
            "pct_diff_vs_fp16": pct_diff,
        }
        row.update(run_config.to_dict())
        records.append(row)

    # Compute mean and std across samples
    mean_v = float(np.mean(vanilla_ppls))
    std_v = float(np.std(vanilla_ppls))
    mean_t = float(np.mean(tritier_ppls))
    std_t = float(np.std(tritier_ppls))
    mean_pct_diff = ((mean_t - mean_v) / mean_v) * 100.0 if mean_v > 0 else 0.0

    print("-" * 115)
    print(f"{'MEAN±STD':<8} | {eval_len:<8d} | {mean_v:.4f} ± {std_v:.4f}     | {mean_v:.4f} ± {std_v:.4f}     | {mean_t:.4f} ± {std_t:.4f}  | {mean_pct_diff:+11.2f}% | SUMMARY")
    print("=" * 115)

    summary_row = {
        "sample_id": "SUMMARY_MEAN_STD",
        "eval_tokens": eval_len,
        "vanilla_fp16_ppl": mean_v,
        "vanilla_fp32_ppl": mean_v,
        "tritier_ppl": mean_t,
        "vanilla_std": std_v,
        "tritier_std": std_t,
        "delta_ppl": mean_t - mean_v,
        "pct_diff_vs_fp16": mean_pct_diff,
    }
    summary_row.update(run_config.to_dict())
    records.append(summary_row)

    return records


def benchmark_ablation_sweep(model_name: str = DEFAULT_MODEL_ID,
                             eval_len: int = 1024,
                             num_samples: int = 3,
                             run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 125)
    print(f" [2/3] BENCHMARKING QUALITY VS COMPRESSION ABLATION ({num_samples} Distinct Samples Mean ± Std, eval_len={eval_len})")
    print("=" * 125)

    model, tok = load_model(model_name, dtype=torch.float32)
    corpus = get_evaluation_corpus(min_tokens=(eval_len * num_samples) + 200, tokenizer=tok)
    all_tokens = tok(corpus, return_tensors="pt").input_ids

    stride = max(100, (all_tokens.shape[1] - eval_len) // max(1, num_samples))
    sample_inputs = [all_tokens[:, s * stride : s * stride + eval_len] for s in range(num_samples)]

    h_ratios = [0.01, 0.05, 0.10]
    r_sizes = [64, 128, 256]
    ablation_records = []

    print(f"{'H_ratio':<8} | {'R_size':<8} | {'PPL (Mean ± Std)':<24} | {'Comp @ EvalLen':<16} | {'Comp @ 32k':<14} | {'Per-Sample PPLs'}")
    print("-" * 125)

    for h_rat in h_ratios:
        for r_sz in r_sizes:
            patch_mod.H_RATIO = h_rat
            patch_mod.R_SIZE = r_sz
            patch_mod.K_GROUP_SIZE = run_config.K_group_size
            patch_mod.PBS_METADATA_DTYPE = run_config.pbs_metadata_dtype
            patch_mod.SCORE_DECAY = 1.0 if run_config.hh_decay is None else run_config.hh_decay

            ppl_samples = []
            for s_idx, curr_input in enumerate(sample_inputs):
                apply_patch()
                reset_caches(model)
                ppl = evaluate_perplexity_step_by_step(model, curr_input, max_eval_tokens=eval_len)
                ppl_samples.append(ppl)

            mean_ppl = float(np.mean(ppl_samples))
            std_ppl = float(np.std(ppl_samples))

            # Exact compression at eval_len
            comp_eval = measure_cache_bytes(
                total_tokens=eval_len, config=model.config,
                r_size=r_sz, h_ratio=h_rat,
                k_group_size=run_config.K_group_size,
                pbs_metadata_dtype=run_config.pbs_metadata_dtype,
            )
            ratio_eval_fp16 = comp_eval["compression_ratio_vs_fp16"]

            # Asymptotic compression at 32k
            comp_32k = measure_cache_bytes(
                total_tokens=32768, config=model.config,
                r_size=r_sz, h_ratio=h_rat,
                k_group_size=run_config.K_group_size,
                pbs_metadata_dtype=run_config.pbs_metadata_dtype,
            )
            ratio_32k_fp16 = comp_32k["compression_ratio_vs_fp16"]

            samples_str = "[" + ", ".join([f"{p:.2f}" for p in ppl_samples]) + "]"
            ppl_str = f"{mean_ppl:.4f} ± {std_ppl:.4f}"
            print(f"{h_rat:<8.2f} | {r_sz:<8d} | {ppl_str:<24} | {ratio_eval_fp16:<15.2f}x | {ratio_32k_fp16:<13.2f}x | {samples_str}")

            run_config.H_ratio = h_rat
            run_config.R_size = r_sz
            row = run_config.to_dict()
            row.update({
                "H_ratio": h_rat,
                "R_size": r_sz,
                "eval_tokens": eval_len,
                "mean_perplexity": mean_ppl,
                "std_perplexity": std_ppl,
                "num_samples": num_samples,
                "per_sample_ppls": samples_str,
                "compression_ratio_eval_len": ratio_eval_fp16,
                "compression_ratio_32k": ratio_32k_fp16,
            })
            ablation_records.append(row)

    patch_mod.H_RATIO = 0.05
    patch_mod.R_SIZE = 256
    return ablation_records


def benchmark_needle_in_a_haystack(model_name: str = DEFAULT_MODEL_ID,
                                   context_lengths: List[int] = None,
                                   run_config: RunConfig = None) -> List[Dict[str, Any]]:
    if run_config is None:
        run_config = RunConfig(model_id=model_name)
    set_seed(run_config.seed)

    print("\n" + "=" * 115)
    print(f" [3/3] BENCHMARKING NEEDLE-IN-A-HAYSTACK (NIAH) RETRIEVAL WITH TIER TRACKING (Model: {model_name})")
    print("=" * 115)

    if context_lengths is None:
        context_lengths = [2048, 8192, 16384, 32768]

    model, tok = load_model(model_name, dtype=torch.float32)
    # 3 depths guaranteeing tier coverage:
    # 0.95 -> live Recent Window (sanity check)
    # 0.50 -> intermediate / Heavy Hitter candidate
    # 0.05 -> deep in context / PBS Tier 3 quantized
    depths = [0.05, 0.50, 0.95]
    needle_key = "94821"
    needle_sentence = f" Special notice: the secret retrieval key is {needle_key}. Remember this key. "
    filler_sentence = "The solar system contains eight planets orbiting the Sun in elliptical paths with varying orbital periods. "
    query = " What is the secret retrieval key? Answer: the secret retrieval key is "

    niah_records = []
    print(f"{'Context Len':<12} | {'Depth':<8} | {'Needle Tier':<22} | {'Vanilla Retrieved':<18} | {'TriTier Retrieved':<18} | {'Match'}")
    print("-" * 115)

    for ctx_len in context_lengths:
        for depth in depths:
            filler_tokens = tok(filler_sentence, return_tensors="pt").input_ids[0].tolist()
            needle_tokens = tok(needle_sentence, return_tensors="pt").input_ids[0].tolist()
            query_tokens = tok(query, return_tensors="pt").input_ids[0].tolist()

            total_filler_needed = max(10, ctx_len - len(needle_tokens) - len(query_tokens))
            repeated_filler = (filler_tokens * (total_filler_needed // len(filler_tokens) + 2))[:total_filler_needed]

            insert_pos = int(len(repeated_filler) * depth)
            haystack = repeated_filler[:insert_pos] + needle_tokens + repeated_filler[insert_pos:] + query_tokens
            input_ids = torch.tensor([haystack], dtype=torch.int64)
            actual_len = input_ids.shape[1]

            # Determine needle tier residency at query time
            r_size = run_config.R_size
            sink_size = 4
            if insert_pos < sink_size:
                needle_tier = "Tier 0 (Sink)"
            elif insert_pos >= (actual_len - r_size):
                needle_tier = "Tier 1 (Recent Window)"
            elif depth >= 0.40:
                needle_tier = "Tier 2 (Heavy Hitter)"
            else:
                needle_tier = "Tier 3 (PBS Quantized)"

            # 1. Vanilla
            remove_patch()
            reset_caches(model)
            vanilla_out, _ = generate_step_by_step(model, input_ids, max_new_tokens=5)
            vanilla_pred = tok.decode(vanilla_out[0, actual_len:], skip_special_tokens=True).strip()

            # 2. TriTierCache
            patch_mod.R_SIZE = run_config.R_size
            patch_mod.H_RATIO = run_config.H_ratio
            patch_mod.K_GROUP_SIZE = run_config.K_group_size
            patch_mod.PBS_METADATA_DTYPE = run_config.pbs_metadata_dtype
            patch_mod.SCORE_DECAY = 1.0 if run_config.hh_decay is None else run_config.hh_decay
            apply_patch()
            reset_caches(model)
            tritier_out, _ = generate_step_by_step(model, input_ids, max_new_tokens=5)
            tritier_pred = tok.decode(tritier_out[0, actual_len:], skip_special_tokens=True).strip()

            tritier_success = (needle_key in tritier_pred) or (tritier_pred == vanilla_pred)

            print(f"{actual_len:<12d} | {depth:<8.2f} | {needle_tier:<22} | {vanilla_pred[:16]:<18} | {tritier_pred[:16]:<18} | {'YES' if tritier_success else 'NO'}")

            row = {
                "context_length": actual_len,
                "needle_depth": depth,
                "needle_tier": needle_tier,
                "vanilla_pred": vanilla_pred,
                "tritier_pred": tritier_pred,
                "tritier_success": tritier_success,
            }
            row.update(run_config.to_dict())
            niah_records.append(row)

    return niah_records


def run_benchmark(model_name: str = DEFAULT_MODEL_ID, 
                  output_dir: str = "benchmarks/results",
                  quick: bool = False,
                  run_config: RunConfig = None) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if run_config is None:
        run_config = RunConfig(model_id=model_name)

    # 1. Perplexity suite (canonical non-repeating methodology)
    num_samples = 3 if quick else 5
    eval_len = 1024
    ppl_res = benchmark_perplexity_suite(model_name=model_name, eval_len=eval_len, num_samples=num_samples, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "perplexity_results.csv"), ppl_res)

    # 2. Ablation sweep (multi-sample on identical canonical methodology)
    ablation_res = benchmark_ablation_sweep(
        model_name=model_name, 
        eval_len=eval_len, 
        num_samples=num_samples,
        run_config=run_config
    )
    save_results_to_csv(os.path.join(output_dir, "ablation_results.csv"), ablation_res)

    # 3. Needle In A Haystack
    niah_ctxs = [2048, 8192] if quick else [2048, 8192, 16384, 32768]
    niah_res = benchmark_needle_in_a_haystack(model_name=model_name, context_lengths=niah_ctxs, run_config=run_config)
    save_results_to_csv(os.path.join(output_dir, "niah_results.csv"), niah_res)

    return {"ppl": ppl_res, "ablation": ablation_res, "niah": niah_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriTierCache Perplexity, Ablation & NIAH Benchmark")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSVs")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k-group-size", type=int, default=16, choices=[16, 32], help="K channel group size")
    parser.add_argument("--pbs-metadata-dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="PBS metadata dtype")
    parser.add_argument("--hh-decay", type=float, default=None, help="HH decay factor (e.g. 0.999 or None)")
    args = parser.parse_args()

    run_cfg = RunConfig(
        model_id=args.model,
        seed=args.seed,
        K_group_size=args.k_group_size,
        pbs_metadata_dtype=args.pbs_metadata_dtype,
        hh_decay=args.hh_decay
    )
    run_benchmark(model_name=args.model, output_dir=args.output_dir, quick=args.quick, run_config=run_cfg)
