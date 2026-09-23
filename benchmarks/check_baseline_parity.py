#!/usr/bin/env python3
import math
import torch
import torch.nn.functional as F
from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    set_seed,
    save_results_to_csv,
    RunConfig,
)
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def evaluate_ppl(model, input_ids: torch.Tensor) -> float:
    seq_len = input_ids.shape[1]
    total_nll = 0.0
    count = 0
    past_kv = None
    with torch.no_grad():
        for pos in range(seq_len - 1):
            tok_in = input_ids[:, pos:pos+1]
            target_tok = input_ids[:, pos+1]
            if not hasattr(model.model.layers[0].self_attn, "tri_tier_cache") or model.model.layers[0].self_attn.tri_tier_cache is None:
                out = model(tok_in, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            else:
                out = model(tok_in, position_ids=torch.tensor([[pos]], dtype=torch.int64))
                past_kv = None
            logits = out.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target_tok[0]].item()
            if not math.isnan(nll) and not math.isinf(nll):
                total_nll += nll
                count += 1
    return math.exp(total_nll / count) if count > 0 else float("nan")


def main():
    model_name = "meta-llama/Llama-3.2-1B"
    eval_len = 1024
    num_samples = 3
    seed = 42
    set_seed(seed)

    print("Loading model...")
    model, tok = load_model(model_name, dtype=torch.float32)

    # Replicate exact original corpus extraction from benchmark_perplexity.py
    corpus = get_evaluation_corpus(min_tokens=(eval_len * num_samples) + 200, tokenizer=tok)
    all_tokens = tok(corpus, return_tensors="pt").input_ids

    stride = max(200, (all_tokens.shape[1] - eval_len) // max(1, num_samples))
    sample_inputs = [all_tokens[:, s * stride : s * stride + eval_len] for s in range(num_samples)]

    print(f"Total tokens in corpus: {all_tokens.shape[1]}, Stride: {stride}")

    # 1. Vanilla FP32 baseline
    print("Evaluating Vanilla FP32...")
    remove_patch()
    reset_caches(model)
    v_ppls = [evaluate_ppl(model, s) for s in sample_inputs]
    print(f"Vanilla PPLs: {v_ppls}")

    # 2. Config A (grp-16, fp32 metadata)
    print("Evaluating Config A (grp-16, fp32)...")
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = 16
    patch_mod.PBS_METADATA_DTYPE = "fp32"
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()
    reset_caches(model)
    a_ppls = []
    for s in sample_inputs:
        reset_caches(model)
        a_ppls.append(evaluate_ppl(model, s))
    print(f"Config A PPLs: {a_ppls}")

    # 3. Config B (grp-32, fp16 metadata)
    print("Evaluating Config B (grp-32, fp16)...")
    patch_mod.K_GROUP_SIZE = 32
    patch_mod.PBS_METADATA_DTYPE = "fp16"
    apply_patch()
    reset_caches(model)
    b_ppls = []
    for s in sample_inputs:
        reset_caches(model)
        b_ppls.append(evaluate_ppl(model, s))
    print(f"Config B PPLs: {b_ppls}")

    rows = []
    for i in range(num_samples):
        rows.append({
            "sample_id": i + 1,
            "eval_tokens": eval_len,
            "vanilla_fp32_ppl": v_ppls[i],
            "config_a_ppl": a_ppls[i],
            "config_a_delta": a_ppls[i] - v_ppls[i],
            "config_a_pct": ((a_ppls[i] - v_ppls[i]) / v_ppls[i]) * 100.0,
            "config_b_ppl": b_ppls[i],
            "config_b_delta": b_ppls[i] - v_ppls[i],
            "config_b_pct": ((b_ppls[i] - v_ppls[i]) / v_ppls[i]) * 100.0,
            "delta_b_vs_a": b_ppls[i] - a_ppls[i],
        })
    rows.append({
        "sample_id": "MEAN",
        "eval_tokens": eval_len,
        "vanilla_fp32_ppl": float(torch.tensor(v_ppls).mean()),
        "config_a_ppl": float(torch.tensor(a_ppls).mean()),
        "config_a_delta": float(torch.tensor(a_ppls).mean() - torch.tensor(v_ppls).mean()),
        "config_a_pct": float(((torch.tensor(a_ppls).mean() - torch.tensor(v_ppls).mean()) / torch.tensor(v_ppls).mean()) * 100.0),
        "config_b_ppl": float(torch.tensor(b_ppls).mean()),
        "config_b_delta": float(torch.tensor(b_ppls).mean() - torch.tensor(v_ppls).mean()),
        "config_b_pct": float(((torch.tensor(b_ppls).mean() - torch.tensor(v_ppls).mean()) / torch.tensor(v_ppls).mean()) * 100.0),
        "delta_b_vs_a": float(torch.tensor(b_ppls).mean() - torch.tensor(a_ppls).mean()),
    })
    save_results_to_csv("benchmarks/results/matched_baseline_parity.csv", rows)


if __name__ == "__main__":
    main()
