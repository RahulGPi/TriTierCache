#!/usr/bin/env python3
"""
benchmarks/run_nonrepeating_graduated_niah.py
Graduated Needle-In-A-Haystack sweep using diverse, non-repeating corpus (corpus.py).
Evaluates both HuggingFaceTB/SmolLM-135M and meta-llama/Llama-3.2-1B:
- SmolLM-135M: 1.0x to 1.5x of 2048 (2048, 2252, 2457, 2560, 2662, 2867, 3072)
- Llama-3.2-1B: 1.0x to 2.0x of 8192 (8192, 9011, 9830, 10240, 10650, 11469, 12288, 14336, 16384)
- Depths: 0.20, 0.70, 0.95
- RoPE modes: 'a' (native RoPE) vs 'b' (window-clamped)
- H_ratios: 0.05, 0.15
- Records: dominant tier, mean K cosine, mean V cosine, storage_intact, decoded output, retrieval success.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import time
import csv
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from benchmarks.common import load_model, set_seed
from benchmarks.construct_nonrepeating_niah import construct_nonrepeating_niah_prompt
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod
import tri_tier._C as _C


def evaluate_niah_row(
    model,
    tok,
    total_len: int,
    depth: float,
    rope_mode: str,
    h_ratio: float = 0.05,
    needle_key: str = "94821",
    gen_steps: int = 8,
    is_vanilla: bool = False,
):
    if is_vanilla:
        remove_patch()
        reset_caches(model)
    else:
        patch_mod.R_SIZE = 256
        patch_mod.H_RATIO = h_ratio
        patch_mod.K_GROUP_SIZE = 16
        patch_mod.PBS_METADATA_DTYPE = "fp16"
        patch_mod.SCORE_DECAY = 0.999
        apply_patch()
        reset_caches(model)

    full_prompt, needle_start, needle_end, key_pos = construct_nonrepeating_niah_prompt(
        tok, total_len, depth, needle_key
    )
    actual_len = full_prompt.shape[1]

    if not is_vanilla:
        for layer in model.model.layers:
            layer.self_attn.capture_needle_pos = needle_start

    t0 = time.time()
    with torch.no_grad():
        if is_vanilla:
            out = model(full_prompt, use_cache=True)
            past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens = [curr_id.item()]
            for step in range(gen_steps - 1):
                out = model(curr_id, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
                curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(curr_id.item())
        else:
            base_out = model.model(full_prompt)
            last_h = base_out.last_hidden_state[:, -1:, :]
            curr_id = model.lm_head(last_h).argmax(dim=-1)
            gen_tokens = [curr_id.item()]

            for step in range(gen_steps - 1):
                dec_pos = actual_len + step
                if rope_mode == "a":
                    pos_tensor = torch.tensor([[dec_pos]], dtype=torch.int64)
                else:
                    rel_pos = min(dec_pos, patch_mod.R_SIZE + gen_steps + step)
                    pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)

                out = model(curr_id, position_ids=pos_tensor)
                curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(curr_id.item())

    elapsed = time.time() - t0
    pred_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    success = (needle_key in pred_text)

    if is_vanilla:
        return {
            "context_length": actual_len,
            "needle_start_pos": needle_start,
            "depth": depth,
            "rope_mode": "vanilla",
            "h_ratio": 0.0,
            "dominant_tier": "vanilla_fp32",
            "mean_k_cosine_sim": 1.0,
            "mean_v_cosine_sim": 1.0,
            "storage_intact": True,
            "generated_text": pred_text,
            "success": success,
            "elapsed": elapsed,
        }

    # Reconstruct needle K/V from TriTierCache
    k_cos_sims = []
    v_cos_sims = []
    tier_counts = {"sink": 0, "rw": 0, "hh": 0, "pbs": 0, "missing": 0}
    target_tok_id = needle_start

    for layer_idx, layer in enumerate(model.model.layers):
        eng = layer.self_attn.tri_tier_cache._engine
        orig_k = layer.self_attn.captured_needle_k
        orig_v = layer.self_attn.captured_needle_v

        # Check Sink
        if target_tok_id < eng.s_count:
            tier_counts["sink"] += 1
            s_k = torch.from_numpy(eng.get_S_K())
            s_v = torch.from_numpy(eng.get_S_V())
            recon_k = s_k[target_tok_id]
            recon_v = s_v[target_tok_id]
        # Check RW (Recent Window) using the invariant ring-buffer formula
        elif target_tok_id >= (eng.total_processed_tokens - eng.rw_count):
            tier_counts["rw"] += 1
            rw_idx = (eng.rw_head_index + target_tok_id - eng.total_processed_tokens) % eng.rw_size
            rw_k = torch.from_numpy(eng.get_RW_K())
            rw_v = torch.from_numpy(eng.get_RW_V())
            recon_k = rw_k[rw_idx]
            recon_v = rw_v[rw_idx]
        else:
            # Check PBS
            pbs_ids = eng.get_PBS_token_ids()
            total_pbs_tokens = eng.pbs_blocks_used * eng.chunk_size
            pbs_match = np.where(pbs_ids[:total_pbs_tokens] == target_tok_id)[0]
            if len(pbs_match) > 0:
                tier_counts["pbs"] += 1
                match_idx = pbs_match[0]
                block_idx = match_idx // eng.chunk_size
                tok_in_block = match_idx % eng.chunk_size

                k_packed = eng.get_PBS_K_Packed()[block_idx:block_idx+1]
                k_scales = eng.get_PBS_K_Scales()[block_idx:block_idx+1]
                k_zeroes = eng.get_PBS_K_Zeroes()[block_idx:block_idx+1]
                K_block_out = torch.empty((eng.chunk_size, eng.num_kv_heads, eng.head_dim), dtype=torch.float32)
                _C.dequantize_k(torch.from_numpy(k_packed), torch.from_numpy(k_scales), torch.from_numpy(k_zeroes),
                                K_block_out, 1, eng.num_kv_heads, eng.head_dim, eng.k_group_size, eng.pbs_metadata_dtype)
                recon_k = K_block_out[tok_in_block]

                v_start = block_idx * eng.chunk_size
                v_packed = eng.get_PBS_V_Packed()[v_start:v_start+eng.chunk_size]
                v_scales = eng.get_PBS_V_Scales()[v_start:v_start+eng.chunk_size]
                v_zeroes = eng.get_PBS_V_Zeroes()[v_start:v_start+eng.chunk_size]
                V_block_out = torch.empty((eng.chunk_size, eng.num_kv_heads, eng.head_dim), dtype=torch.float32)
                _C.dequantize_v(torch.from_numpy(v_packed), torch.from_numpy(v_scales), torch.from_numpy(v_zeroes),
                                V_block_out, eng.chunk_size, eng.num_kv_heads, eng.head_dim, eng.pbs_metadata_dtype)
                recon_v = V_block_out[tok_in_block]
            else:
                tier_counts["missing"] += 1
                recon_k, recon_v = None, None

        if recon_k is not None:
            k_sim = F.cosine_similarity(orig_k.float().view(-1), recon_k.float().view(-1), dim=0).item()
            v_sim = F.cosine_similarity(orig_v.float().view(-1), recon_v.float().view(-1), dim=0).item()
            k_cos_sims.append(k_sim)
            v_cos_sims.append(v_sim)

    dominant_tier = max(tier_counts, key=tier_counts.get)
    mean_k = float(np.mean(k_cos_sims)) if k_cos_sims else 0.0
    mean_v = float(np.mean(v_cos_sims)) if v_cos_sims else 0.0
    intact = (mean_k > 0.95 and mean_v > 0.90)

    return {
        "context_length": actual_len,
        "needle_start_pos": needle_start,
        "depth": depth,
        "rope_mode": rope_mode,
        "h_ratio": h_ratio,
        "dominant_tier": dominant_tier,
        "mean_k_cosine_sim": mean_k,
        "mean_v_cosine_sim": mean_v,
        "storage_intact": intact,
        "generated_text": pred_text,
        "success": success,
        "elapsed": elapsed,
    }


def run_sweep(model_id: str, output_csv: str, include_vanilla: bool = False):
    set_seed(42)
    print(f"Loading {model_id}...")
    model, tok = load_model(model_id, dtype=torch.float32)

    if "SmolLM" in model_id:
        base_len = 2048
        lengths = [
            (1.0, 2048),
            (1.1, 2252),
            (1.2, 2457),
            (1.25, 2560),
            (1.3, 2662),
            (1.4, 2867),
            (1.5, 3072),
        ]
    else:
        base_len = 8192
        lengths = [
            (1.0, 8192),
            (1.1, 9011),
            (1.2, 9830),
            (1.25, 10240),
            (1.3, 10650),
            (1.4, 11469),
            (1.5, 12288),
            (1.75, 14336),
            (2.0, 16384),
        ]

    depths = [0.20, 0.70, 0.95]
    rope_modes = ["a", "b"]
    h_ratios = [0.05, 0.15]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "model", "length_ratio", "context_length", "depth", "rope_mode", "h_ratio",
        "k_group_size", "pbs_metadata_dtype", "r_size", "seed",
        "needle_tier", "mean_k_cosine_sim", "mean_v_cosine_sim", "storage_intact",
        "generated_text", "success", "elapsed_sec"
    ]

    results = []
    print("=" * 145)
    print(f"{'Ratio':<7} | {'Context':<8} | {'Depth':<6} | {'RoPE':<8} | {'H_ratio':<7} | {'Tier':<12} | {'K Cos':<12} | {'V Cos':<12} | {'Intact':<7} | {'Success':<8} | {'Elapsed':<7}")
    print("-" * 145)

    # 1. Optionally run Vanilla FP32 baseline on key points
    if include_vanilla:
        print("\n--- Running Vanilla FP32 Baseline ---")
        for ratio, length in lengths:
            for depth in depths:
                res = evaluate_niah_row(model, tok, length, depth, rope_mode="vanilla", is_vanilla=True)
                row = {
                    "model": model_id,
                    "length_ratio": ratio,
                    "context_length": res["context_length"],
                    "depth": depth,
                    "rope_mode": "vanilla",
                    "h_ratio": 0.0,
                    "k_group_size": 16,
                    "pbs_metadata_dtype": "none",
                    "r_size": 0,
                    "seed": 42,
                    "needle_tier": "vanilla_fp32",
                    "mean_k_cosine_sim": 1.0,
                    "mean_v_cosine_sim": 1.0,
                    "storage_intact": True,
                    "generated_text": res["generated_text"],
                    "success": res["success"],
                    "elapsed_sec": round(res["elapsed"], 2),
                }
                results.append(row)
                with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
                succ_str = "YES" if res["success"] else "NO"
                print(f"{ratio:<7.2f} | {res['context_length']:<8d} | {depth:<6.2f} | {'vanilla':<8} | {0.0:<7.2f} | {'vanilla_fp32':<12} | {1.0:<12.8f} | {1.0:<12.8f} | {'YES':<7} | {succ_str:<8} | {res['elapsed']:<7.2f}s", flush=True)

    # 2. Run TriTierCache Grid
    print("\n--- Running TriTierCache Sweep ---")
    for ratio, length in lengths:
        for h_ratio in h_ratios:
            for mode in rope_modes:
                for depth in depths:
                    res = evaluate_niah_row(model, tok, length, depth, mode, h_ratio, is_vanilla=False)
                    row = {
                        "model": model_id,
                        "length_ratio": ratio,
                        "context_length": res["context_length"],
                        "depth": depth,
                        "rope_mode": mode,
                        "h_ratio": h_ratio,
                        "k_group_size": 16,
                        "pbs_metadata_dtype": "fp16",
                        "r_size": 256,
                        "seed": 42,
                        "needle_tier": res["dominant_tier"],
                        "mean_k_cosine_sim": res["mean_k_cosine_sim"],
                        "mean_v_cosine_sim": res["mean_v_cosine_sim"],
                        "storage_intact": res["storage_intact"],
                        "generated_text": res["generated_text"],
                        "success": res["success"],
                        "elapsed_sec": round(res["elapsed"], 2),
                    }
                    results.append(row)
                    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                    intact_str = "YES" if res["storage_intact"] else "NO"
                    succ_str = "YES" if res["success"] else "NO"
                    print(f"{ratio:<7.2f} | {res['context_length']:<8d} | {depth:<6.2f} | {mode:<8} | {h_ratio:<7.2f} | {res['dominant_tier']:<12} | {res['mean_k_cosine_sim']:<12.8f} | {res['mean_v_cosine_sim']:<12.8f} | {intact_str:<7} | {succ_str:<8} | {res['elapsed']:<7.2f}s", flush=True)

    print("=" * 145)
    print(f"Sweep complete for {model_id}. Saved {len(results)} rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--output", type=str, default="benchmarks/results/nonrepeating_niah_smol.csv")
    parser.add_argument("--include-vanilla", action="store_true")
    args = parser.parse_args()

    run_sweep(args.model, args.output, include_vanilla=args.include_vanilla)
