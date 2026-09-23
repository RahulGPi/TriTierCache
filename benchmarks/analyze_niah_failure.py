#!/usr/bin/env python3
"""
benchmarks/analyze_niah_failure.py
Phase 4: NIAH Long-Context Retrieval Failure Isolation.
Steps 12-15:
  12. Direct Cache Tier Residency & K/V Reconstruction Cosine Similarity.
  13. Graduated-depth NIAH curve (1.0x to 1.5x trained context).
  14. Disentangling Storage vs Attention Alignment.
  15. H_ratio sensitivity test (0.05 vs 0.15 vs 0.25).
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
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def construct_niah_prompt(tok,
                           total_len: int,
                           needle_depth: float,
                           needle_key: str = "94821") -> Tuple[torch.Tensor, int, int, int]:
    """
    Constructs NIAH prompt: filler_1 + needle_sentence + filler_2 + query.
    Returns:
        (input_ids, needle_start_pos, needle_end_pos, needle_key_token_pos)
    """
    needle_sentence = f" Special notice: the secret retrieval key is {needle_key}. Remember this key. "
    filler_sentence = "The solar system contains eight planets orbiting the Sun in elliptical paths with varying orbital periods. "
    query = " What is the secret retrieval key? Answer: the secret retrieval key is "

    query_ids = tok(query, return_tensors="pt").input_ids
    needle_ids = tok(needle_sentence, return_tensors="pt").input_ids
    filler_ids = tok(filler_sentence, return_tensors="pt").input_ids

    body_len = total_len - query_ids.shape[1] - needle_ids.shape[1] - 4
    insert_pos = int(body_len * needle_depth)

    repeat_count = (body_len // filler_ids.shape[1]) + 2
    filler_pool = filler_ids.repeat(1, repeat_count)[:, :body_len]

    part1 = filler_pool[:, :insert_pos]
    part2 = filler_pool[:, insert_pos:]

    full_prompt = torch.cat([part1, needle_ids, part2, query_ids], dim=-1)[:, :total_len]
    actual_len = full_prompt.shape[1]

    needle_start = insert_pos
    needle_end = insert_pos + needle_ids.shape[1]
    # Find position of the key token itself inside needle_ids
    key_ids = tok(needle_key, add_special_tokens=False).input_ids
    key_pos = needle_start + 1 # approx location

    return full_prompt, needle_start, needle_end, key_pos


def analyze_needle_storage_fidelity(model,
                                    tok,
                                    total_len: int = 3072,
                                    depth: float = 0.20,
                                    needle_key: str = "94821",
                                    rope_mode: str = "a",
                                    trained_len: int = 2048,
                                    k_group_size: int = 16,
                                    pbs_metadata_dtype: str = "fp16",
                                    h_ratio: float = 0.05,
                                    r_size: int = 256) -> Dict[str, Any]:
    """
    Phase 4: Ingests prompt, records unquantized K/V, evaluates cache tier residency,
    computes reconstruction cosine similarity, and executes greedy decode in a single pass.
    """
    patch_mod.R_SIZE = r_size
    patch_mod.H_RATIO = h_ratio
    patch_mod.K_GROUP_SIZE = k_group_size
    patch_mod.PBS_METADATA_DTYPE = pbs_metadata_dtype
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()
    reset_caches(model)

    full_prompt, needle_start, needle_end, key_pos = construct_niah_prompt(tok, total_len, depth, needle_key)
    actual_len = full_prompt.shape[1]

    # Ingest prompt token-by-token while recording original K and V for the needle key token
    orig_needle_k = {}
    orig_needle_v = {}

    with torch.no_grad():
        for p in range(actual_len):
            tok_in = full_prompt[:, p:p+1]
            if rope_mode == "a":
                pos_tensor = torch.tensor([[p]], dtype=torch.int64)
            else:
                rel_pos = p if p < trained_len else (p % trained_len)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)
            out = model(tok_in, position_ids=pos_tensor)

            # Capture original unquantized K and V at needle token position
            if p == needle_start:
                for layer_idx, layer in enumerate(model.model.layers):
                    if hasattr(layer.self_attn, "tri_tier_cache"):
                        c = layer.self_attn.tri_tier_cache
                        if c._engine is not None:
                            eng = c._engine
                            idx = (eng.rw_count - 1) if eng.rw_count < eng.rw_size else (eng.rw_head_index - 1) % eng.rw_size
                            rw_k = torch.from_numpy(eng.get_RW_K())
                            rw_v = torch.from_numpy(eng.get_RW_V())
                            orig_needle_k[layer_idx] = rw_k[idx].clone()
                            orig_needle_v[layer_idx] = rw_v[idx].clone()
                        else:
                            idx = (c.RW_count - 1) if c.RW_count < c.R_size else (c.RW_head_index - 1) % c.R_size
                            orig_needle_k[layer_idx] = c.RW_K_Buffer[idx].clone()
                            orig_needle_v[layer_idx] = c.RW_V_Buffer[idx].clone()

        # Decode 5 tokens directly from prompt output
        gen_tokens = []
        curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_tokens.append(curr_id.item())

        for step in range(4):
            dec_pos = actual_len + step
            if rope_mode == "a":
                pos_tensor = torch.tensor([[dec_pos]], dtype=torch.int64)
            else:
                rel_pos = dec_pos if dec_pos < trained_len else (dec_pos % trained_len)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)
            out = model(curr_id, position_ids=pos_tensor)
            curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens.append(curr_id.item())

    pred_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    success = (needle_key in pred_text)

    # Inspect cache tier residency at query time
    tier_counts = {"sink": 0, "rw": 0, "hh": 0, "pbs": 0, "missing": 0}
    k_cos_sims = []
    v_cos_sims = []

    target_tok_id = needle_start

    from tri_tier import _C

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.self_attn, "tri_tier_cache"):
            continue
        c = layer.self_attn.tri_tier_cache
        eng = c._engine

        # Check Sink
        if target_tok_id < eng.s_count:
            tier_counts["sink"] += 1
            s_k = torch.from_numpy(eng.get_S_K())
            s_v = torch.from_numpy(eng.get_S_V())
            recon_k = s_k[target_tok_id]
            recon_v = s_v[target_tok_id]
        # Check Recent Window
        elif target_tok_id >= (eng.total_processed_tokens - eng.rw_count):
            tier_counts["rw"] += 1
            rw_idx = (eng.rw_head_index + target_tok_id - eng.total_processed_tokens) % eng.rw_size
            rw_k = torch.from_numpy(eng.get_RW_K())
            rw_v = torch.from_numpy(eng.get_RW_V())
            recon_k = rw_k[rw_idx]
            recon_v = rw_v[rw_idx]
        else:
            # Check Heavy Hitter
            hh_ids = eng.get_HH_token_ids()
            hh_match = np.where(hh_ids[:eng.hh_count] == target_tok_id)[0]
            if len(hh_match) > 0:
                tier_counts["hh"] += 1
                hh_k = torch.from_numpy(eng.get_HH_K())
                hh_v = torch.from_numpy(eng.get_HH_V())
                recon_k = hh_k[hh_match[0]]
                recon_v = hh_v[hh_match[0]]
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

                    # Dequantize K block
                    k_packed = eng.get_PBS_K_Packed()[block_idx:block_idx+1]
                    k_scales = eng.get_PBS_K_Scales()[block_idx:block_idx+1]
                    k_zeroes = eng.get_PBS_K_Zeroes()[block_idx:block_idx+1]
                    K_block_out = torch.empty((eng.chunk_size, eng.num_kv_heads, eng.head_dim), dtype=torch.float32)
                    _C.dequantize_k(torch.from_numpy(k_packed), torch.from_numpy(k_scales), torch.from_numpy(k_zeroes),
                                    K_block_out, 1, eng.num_kv_heads, eng.head_dim, eng.k_group_size, eng.pbs_metadata_dtype)
                    recon_k = K_block_out[tok_in_block]

                    # Dequantize V block
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

        if recon_k is not None and layer_idx in orig_needle_k:
            k_sim = F.cosine_similarity(orig_needle_k[layer_idx].float().view(-1), recon_k.float().view(-1), dim=0).item()
            v_sim = F.cosine_similarity(orig_needle_v[layer_idx].float().view(-1), recon_v.float().view(-1), dim=0).item()
            k_cos_sims.append(k_sim)
            v_cos_sims.append(v_sim)

    dominant_tier = max(tier_counts, key=tier_counts.get)
    mean_k_sim = float(np.mean(k_cos_sims)) if k_cos_sims else 0.0
    mean_v_sim = float(np.mean(v_cos_sims)) if v_cos_sims else 0.0

    return {
        "context_length": actual_len,
        "needle_start_pos": needle_start,
        "depth": depth,
        "rope_mode": rope_mode,
        "h_ratio": h_ratio,
        "dominant_tier": dominant_tier,
        "tier_distribution": tier_counts,
        "mean_k_cosine_sim": mean_k_sim,
        "mean_v_cosine_sim": mean_v_sim,
        "storage_intact": (dominant_tier in ["hh", "pbs", "rw"] and mean_k_sim > 0.80),
        "generated_text": pred_text,
        "success": success,
    }


def run_graduated_niah_curve(model,
                             tok,
                             base_trained_len: int = 2048,
                             length_ratios: List[float] = [1.0, 1.25, 1.5],
                             depths: List[float] = [0.20, 0.70, 0.95],
                             rope_modes: List[str] = ["a", "b"],
                             h_ratios: List[float] = [0.05],
                             needle_key: str = "94821") -> List[Dict[str, Any]]:
    """
    Phase 4 Steps 12-15: Runs single-pass graduated NIAH and storage fidelity evaluation.
    """
    records = []

    print("\n" + "=" * 135)
    print(" [PHASE 4] GRADUATED-DEPTH NIAH RETRIEVAL & STORAGE FIDELITY EVALUATION")
    print("=" * 135)
    print(f"{'Length Ratio':<14} | {'Context':<8} | {'Depth':<8} | {'RoPE':<6} | {'H_ratio':<8} | {'Needle Tier':<14} | {'K Recon Cos':<12} | {'Output':<16} | {'Success'}")
    print("-" * 135)

    # 1. Main graduated curve at default h_ratio=0.05
    for ratio in length_ratios:
        total_len = int(ratio * base_trained_len)
        for mode in rope_modes:
            for depth in depths:
                res = analyze_needle_storage_fidelity(
                    model, tok, total_len=total_len, depth=depth,
                    needle_key=needle_key, rope_mode=mode,
                    trained_len=base_trained_len, h_ratio=0.05
                )

                print(f"{ratio:<14.2f} | {res['context_length']:<8d} | {depth:<8.2f} | {mode:<6} | {0.05:<8.2f} | {res['dominant_tier']:<14} | {res['mean_k_cosine_sim']:<12.4f} | {res['generated_text'][:14]:<16} | {'YES' if res['success'] else 'NO'}")

                records.append({
                    "length_ratio": ratio,
                    "context_length": res["context_length"],
                    "depth": depth,
                    "rope_mode": mode,
                    "h_ratio": 0.05,
                    "needle_tier": res["dominant_tier"],
                    "mean_k_cosine_sim": res["mean_k_cosine_sim"],
                    "mean_v_cosine_sim": res["mean_v_cosine_sim"],
                    "storage_intact": res["storage_intact"],
                    "generated_text": res["generated_text"],
                    "success": res["success"],
                })

    # 2. H_ratio sensitivity test at 1.5x context (ratio=1.5, h_ratio=0.15)
    print("-" * 135)
    print(" [PHASE 4 STEP 15] H_RATIO SENSITIVITY TEST (H_ratio = 0.15 at 1.5x Context)")
    print("-" * 135)
    total_len = int(1.5 * base_trained_len)
    for mode in rope_modes:
        for depth in depths:
            res = analyze_needle_storage_fidelity(
                model, tok, total_len=total_len, depth=depth,
                needle_key=needle_key, rope_mode=mode,
                trained_len=base_trained_len, h_ratio=0.15
            )

            print(f"{1.5:<14.2f} | {res['context_length']:<8d} | {depth:<8.2f} | {mode:<6} | {0.15:<8.2f} | {res['dominant_tier']:<14} | {res['mean_k_cosine_sim']:<12.4f} | {res['generated_text'][:14]:<16} | {'YES' if res['success'] else 'NO'}")

            records.append({
                "length_ratio": 1.5,
                "context_length": res["context_length"],
                "depth": depth,
                "rope_mode": mode,
                "h_ratio": 0.15,
                "needle_tier": res["dominant_tier"],
                "mean_k_cosine_sim": res["mean_k_cosine_sim"],
                "mean_v_cosine_sim": res["mean_v_cosine_sim"],
                "storage_intact": res["storage_intact"],
                "generated_text": res["generated_text"],
                "success": res["success"],
            })

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 NIAH Failure Analysis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-csv", type=str, default="benchmarks/results/niah_graduated_curve.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tok = load_model(args.model, dtype=torch.float32)

    # 1. Step 12 Detailed Storage Fidelity Diagnostic
    print("=" * 115)
    print(" [PHASE 4 STEP 12] DIRECT NEEDLE STORAGE & RECONSTRUCTION FIDELITY DIAGNOSTIC")
    print("=" * 115)
    fid = analyze_needle_storage_fidelity(model, tok, total_len=3072, depth=0.20)
    print(f"Total Context Length    : {fid['context_length']}")
    print(f"Needle Position         : Token {fid['needle_start_pos']} (Depth {fid['depth']:.2f})")
    print(f"Dominant Cache Tier     : {fid['dominant_tier'].upper()}")
    print(f"Layer Tier Distribution : {fid['tier_distribution']}")
    print(f"K Reconstructed Cos Sim : {fid['mean_k_cosine_sim']:.6f}")
    print(f"V Reconstructed Cos Sim : {fid['mean_v_cosine_sim']:.6f}")
    print(f"Storage Intact          : {'YES' if fid['storage_intact'] else 'NO'}")

    # 2. Step 13 & 15 Graduated Depth Curve (1.0x to 1.5x) with H_ratio variation
    curve_records = run_graduated_niah_curve(
        model, tok,
        base_trained_len=2048,
        length_ratios=[1.0, 1.25, 1.5],
        depths=[0.20, 0.70, 0.95],
        rope_modes=["a", "b"],
    )
    save_results_to_csv(args.output_csv, curve_records)
