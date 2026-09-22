#!/usr/bin/env python3
"""
benchmarks/benchmark_rope_long_context.py
Phase 3: Scale-Appropriate RoPE Mode Validation past Trained Context Boundary (>= 1.5x - 2.0x).
Evaluates rope_mode='a' (absolute indexing) vs rope_mode='b' (StreamingLLM re-rotation) independently
WITHOUT comparing against vanilla HF (which degrades past context limits).

Metrics reported side-by-side:
1. Segmented Per-Position Perplexity across context windows:
   - Window 1: [0, 0.5 * trained_len] (Early Context)
   - Window 2: [0.5 * trained_len, 1.0 * trained_len] (Approaching Boundary)
   - Window 3: [1.0 * trained_len, 1.5 * trained_len] (Past Trained Boundary)
   - Window 4: [1.5 * trained_len, 2.0 * trained_len] (Deep Extrapolation)
2. Needle-In-A-Haystack (NIAH) retrieval accuracy before and past the trained context boundary.
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
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod


def get_trained_context_length(config) -> Tuple[int, int]:
    """
    Returns (base_trained_length, max_extended_length).
    For Llama-3.2, base is 8192 (original_max_position_embeddings) and extended is 131072.
    For SmolLM-135M, base is 2048.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    base_len = 2048
    if isinstance(rope_scaling, dict) and "original_max_position_embeddings" in rope_scaling:
        base_len = rope_scaling["original_max_position_embeddings"]
    elif hasattr(config, "max_position_embeddings"):
        base_len = config.max_position_embeddings

    max_len = getattr(config, "max_position_embeddings", base_len)
    return base_len, max_len


def evaluate_per_position_ppl(model, 
                              input_ids: torch.Tensor, 
                              rope_mode: str, 
                              trained_len: int,
                              r_size: int = 256,
                              h_ratio: float = 0.05) -> Dict[str, Any]:
    """Evaluates NLL across 4 context segments for a specific RoPE mode."""
    patch_mod.R_SIZE = r_size
    patch_mod.H_RATIO = h_ratio
    patch_mod.K_GROUP_SIZE = 16
    patch_mod.PBS_METADATA_DTYPE = "fp16"
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()
    reset_caches(model)

    seq_len = input_ids.shape[1]
    nlls = []
    positions = []

    # Segment definitions
    seg1_end = int(0.5 * trained_len)
    seg2_end = trained_len
    seg3_end = int(1.5 * trained_len)
    seg4_end = seq_len

    seg_nlls = {"seg1_early": [], "seg2_boundary": [], "seg3_past_boundary": [], "seg4_deep_extrap": []}

    with torch.no_grad():
        for pos in range(seq_len - 1):
            tok_in = input_ids[:, pos:pos+1]
            target_tok = input_ids[:, pos+1]

            if rope_mode == "a":
                pos_tensor = torch.tensor([[pos]], dtype=torch.int64)
            else:
                # Mode 'b': StreamingLLM re-rotation.
                # Pre-eviction and within trained window (pos < trained_len):
                # positions advance naturally with pos (identical to mode 'a' pre-eviction).
                # Past trained boundary (pos >= trained_len): position wraps within trained window.
                if pos < trained_len:
                    rel_pos = pos
                else:
                    rel_pos = (pos % trained_len)
                pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)

            out = model(tok_in, position_ids=pos_tensor)
            logits = out.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs[0, target_tok[0]].item()

            if not math.isnan(nll) and not math.isinf(nll):
                nlls.append(nll)
                positions.append(pos)
                if pos < seg1_end:
                    seg_nlls["seg1_early"].append(nll)
                elif pos < seg2_end:
                    seg_nlls["seg2_boundary"].append(nll)
                elif pos < seg3_end:
                    seg_nlls["seg3_past_boundary"].append(nll)
                else:
                    seg_nlls["seg4_deep_extrap"].append(nll)

    def calc_ppl(vals):
        return math.exp(float(np.mean(vals))) if vals else float("nan")

    return {
        "overall_ppl": calc_ppl(nlls),
        "ppl_seg1_early": calc_ppl(seg_nlls["seg1_early"]),
        "ppl_seg2_boundary": calc_ppl(seg_nlls["seg2_boundary"]),
        "ppl_seg3_past_boundary": calc_ppl(seg_nlls["seg3_past_boundary"]),
        "ppl_seg4_deep_extrap": calc_ppl(seg_nlls["seg4_deep_extrap"]),
        "total_eval_tokens": len(nlls),
    }


def evaluate_niah_at_length(model, 
                            tok, 
                            total_length: int, 
                            trained_len: int, 
                            rope_mode: str, 
                            needle_key: str = "94821") -> List[Dict[str, Any]]:
    """
    Evaluates retrieval accuracy at depths before, near, and past the trained context boundary.
    """
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = 16
    patch_mod.PBS_METADATA_DTYPE = "fp16"
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()

    needle_sentence = f" Special notice: the secret retrieval key is {needle_key}. Remember this key. "
    filler_sentence = "The solar system contains eight planets orbiting the Sun in elliptical paths with varying orbital periods. "
    query = " What is the secret retrieval key? Answer: the secret retrieval key is "

    # Test depths: 0.20 (early), 0.70 (near boundary), 0.95 (recent window)
    depths = [0.20, 0.70, 0.95]
    records = []

    for depth in depths:
        reset_caches(model)
        query_ids = tok(query, return_tensors="pt").input_ids
        needle_ids = tok(needle_sentence, return_tensors="pt").input_ids
        filler_ids = tok(filler_sentence, return_tensors="pt").input_ids

        body_len = total_length - query_ids.shape[1] - needle_ids.shape[1] - 4
        insert_pos = int(body_len * depth)

        repeat_count = (body_len // filler_ids.shape[1]) + 2
        filler_pool = filler_ids.repeat(1, repeat_count)[:, :body_len]

        part1 = filler_pool[:, :insert_pos]
        part2 = filler_pool[:, insert_pos:]

        full_prompt = torch.cat([part1, needle_ids, part2, query_ids], dim=-1)[:, :total_length]
        actual_len = full_prompt.shape[1]

        # Ingest prompt
        with torch.no_grad():
            for p in range(actual_len):
                tok_in = full_prompt[:, p:p+1]
                if rope_mode == "a":
                    pos_tensor = torch.tensor([[p]], dtype=torch.int64)
                else:
                    rel_pos = p if p < trained_len else (p % trained_len)
                    pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)
                out = model(tok_in, position_ids=pos_tensor)

            # Generate 5 tokens
            curr_pos = actual_len
            gen_tokens = []
            curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens.append(curr_id.item())

            for step in range(4):
                if rope_mode == "a":
                    pos_tensor = torch.tensor([[curr_pos + step]], dtype=torch.int64)
                else:
                    dec_pos = curr_pos + step
                    rel_pos = dec_pos if dec_pos < trained_len else (dec_pos % trained_len)
                    pos_tensor = torch.tensor([[rel_pos]], dtype=torch.int64)
                out = model(curr_id, position_ids=pos_tensor)
                curr_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen_tokens.append(curr_id.item())

        retrieved_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        success = (needle_key in retrieved_text)

        needle_token_pos = insert_pos
        is_past_boundary = (needle_token_pos > trained_len)

        records.append({
            "context_length": actual_len,
            "depth": depth,
            "needle_pos": needle_token_pos,
            "trained_boundary": trained_len,
            "is_past_boundary": is_past_boundary,
            "rope_mode": rope_mode,
            "retrieved_text": retrieved_text,
            "success": success,
        })

    return records


def run_rope_long_context_benchmark(model_name: str = DEFAULT_MODEL_ID,
                                    eval_len: int = None,
                                    output_dir: str = "benchmarks/results",
                                    seed: int = 42) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    model, tok = load_model(model_name, dtype=torch.float32)
    base_trained_len, max_extended_len = get_trained_context_length(model.config)

    # Set evaluation length to >= 1.5x trained base length
    if eval_len is None:
        eval_len = int(1.5 * base_trained_len)

    print("=" * 125)
    print(f" [PHASE 3] SCALE-APPROPRIATE ROPE VALIDATION PAST TRAINED CONTEXT (Model: {model_name})")
    print(f" Base Trained Window: {base_trained_len} tokens | Extended Limit: {max_extended_len} tokens")
    print(f" Target Evaluation Length: {eval_len} tokens (Ratio: {eval_len / base_trained_len:.2f}x Trained Context)")
    print("=" * 125)

    corpus = get_evaluation_corpus(min_tokens=eval_len + 100, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :eval_len]
    actual_len = input_ids.shape[1]

    # 1. Per-Position Perplexity for Mode 'a' vs Mode 'b'
    print("\n[1/2] Computing Segmented Per-Position Perplexity for Mode 'a' and Mode 'b'...")
    res_a = evaluate_per_position_ppl(model, input_ids, rope_mode="a", trained_len=base_trained_len)
    print(f"  -> Mode 'a' Overall PPL: {res_a['overall_ppl']:.4f} (Early={res_a['ppl_seg1_early']:.2f}, Boundary={res_a['ppl_seg2_boundary']:.2f}, PastBoundary={res_a['ppl_seg3_past_boundary']:.2f})")

    res_b = evaluate_per_position_ppl(model, input_ids, rope_mode="b", trained_len=base_trained_len)
    print(f"  -> Mode 'b' Overall PPL: {res_b['overall_ppl']:.4f} (Early={res_b['ppl_seg1_early']:.2f}, Boundary={res_b['ppl_seg2_boundary']:.2f}, PastBoundary={res_b['ppl_seg3_past_boundary']:.2f})")

    # 2. NIAH Retrieval across Boundary
    print("\n[2/2] Evaluating NIAH Needle Retrieval across Trained Boundary...")
    niah_a = evaluate_niah_at_length(model, tok, total_length=actual_len, trained_len=base_trained_len, rope_mode="a")
    niah_b = evaluate_niah_at_length(model, tok, total_length=actual_len, trained_len=base_trained_len, rope_mode="b")

    print("\n" + "=" * 125)
    print(" SIDE-BY-SIDE ROPE MODE EVALUATION PAST TRAINED CONTEXT BOUNDARY")
    print("=" * 125)
    print(f"{'Metric / Context Window':<42} | {'RoPE Mode A (Absolute)':<30} | {'RoPE Mode B (StreamingLLM)':<30}")
    print("-" * 125)
    print(f"{'Overall Perplexity @ ' + str(actual_len) + ' toks':<42} | {res_a['overall_ppl']:<30.4f} | {res_b['overall_ppl']:<30.4f}")
    print(f"{'Segment 1 PPL [0 to 0.5x Trained Base]':<42} | {res_a['ppl_seg1_early']:<30.4f} | {res_b['ppl_seg1_early']:<30.4f}")
    print(f"{'Segment 2 PPL [0.5x to 1.0x Trained Base]':<42} | {res_a['ppl_seg2_boundary']:<30.4f} | {res_b['ppl_seg2_boundary']:<30.4f}")
    print(f"{'Segment 3 PPL [1.0x to 1.5x Past Boundary]':<42} | {res_a['ppl_seg3_past_boundary']:<30.4f} | {res_b['ppl_seg3_past_boundary']:<30.4f}")
    if not math.isnan(res_a["ppl_seg4_deep_extrap"]):
        print(f"{'Segment 4 PPL [1.5x to 2.0x Deep Extrap]':<42} | {res_a['ppl_seg4_deep_extrap']:<30.4f} | {res_b['ppl_seg4_deep_extrap']:<30.4f}")

    print("-" * 125)
    for na, nb in zip(niah_a, niah_b):
        lbl = f"NIAH Depth {na['depth']:.2f} (pos={na['needle_pos']}, past_bnd={na['is_past_boundary']})"
        out_a = f"{'YES' if na['success'] else 'NO'} ('{na['retrieved_text'][:12]}')"
        out_b = f"{'YES' if nb['success'] else 'NO'} ('{nb['retrieved_text'][:12]}')"
        print(f"{lbl:<42} | {out_a:<30} | {out_b:<30}")
    print("=" * 125)

    summary_rows = [
        {
            "model_id": model_name,
            "trained_base_length": base_trained_len,
            "evaluated_length": actual_len,
            "length_ratio_vs_base": actual_len / base_trained_len,
            "mode_a_overall_ppl": res_a["overall_ppl"],
            "mode_a_seg1_ppl": res_a["ppl_seg1_early"],
            "mode_a_seg2_ppl": res_a["ppl_seg2_boundary"],
            "mode_a_seg3_ppl": res_a["ppl_seg3_past_boundary"],
            "mode_b_overall_ppl": res_b["overall_ppl"],
            "mode_b_seg1_ppl": res_b["ppl_seg1_early"],
            "mode_b_seg2_ppl": res_b["ppl_seg2_boundary"],
            "mode_b_seg3_ppl": res_b["ppl_seg3_past_boundary"],
            "mode_a_niah_accuracy": sum(1 for n in niah_a if n["success"]) / len(niah_a),
            "mode_b_niah_accuracy": sum(1 for n in niah_b if n["success"]) / len(niah_b),
            "seed": seed,
        }
    ]
    csv_path = os.path.join(output_dir, "rope_modes_long_context_results.csv")
    save_results_to_csv(csv_path, summary_rows)
    print(f"[CSV Saved] -> {csv_path}")

    return {"summary": summary_rows, "niah_a": niah_a, "niah_b": niah_b}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale-Appropriate RoPE Long-Context Validation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or path")
    parser.add_argument("--eval-len", type=int, default=None, help="Context length to evaluate (defaults to 1.5x trained)")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_rope_long_context_benchmark(
        model_name=args.model,
        eval_len=args.eval_len,
        output_dir=args.output_dir,
        seed=args.seed
    )
