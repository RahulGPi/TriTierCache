#!/usr/bin/env python3
"""
benchmarks/analyze_cosine_distribution.py
Evaluates fine-grained Cosine Similarity Distribution for Group-16 vs Group-32:
1. Reconstructed Key Cosine Similarity across sequence length:
   - Chunk-by-chunk breakdown across sequence (chunks of 64 tokens up to 2048 tokens).
   - Global and per-chunk MINIMUM, MEAN, and STD cosine similarity.
2. Step-by-step Logit Drift during autoregressive decoding:
   - Minimum cosine similarity, mean, std, and exact step where drift starts.
Outputs:
   benchmarks/results/cosine_sim_distribution_chunks.csv
   benchmarks/results/cosine_sim_decode_steps.csv
"""

import os
import argparse
import numpy as np
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
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
import src.tri_tier.integration.patch_llama as patch_mod

try:
    import tri_tier._C as _C
except ImportError:
    _C = None


def evaluate_key_reconstruction_chunks(model, 
                                       input_ids: torch.Tensor, 
                                       chunk_size: int = 64) -> List[Dict[str, Any]]:
    """
    Evaluates original vs reconstructed Key vectors across sequence length chunk-by-chunk,
    comparing group-16 vs group-32 AVX2 quantization directly.
    """
    seq_len = input_ids.shape[1]
    num_chunks = seq_len // chunk_size

    # Ingest sequence through unquantized/vanilla attention to collect ground-truth K
    remove_patch()
    reset_caches(model)
    
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=True)
        # Extract keys from first layer (supporting DynamicCache, DynamicLayer, or tuple)
        pkv = out.past_key_values
        if hasattr(pkv, "layers") and len(pkv.layers) > 0:
            k_tensor = pkv.layers[0].keys
        elif hasattr(pkv, "key_cache"):
            k_tensor = pkv.key_cache[0]
        elif isinstance(pkv, (list, tuple)):
            k_tensor = pkv[0][0] if isinstance(pkv[0], (list, tuple)) else pkv[0]
        else:
            k_tensor = getattr(pkv, "keys", None)
        # Shape: (batch_size, num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim)
        orig_k = k_tensor.squeeze(0).permute(1, 0, 2).contiguous().float()



    num_heads = orig_k.shape[1]
    head_dim = orig_k.shape[2]

    # Ensure C extension is available
    assert _C is not None, "tri_tier._C AVX2 extension not found!"

    # Process K in 32-token blocks (must be multiple of 32 for group-32 quantization)
    usable_tokens = (seq_len // 32) * 32
    K_np = orig_k[:usable_tokens].numpy().astype(np.float32)

    dequant_16 = np.zeros_like(K_np)
    dequant_32 = np.zeros_like(K_np)

    num_32_blocks = usable_tokens // 32

    for b in range(num_32_blocks):
        sub_k = K_np[b * 32 : (b + 1) * 32]

        # Group-16: two 16-token quantizations
        packed_16 = np.zeros((2, num_heads, head_dim), dtype=np.int32)
        scale_16 = np.zeros((2, num_heads, head_dim), dtype=np.uint16)
        zero_16 = np.zeros((2, num_heads, head_dim), dtype=np.uint16)

        _C.quantize_k_block(sub_k[:16], packed_16[0], scale_16[0], zero_16[0], num_heads, head_dim, 16, "fp16")
        _C.quantize_k_block(sub_k[16:], packed_16[1], scale_16[1], zero_16[1], num_heads, head_dim, 16, "fp16")
        _C.dequantize_k(packed_16, scale_16, zero_16, dequant_16[b * 32 : (b + 1) * 32], 2, num_heads, head_dim, 16, "fp16")

        # Group-32: one 32-token quantization
        packed_32 = np.zeros((2, num_heads, head_dim), dtype=np.int32)
        scale_32 = np.zeros((1, num_heads, head_dim), dtype=np.uint16)
        zero_32 = np.zeros((1, num_heads, head_dim), dtype=np.uint16)

        _C.quantize_k_block(sub_k, packed_32, scale_32, zero_32, num_heads, head_dim, 32, "fp16")
        _C.dequantize_k(packed_32, scale_32, zero_32, dequant_32[b * 32 : (b + 1) * 32], 1, num_heads, head_dim, 32, "fp16")

    # Compute per-token cosine similarity across heads
    t_orig = torch.from_numpy(K_np)
    t_deq16 = torch.from_numpy(dequant_16)
    t_deq32 = torch.from_numpy(dequant_32)

    cos16_per_token = F.cosine_similarity(t_orig, t_deq16, dim=-1).mean(dim=-1).numpy()
    cos32_per_token = F.cosine_similarity(t_orig, t_deq32, dim=-1).mean(dim=-1).numpy()

    chunk_rows = []
    chunk_tokens = 64
    num_chunks = usable_tokens // chunk_tokens

    print(f"\n--- Per-Chunk Key Reconstruction Cosine Similarity (PBS Tier, Chunk Size = {chunk_tokens}) ---")
    print(f"{'Chunk':<6} | {'Tokens':<14} | {'G16 Mean':<10} | {'G16 Min':<10} | {'G32 Mean':<10} | {'G32 Min':<10} | {'Delta Min'}")
    print("-" * 75)

    for c_idx in range(num_chunks):
        start = c_idx * chunk_tokens
        end = start + chunk_tokens
        g16_slice = cos16_per_token[start:end]
        g32_slice = cos32_per_token[start:end]

        mean16 = float(np.mean(g16_slice))
        min16 = float(np.min(g16_slice))
        std16 = float(np.std(g16_slice))

        mean32 = float(np.mean(g32_slice))
        min32 = float(np.min(g32_slice))
        std32 = float(np.std(g32_slice))

        delta_min = min32 - min16
        print(f"{c_idx:<6d} | {start:<6d}-{end:<7d} | {mean16:<10.6f} | {min16:<10.6f} | {mean32:<10.6f} | {min32:<10.6f} | {delta_min:+10.6f}")

        chunk_rows.append({
            "chunk_index": c_idx,
            "token_start": start,
            "token_end": end,
            "g16_mean_cosine_sim": mean16,
            "g16_min_cosine_sim": min16,
            "g16_std_cosine_sim": std16,
            "g32_mean_cosine_sim": mean32,
            "g32_min_cosine_sim": min32,
            "g32_std_cosine_sim": std32,
            "delta_mean": mean32 - mean16,
            "delta_min": delta_min,
        })

    # Overall Summary Row
    overall_mean16 = float(np.mean(cos16_per_token))
    overall_min16 = float(np.min(cos16_per_token))
    overall_mean32 = float(np.mean(cos32_per_token))
    overall_min32 = float(np.min(cos32_per_token))

    print("-" * 75)
    print(f"OVERALL | 0-{usable_tokens:<10d} | {overall_mean16:<10.6f} | {overall_min16:<10.6f} | {overall_mean32:<10.6f} | {overall_min32:<10.6f} | {overall_min32 - overall_min16:+10.6f}")

    chunk_rows.append({
        "chunk_index": -1,
        "token_start": 0,
        "token_end": usable_tokens,
        "g16_mean_cosine_sim": overall_mean16,
        "g16_min_cosine_sim": overall_min16,
        "g16_std_cosine_sim": float(np.std(cos16_per_token)),
        "g32_mean_cosine_sim": overall_mean32,
        "g32_min_cosine_sim": overall_min32,
        "g32_std_cosine_sim": float(np.std(cos32_per_token)),
        "delta_mean": overall_mean32 - overall_mean16,
        "delta_min": overall_min32 - overall_min16,
    })


    return chunk_rows


def evaluate_decode_step_drift(model, tok, prompt_len: int = 512, decode_steps: int = 50) -> List[Dict[str, Any]]:
    """
    Evaluates step-by-step logit cosine similarity and MSE during decoding,
    comparing Group-16 vs Group-32 side-by-side to pinpoint exactly where drift starts.
    """
    corpus = get_evaluation_corpus(min_tokens=prompt_len + 100, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :prompt_len]

    # 1. Vanilla Baseline
    remove_patch()
    reset_caches(model)
    v_logits = []
    curr_ids = input_ids.clone()
    past_kv = None
    with torch.no_grad():
        out = model(curr_ids, use_cache=True)
        past_kv = out.past_key_values
        for step in range(decode_steps):
            last_logits = out.logits[:, -1, :].clone().float()
            v_logits.append(last_logits)
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            out = model(next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values

    # 2. TriTier Group-16
    patch_mod.R_SIZE = 256
    patch_mod.H_RATIO = 0.05
    patch_mod.K_GROUP_SIZE = 16
    patch_mod.PBS_METADATA_DTYPE = "fp16"
    patch_mod.SCORE_DECAY = 0.999
    apply_patch()
    reset_caches(model)
    g16_logits = []
    curr_ids = input_ids.clone()
    with torch.no_grad():
        out = model(curr_ids)
        for step in range(decode_steps):
            last_logits = out.logits[:, -1, :].clone().float()
            g16_logits.append(last_logits)
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            pos = prompt_len + step
            out = model(next_id, position_ids=torch.tensor([[pos]], dtype=torch.int64))

    # 3. TriTier Group-32
    patch_mod.K_GROUP_SIZE = 32
    apply_patch()
    reset_caches(model)
    g32_logits = []
    curr_ids = input_ids.clone()
    with torch.no_grad():
        out = model(curr_ids)
        for step in range(decode_steps):
            last_logits = out.logits[:, -1, :].clone().float()
            g32_logits.append(last_logits)
            next_id = last_logits.argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            pos = prompt_len + step
            out = model(next_id, position_ids=torch.tensor([[pos]], dtype=torch.int64))

    step_rows = []
    print(f"\n--- Per-Step Decode Logit Drift (Prompt = {prompt_len}, Steps = {decode_steps}) ---")
    print(f"{'Step':<6} | {'G16 Cos Sim':<12} | {'G32 Cos Sim':<12} | {'G16 Match':<10} | {'G32 Match':<10} | {'Drift Status'}")
    print("-" * 75)

    for step in range(decode_steps):
        v = v_logits[step]
        t16 = g16_logits[step]
        t32 = g32_logits[step]

        cos16 = F.cosine_similarity(v, t16, dim=-1).item()
        cos32 = F.cosine_similarity(v, t32, dim=-1).item()

        m16 = bool(v.argmax(dim=-1) == t16.argmax(dim=-1))
        m32 = bool(v.argmax(dim=-1) == t32.argmax(dim=-1))

        status = "Both Match" if (m16 and m32) else ("G16 only" if m16 else ("G32 only" if m32 else "Both Diverged"))
        print(f"{step+1:<6d} | {cos16:<12.6f} | {cos32:<12.6f} | {str(m16):<10} | {str(m32):<10} | {status}")

        step_rows.append({
            "step": step + 1,
            "g16_cosine_sim": cos16,
            "g32_cosine_sim": cos32,
            "g16_match": m16,
            "g32_match": m32,
            "status": status,
        })

    return step_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    model, tok = load_model(args.model, dtype=torch.float32)

    # 1. Key reconstruction chunk analysis
    corpus = get_evaluation_corpus(min_tokens=args.prompt_len + 100, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :args.prompt_len]

    chunk_results = evaluate_key_reconstruction_chunks(model, input_ids, chunk_size=64)
    save_results_to_csv(os.path.join(args.output_dir, "cosine_sim_distribution_chunks.csv"), chunk_results)

    # 2. Decode step drift analysis
    step_results = evaluate_decode_step_drift(model, tok, prompt_len=512, decode_steps=args.decode_steps)
    save_results_to_csv(os.path.join(args.output_dir, "cosine_sim_decode_steps.csv"), step_results)


if __name__ == "__main__":
    main()
