#!/usr/bin/env python3
"""
benchmarks/check_scale_underflow.py
Direct detector for scale and zero underflow / NaN / subnormal representation in FP16 vs FP32.
Scans stored scale and zero buffers after K/V quantization passes on real model activations.
Outputs:
  benchmarks/results/scale_underflow_results.csv:
    chunk_id, tier, computed_scale_fp32, stored_scale_fp16, underflowed (bool)
"""

import os
import math
import argparse
import numpy as np
import torch
from typing import List, Dict, Any

from benchmarks.common import (
    load_model,
    get_evaluation_corpus,
    save_results_to_csv,
    RunConfig,
    set_seed,
    DEFAULT_MODEL_ID,
)
from src.tri_tier.integration.patch_llama import remove_patch, reset_caches
import tri_tier._C as _C


def detect_scale_underflow(model_id: str = DEFAULT_MODEL_ID,
                           prompt_len: int = 512,
                           output_dir: str = "benchmarks/results",
                           seed: int = 42) -> List[Dict[str, Any]]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    run_cfg = RunConfig(model_id=model_id, seed=seed)

    print("=" * 90)
    print(f" [PHASE 1.2] DIRECT SCALE & ZERO UNDERFLOW DETECTOR (Model: {model_id})")
    print("=" * 90)

    model, tok = load_model(model_id)
    corpus = get_evaluation_corpus(min_tokens=prompt_len + 50, tokenizer=tok)
    input_ids = tok(corpus, return_tensors="pt").input_ids[:, :prompt_len]

    remove_patch()
    reset_caches(model)

    k_dict = {}
    v_dict = {}
    handles = []

    for idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        def make_k_hook(l_idx):
            def hook(mod, inp, out):
                k_dict[l_idx] = out.squeeze(0).float().detach()
            return hook

        def make_v_hook(l_idx):
            def hook(mod, inp, out):
                v_dict[l_idx] = out.squeeze(0).float().detach()
            return hook

        handles.append(attn.k_proj.register_forward_hook(make_k_hook(idx)))
        handles.append(attn.v_proj.register_forward_hook(make_v_hook(idx)))

    with torch.no_grad():
        _ = model(input_ids)

    for h in handles:
        h.remove()

    print(f"Captured KV activations from {len(k_dict)} layers across {prompt_len} tokens.")

    records = []
    total_chunks_scanned = 0
    total_underflow_count = 0
    total_nan_count = 0
    total_subnormal_count = 0

    chunk_size = 16

    for layer_idx in sorted(k_dict.keys()):
        K_raw = k_dict[layer_idx]  # [q_len, num_kv_heads * head_dim]
        V_raw = v_dict[layer_idx]

        num_kv_heads = model.model.layers[layer_idx].self_attn.config.num_key_value_heads
        head_dim = model.model.layers[layer_idx].self_attn.head_dim
        q_len = K_raw.shape[0]

        K_act = K_raw.view(q_len, num_kv_heads, head_dim)
        V_act = V_raw.view(q_len, num_kv_heads, head_dim)

        num_chunks = q_len // chunk_size

        for c_idx in range(num_chunks):
            chunk_start = c_idx * chunk_size
            k_chunk = K_act[chunk_start:chunk_start + chunk_size].contiguous()
            v_chunk = V_act[chunk_start:chunk_start + chunk_size].contiguous()

            # 1. Quantize K block
            packed_k = torch.empty((num_kv_heads, head_dim), dtype=torch.int32)
            scale_k_fp32 = torch.empty((num_kv_heads, head_dim), dtype=torch.float32)
            zero_k_fp32 = torch.empty((num_kv_heads, head_dim), dtype=torch.float32)

            _C.quantize_k_block(
                k_chunk.data_ptr(),
                packed_k.data_ptr(),
                scale_k_fp32.data_ptr(),
                zero_k_fp32.data_ptr(),
                num_kv_heads,
                head_dim
            )

            # Convert to FP16 and test underflow / subnormals
            scale_k_fp16 = scale_k_fp32.to(torch.float16)
            k_scales_flat = scale_k_fp32.flatten()
            k_scales_fp16_flat = scale_k_fp16.flatten()

            for elem_idx in range(k_scales_flat.numel()):
                s_fp32 = k_scales_flat[elem_idx].item()
                s_fp16 = float(k_scales_fp16_flat[elem_idx].item())
                underflowed = (s_fp32 > 0.0 and s_fp16 == 0.0)
                is_subnormal = (0.0 < abs(s_fp16) < 6.103515625e-5)
                is_nan_or_inf = math.isnan(s_fp16) or math.isinf(s_fp16)

                if underflowed:
                    total_underflow_count += 1
                if is_subnormal:
                    total_subnormal_count += 1
                if is_nan_or_inf:
                    total_nan_count += 1

                records.append({
                    "chunk_id": f"L{layer_idx}_K_chunk{c_idx}_c{elem_idx}",
                    "tier": "Tier 3 (PBS-K)",
                    "computed_scale_fp32": s_fp32,
                    "stored_scale_fp16": s_fp16,
                    "underflowed": underflowed,
                    "is_subnormal": is_subnormal,
                })

            # 2. Quantize V block
            quant_head_dim = (head_dim + 15) // 16
            packed_v = torch.empty((chunk_size, num_kv_heads, quant_head_dim), dtype=torch.int32)
            scale_v_fp32 = torch.empty((chunk_size, num_kv_heads, 1), dtype=torch.float32)
            zero_v_fp32 = torch.empty((chunk_size, num_kv_heads, 1), dtype=torch.float32)

            _C.quantize_v_block(
                v_chunk.data_ptr(),
                packed_v.data_ptr(),
                scale_v_fp32.data_ptr(),
                zero_v_fp32.data_ptr(),
                num_kv_heads,
                head_dim
            )

            scale_v_fp16 = scale_v_fp32.to(torch.float16)
            v_scales_flat = scale_v_fp32.flatten()
            v_scales_fp16_flat = scale_v_fp16.flatten()

            for elem_idx in range(v_scales_flat.numel()):
                s_fp32 = v_scales_flat[elem_idx].item()
                s_fp16 = float(v_scales_fp16_flat[elem_idx].item())
                underflowed = (s_fp32 > 0.0 and s_fp16 == 0.0)
                is_subnormal = (0.0 < abs(s_fp16) < 6.103515625e-5)
                is_nan_or_inf = math.isnan(s_fp16) or math.isinf(s_fp16)

                if underflowed:
                    total_underflow_count += 1
                if is_subnormal:
                    total_subnormal_count += 1
                if is_nan_or_inf:
                    total_nan_count += 1

                records.append({
                    "chunk_id": f"L{layer_idx}_V_chunk{c_idx}_t{elem_idx}",
                    "tier": "Tier 3 (PBS-V)",
                    "computed_scale_fp32": s_fp32,
                    "stored_scale_fp16": s_fp16,
                    "underflowed": underflowed,
                    "is_subnormal": is_subnormal,
                })

            total_chunks_scanned += 1

    print("\n" + "=" * 90)
    print(" SCALE & ZERO UNDERFLOW SCAN SUMMARY")
    print("=" * 90)
    print(f"Total 16-Token Chunks Scanned : {total_chunks_scanned}")
    print(f"Total Scale Entries Tested   : {len(records)}")
    print(f"Underflowed Entries (FP16==0): {total_underflow_count}")
    print(f"Subnormal Entries in FP16    : {total_subnormal_count}")
    print(f"NaN / Inf Entries in FP16    : {total_nan_count}")
    print("=" * 90)

    # Save to CSV
    csv_path = os.path.join(output_dir, "scale_underflow_results.csv")
    save_results_to_csv(csv_path, records[:5000])
    print(f"Scale underflow report saved -> {csv_path}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct Scale Underflow Detector")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Model name or path")
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt length to test")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory for CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    detect_scale_underflow(model_id=args.model, prompt_len=args.prompt_len, output_dir=args.output_dir, seed=args.seed)
