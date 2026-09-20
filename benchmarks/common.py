"""
benchmarks/common.py
Canonical shared infrastructure for TriTierCache benchmarking suite:
- Model and tokenizer loading
- Canonical per-tier memory accounting (measure_cache_bytes)
- Hardware memory bandwidth plausibility gating (check_bandwidth_plausible)
- Unified RunConfig capturing all execution hyperparameters
- Deterministic random seeding
"""

import os
import gc
import csv
import math
import random
import psutil
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM-135M"


@dataclass
class RunConfig:
    """Dataclass capturing every configuration knob for reproducible benchmark execution."""
    model_id: str = DEFAULT_MODEL_ID
    dtype: str = "float32"
    R_size: int = 256
    H_ratio: float = 0.05
    K_group_size: int = 16
    rope_mode: str = "a"  # 'a' (current absolute-position) | 'b' (StreamingLLM re-rotation)
    pbs_metadata_dtype: str = "fp16"  # 'fp16' | 'fp32'
    hh_decay: Optional[float] = None  # None (cumulative sum) | float (e.g. 0.999)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hh_decay_str"] = "none" if self.hh_decay is None else f"{self.hh_decay:.4f}"
        return d


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_process_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    gc.collect()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def save_results_to_csv(filepath: str, rows: List[Dict[str, Any]]) -> None:
    """Writes a list of dictionaries to a CSV file."""
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV Saved] -> {filepath}")


def check_bandwidth_plausible(latency_ms: float, model_bytes: int, threshold_gb_s: float = 100.0) -> Tuple[bool, float]:
    """
    Checks if measured decode latency implies a physically plausible memory bandwidth on CPU.
    Returns:
        (is_plausible: bool, implied_gb_s: float)
    """
    sec_per_tok = latency_ms / 1000.0
    if sec_per_tok <= 0:
        return False, float("inf")
    implied_gb_s = (model_bytes / sec_per_tok) / 1e9
    is_plausible = (implied_gb_s <= threshold_gb_s)
    return is_plausible, implied_gb_s


def load_model(model_id: str = DEFAULT_MODEL_ID,
               device: str = "cpu",
               dtype: torch.dtype = torch.float32,
               use_synthetic_if_missing: bool = True) -> Tuple[Any, Any]:
    """
    Single canonical model and tokenizer loader used across all benchmark scripts.
    """
    try:
        print(f"Loading model: '{model_id}' (dtype={dtype}, device={device})...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.to(device)
        model.eval()
        print(f"Successfully loaded '{model_id}'")
        return model, tokenizer
    except Exception as e:
        if not use_synthetic_if_missing:
            raise e
        print(f"Warning: Failed to load '{model_id}' ({e}). Falling back to synthetic Llama architecture...")
        is_1b = "1b" in str(model_id).lower() or "llama-3" in str(model_id).lower()
        if is_1b:
            config = LlamaConfig(
                vocab_size=128256,
                hidden_size=2048,
                intermediate_size=8192,
                num_hidden_layers=16,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=64,
                max_position_embeddings=131072,
                rope_theta=500000.0,
                pad_token_id=128004,
                bos_token_id=128000,
                eos_token_id=128001,
            )
        else:
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=32,
                max_position_embeddings=32768,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
        model = LlamaForCausalLM(config).to(dtype).to(device)
        model.eval()

        class SyntheticTokenizer:
            def __init__(self, vocab_size=32000):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"

            def __call__(self, text, return_tensors="pt"):
                if isinstance(text, str):
                    words = text.split()
                    ids = [hash(w) % (self.vocab_size - 10) + 10 for w in words]
                    if not ids:
                        ids = [self.bos_token_id]
                else:
                    ids = [self.bos_token_id]
                t = torch.tensor([ids], dtype=torch.int64)
                class Tokens:
                    def __init__(self, tensor):
                        self.input_ids = tensor
                return Tokens(t)

            def encode(self, text, return_tensors="pt"):
                return self(text, return_tensors=return_tensors).input_ids

            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return " ".join([f"tok_{t}" for t in token_ids])

        return model, SyntheticTokenizer(config.vocab_size)


def measure_cache_bytes(cache_obj: Any = None,
                        total_tokens: Optional[int] = None,
                        config: Optional[Any] = None,
                        num_kv_heads: Optional[int] = None,
                        head_dim: Optional[int] = None,
                        num_layers: int = 1,
                        sink_size: int = 4,
                        r_size: int = 256,
                        h_ratio: float = 0.05,
                        chunk_size: int = 16,
                        k_group_size: int = 16,
                        pbs_metadata_dtype: str = "fp16") -> Dict[str, Any]:
    """
    Canonical single source of truth for KV cache memory accounting.
    Returns per-tier byte breakdown:
      - sink_bytes
      - rw_bytes
      - hh_bytes
      - pbs_payload_bytes
      - pbs_metadata_bytes
      - total_bytes (and total_mb)
    """
    # Extract dimensions from model config or cache_obj if available
    if config is not None:
        num_layers = getattr(config, "num_hidden_layers", num_layers)
        num_kv_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", num_kv_heads))
        head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 0) // getattr(config, "num_attention_heads", 1))

    if cache_obj is not None:
        if hasattr(cache_obj, "num_heads"):
            num_kv_heads = cache_obj.num_heads
        if hasattr(cache_obj, "head_dim"):
            head_dim = cache_obj.head_dim
        if hasattr(cache_obj, "R_size"):
            r_size = cache_obj.R_size
        if hasattr(cache_obj, "H_ratio"):
            h_ratio = cache_obj.H_ratio
        if total_tokens is None:
            total_tokens = getattr(cache_obj, "Total_Processed_Tokens", 0)

    if num_kv_heads is None or head_dim is None:
        # Fallback to standard 135M defaults if unspecified
        num_kv_heads = 3
        head_dim = 64

    if total_tokens is None:
        total_tokens = 0

    meta_bytes_per_elem = 2 if pbs_metadata_dtype == "fp16" else 4
    fp32_bytes_per_elem = 4  # dense buffers in FP32
    fp16_bytes_per_elem = 2  # baseline comparison standard

    # 1. Token tier partitioning
    max_hh = math.ceil(total_tokens * h_ratio)
    actual_sinks = min(sink_size, total_tokens)
    actual_rw = min(r_size, max(0, total_tokens - actual_sinks))
    actual_hh = min(max_hh, max(0, total_tokens - actual_sinks - actual_rw))
    actual_pbs = max(0, total_tokens - actual_sinks - actual_rw - actual_hh)

    # 2. Per-tier byte calculations per layer
    # Sink: K + V (FP32)
    layer_sink_bytes = 2 * actual_sinks * num_kv_heads * head_dim * fp32_bytes_per_elem

    # Recent Window: K + V (FP32)
    layer_rw_bytes = 2 * actual_rw * num_kv_heads * head_dim * fp32_bytes_per_elem

    # Heavy Hitters: K + V (FP32) + token_id (int64) + score (float32)
    layer_hh_bytes = (2 * actual_hh * num_kv_heads * head_dim * fp32_bytes_per_elem +
                      actual_hh * 8 + actual_hh * 4)

    # PBS Payload: 2-bit K (0.25 bytes) + 2-bit V (0.25 bytes)
    pbs_k_payload = actual_pbs * num_kv_heads * head_dim * 0.25
    pbs_v_payload = actual_pbs * num_kv_heads * head_dim * 0.25
    layer_pbs_payload_bytes = pbs_k_payload + pbs_v_payload

    # PBS Metadata:
    k_blocks = math.ceil(actual_pbs / k_group_size) if actual_pbs > 0 else 0
    pbs_k_meta = k_blocks * num_kv_heads * head_dim * (2 * meta_bytes_per_elem)  # scale + zero
    pbs_v_meta = actual_pbs * num_kv_heads * (2 * meta_bytes_per_elem)          # scale + zero
    pbs_ids = actual_pbs * 8                                                    # int64 token id
    layer_pbs_metadata_bytes = pbs_k_meta + pbs_v_meta + pbs_ids

    # Layer total
    layer_total_bytes = (layer_sink_bytes + layer_rw_bytes + layer_hh_bytes +
                         layer_pbs_payload_bytes + layer_pbs_metadata_bytes)

    # Multiplied across model layers
    total_sink_bytes = layer_sink_bytes * num_layers
    total_rw_bytes = layer_rw_bytes * num_layers
    total_hh_bytes = layer_hh_bytes * num_layers
    total_pbs_payload_bytes = layer_pbs_payload_bytes * num_layers
    total_pbs_metadata_bytes = layer_pbs_metadata_bytes * num_layers
    total_bytes = layer_total_bytes * num_layers

    # Vanilla baselines (uncompressed)
    vanilla_fp16_bytes = 2 * num_layers * total_tokens * num_kv_heads * head_dim * fp16_bytes_per_elem
    vanilla_fp32_bytes = 2 * num_layers * total_tokens * num_kv_heads * head_dim * fp32_bytes_per_elem

    compression_ratio_vs_fp16 = vanilla_fp16_bytes / max(1.0, total_bytes)
    compression_ratio_vs_fp32 = vanilla_fp32_bytes / max(1.0, total_bytes)

    return {
        "total_tokens": total_tokens,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "actual_sinks": actual_sinks,
        "actual_rw": actual_rw,
        "actual_hh": actual_hh,
        "actual_pbs": actual_pbs,
        "sink_bytes": total_sink_bytes,
        "rw_bytes": total_rw_bytes,
        "hh_bytes": total_hh_bytes,
        "pbs_payload_bytes": total_pbs_payload_bytes,
        "pbs_metadata_bytes": total_pbs_metadata_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024.0 * 1024.0),
        "vanilla_fp16_bytes": vanilla_fp16_bytes,
        "vanilla_fp16_mb": vanilla_fp16_bytes / (1024.0 * 1024.0),
        "vanilla_fp32_bytes": vanilla_fp32_bytes,
        "vanilla_fp32_mb": vanilla_fp32_bytes / (1024.0 * 1024.0),
        "compression_ratio_vs_fp16": compression_ratio_vs_fp16,
        "compression_ratio_vs_fp32": compression_ratio_vs_fp32,
    }


def get_evaluation_corpus(min_tokens: int = 2048, tokenizer=None) -> str:
    """Provides rich, non-repeating natural text corpus for perplexity and correctness benchmarks."""
    from benchmarks.corpus import get_large_corpus
    return get_large_corpus(min_tokens=min_tokens, tokenizer=tokenizer)
