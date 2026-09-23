"""
Common utility functions for TriTierCache benchmarking suite.
"""
import os
import gc
import csv
import time
import math
import psutil
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
from src.tri_tier.cache import TriTierCache


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM-135M"


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


def load_model_and_tokenizer(model_name_or_path: str = DEFAULT_MODEL_ID, 
                             device: str = "cpu",
                             dtype: torch.dtype = torch.float32,
                             use_synthetic_if_missing: bool = True) -> Tuple[Any, Any]:
    """
    Loads a pretrained causal LM and tokenizer, falling back to synthetic Llama model
    only if network / checkpoint is completely unreachable.
    """
    try:
        print(f"Loading model: '{model_name_or_path}' (dtype={dtype}, device={device})...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype)
        model.to(device)
        model.eval()
        print(f"Successfully loaded '{model_name_or_path}'")
        return model, tokenizer
    except Exception as e:
        if not use_synthetic_if_missing:
            raise e
        print(f"Warning: Failed to load '{model_name_or_path}' ({e}). Falling back to synthetic Llama architecture...")
        is_1b = "1b" in str(model_name_or_path).lower() or "llama-3" in str(model_name_or_path).lower()
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


def get_evaluation_corpus(min_tokens: int = 2048, tokenizer=None) -> str:
    """Provides rich, non-repeating natural text corpus for perplexity and correctness benchmarks."""
    from benchmarks.corpus import get_large_corpus
    return get_large_corpus(min_tokens=min_tokens, tokenizer=tokenizer)


def generate_step_by_step(model, input_ids: torch.Tensor, max_new_tokens: int = 20) -> Tuple[torch.Tensor, List[float]]:
    """
    Executes auto-regressive decoding token-by-token (batch_size=1) compatible with
    both TriTierCache and Vanilla HF Attention.
    Returns:
        (generated_ids, step_latencies_seconds)
    """
    bsz, seq_len = input_ids.shape
    assert bsz == 1, "Only batch_size=1 is supported"
    
    latencies = []
    curr_ids = input_ids.clone()
    past_kv = None
    
    from src.tri_tier.integration.patch_model import is_patched as check_is_patched
    is_patched = check_is_patched()
    
    with torch.no_grad():
        if not is_patched:
            # Vanilla Hugging Face: single batched GEMM prefill
            out = model(curr_ids, use_cache=True)
            past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
        else:
            # TriTier batched prefill (Phase 3)
            out = model(curr_ids)
            past_kv = None
            
        # Decode loop
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            pos = seq_len + step
            if not is_patched:
                out = model(next_id, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            else:
                out = model(next_id, position_ids=torch.tensor([[pos]], dtype=torch.int64))
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            
    return curr_ids, latencies


def get_model_cache_bytes(model, use_tritier: bool = False, total_tokens: int = 0) -> int:
    """Computes exact allocated buffer memory (in bytes) across all attention layers in model."""
    if use_tritier:
        total_bytes = 0
        for module in model.modules():
            if hasattr(module, "tri_tier_cache") and module.tri_tier_cache is not None:
                total_bytes += module.tri_tier_cache.get_buffer_bytes()
        return total_bytes
    else:
        # Theoretical Vanilla Cache calculation based on model config
        config = model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # K and V tensors
        return 2 * num_layers * total_tokens * num_kv_heads * head_dim * 4


def assert_all_tritier_caches_evicted(model) -> int:
    """Asserts that all TriTierCache instances on the model experienced non-zero evictions."""
    total_evictions = 0
    cache_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "tri_tier_cache") and module.tri_tier_cache is not None:
            cache = module.tri_tier_cache
            cache.assert_evictions_occurred()
            total_evictions += cache.total_evictions
            cache_count += 1
    assert cache_count > 0, "No TriTierCache instances found on model!"
    return total_evictions


def calculate_tri_tier_cache_bytes(total_tokens: int, num_kv_heads: int, head_dim: int,
                                   sink_size: int = 4, r_size: int = 256, h_ratio: float = 0.05,
                                   chunk_size: int = 16) -> Dict[str, float]:
    """
    Calculates exact memory footprint (in bytes) of TriTierCache vs Vanilla FP16 (Primary) & FP32 Cache.
    """
    fp32_bytes_per_elem = 4
    fp16_bytes_per_elem = 2
    
    # Vanilla Uncompressed KV cache bytes (K + V)
    vanilla_fp32_bytes = 2 * total_tokens * num_kv_heads * head_dim * fp32_bytes_per_elem
    vanilla_fp16_bytes = 2 * total_tokens * num_kv_heads * head_dim * fp16_bytes_per_elem
    
    # TriTier Cache:
    max_hh = math.ceil(total_tokens * h_ratio)
    actual_sinks = min(sink_size, total_tokens)
    actual_rw = min(r_size, max(0, total_tokens - actual_sinks))
    actual_hh = min(max_hh, max(0, total_tokens - actual_sinks - actual_rw))
    actual_pbs = max(0, total_tokens - actual_sinks - actual_rw - actual_hh)
    
    # Dense tiers (FP16 K + V per Phase 4)
    dense_tokens = actual_sinks + actual_rw + actual_hh
    dense_bytes = 2 * dense_tokens * num_kv_heads * head_dim * fp16_bytes_per_elem
    
    # PBS Tier (2-bit per element + FP16 metadata + token_ids)
    pbs_blocks = math.ceil(actual_pbs / chunk_size)
    pbs_k_bits_bytes = actual_pbs * num_kv_heads * head_dim * 0.25
    pbs_v_bits_bytes = actual_pbs * num_kv_heads * head_dim * 0.25
    pbs_k_metadata_bytes = pbs_blocks * num_kv_heads * head_dim * 4  # 2B scale + 2B zero per block (FP16)
    pbs_v_metadata_bytes = actual_pbs * num_kv_heads * 4            # 2B scale + 2B zero per token (FP16)
    pbs_id_bytes = actual_pbs * 8                                   # int64 token id
    
    tri_tier_total_bytes = (dense_bytes + pbs_k_bits_bytes + pbs_v_bits_bytes + 
                            pbs_k_metadata_bytes + pbs_v_metadata_bytes + pbs_id_bytes)
    
    ratio_vs_fp16 = vanilla_fp16_bytes / max(1.0, tri_tier_total_bytes)
    ratio_vs_fp32 = vanilla_fp32_bytes / max(1.0, tri_tier_total_bytes)
    
    return {
        "total_tokens": total_tokens,
        "vanilla_fp16_bytes": vanilla_fp16_bytes,
        "vanilla_fp32_bytes": vanilla_fp32_bytes,
        "tri_tier_bytes": tri_tier_total_bytes,
        "compression_ratio_vs_fp16": ratio_vs_fp16,
        "compression_ratio_vs_fp32": ratio_vs_fp32,
    }
