#!/usr/bin/env python3
"""
example.py
End-to-end example comparing Vanilla Hugging Face Attention vs TriTierCache
on an actual causal language model (e.g. HuggingFaceTB/SmolLM-135M).

Measures and prints:
1. Generated output text
2. Prefill Latency / Time-To-First-Token (TTFT)
3. Average decode step latency (ms/token) & throughput (tokens/sec)
4. KV cache memory footprint & compression ratio
5. Tier allocation stats (Sinks, Recent Window, Heavy Hitters, PBS 2-bit blocks)
"""

import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from src.tri_tier.integration.patch_llama import apply_patch, remove_patch, reset_caches
from src.tri_tier.cache import TriTierCache


def load_model_and_tokenizer(model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Loads a pretrained model or falls back to synthetic LLaMA if unreachable."""
    try:
        print(f"Loading '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.to(device)
        model.eval()
        print(f"Successfully loaded '{model_id}' on {device}.\n")
        return model, tokenizer
    except Exception as e:
        print(f"Warning: Could not download '{model_id}' ({e}). Using synthetic LLaMA architecture for demo...")
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            max_position_embeddings=4096,
        )
        model = LlamaForCausalLM(config).to(dtype).to(device)
        model.eval()

        class SyntheticTokenizer:
            def __init__(self, vocab_size=32000):
                self.vocab_size = vocab_size
                self.eos_token_id = 2
            def __call__(self, text, return_tensors="pt"):
                words = text.split()
                ids = [hash(w) % (self.vocab_size - 10) + 10 for w in words] or [1]
                class Res:
                    input_ids = torch.tensor([ids], dtype=torch.int64)
                return Res()
            def decode(self, token_ids, skip_special_tokens=True):
                if isinstance(token_ids, torch.Tensor):
                    token_ids = token_ids.tolist()
                return " ".join([f"tok_{t}" for t in token_ids])

        return model, SyntheticTokenizer(config.vocab_size)


def run_vanilla(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int):
    """Runs generation with standard Hugging Face Attention and DynamicCache."""
    remove_patch()
    reset_caches(model)

    seq_len = input_ids.shape[1]
    curr_ids = input_ids.clone()
    latencies = []

    # 1. Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(curr_ids, use_cache=True)
        past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
    t1 = time.perf_counter()
    ttft_ms = (t1 - t0) * 1000.0

    # 2. Decode steps
    with torch.no_grad():
        for step in range(max_new_tokens):
            step_t0 = time.perf_counter()
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            out = model(next_id, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values if hasattr(out, "past_key_values") else None
            step_t1 = time.perf_counter()
            latencies.append((step_t1 - step_t0) * 1000.0)

    # Theoretical Vanilla KV cache memory footprint (FP16 bytes: 2 * layers * tokens * heads * dim * 2)
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    total_tokens = seq_len + max_new_tokens
    vanilla_fp16_bytes = 2 * num_layers * total_tokens * num_kv_heads * head_dim * 2
    vanilla_mb = vanilla_fp16_bytes / (1024.0 * 1024.0)

    generated_tokens = curr_ids[0, seq_len:].tolist()
    text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "tokens": generated_tokens,
        "ttft_ms": ttft_ms,
        "decode_latencies_ms": latencies,
        "avg_decode_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "throughput_tok_s": 1000.0 / (sum(latencies) / len(latencies)) if latencies else 0.0,
        "kv_memory_mb": vanilla_mb,
    }


def run_tritier(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int):
    """Runs generation with TriTierCache C++ fused kernel and hierarchical compression."""
    apply_patch()
    reset_caches(model)

    seq_len = input_ids.shape[1]
    curr_ids = input_ids.clone()
    latencies = []

    # 1. Batched Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(curr_ids)
    t1 = time.perf_counter()
    ttft_ms = (t1 - t0) * 1000.0

    # 2. Decode steps
    with torch.no_grad():
        for step in range(max_new_tokens):
            step_t0 = time.perf_counter()
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_id], dim=-1)
            pos = seq_len + step
            out = model(next_id, position_ids=torch.tensor([[pos]], dtype=torch.int64))
            step_t1 = time.perf_counter()
            latencies.append((step_t1 - step_t0) * 1000.0)

    # Calculate active KV cache footprint and total buffer memory
    from benchmarks.utils import calculate_tri_tier_cache_bytes
    total_tokens = seq_len + max_new_tokens
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    comp_info = calculate_tri_tier_cache_bytes(
        total_tokens=total_tokens,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sink_size=4,
        r_size=256,
        h_ratio=0.05
    )
    active_kv_mb = (comp_info["tri_tier_bytes"] * num_layers) / (1024.0 * 1024.0)
    vanilla_fp16_mb = (comp_info["vanilla_fp16_bytes"] * num_layers) / (1024.0 * 1024.0)

    total_tritier_bytes = 0
    s_count, rw_count, hh_count, pbs_count = 0, 0, 0, 0
    for module in model.modules():
        if hasattr(module, "tri_tier_cache") and module.tri_tier_cache is not None:
            c = module.tri_tier_cache
            total_tritier_bytes += c.get_buffer_bytes()
            s_count = getattr(c, "S_count", 0)
            rw_count = getattr(c, "RW_count", 0)
            hh_count = getattr(c, "HH_count", 0)
            pbs_count = getattr(c, "PBS_count", 0)

    allocated_mb = total_tritier_bytes / (1024.0 * 1024.0)
    generated_tokens = curr_ids[0, seq_len:].tolist()
    text = tokenizer.decode(curr_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "tokens": generated_tokens,
        "ttft_ms": ttft_ms,
        "decode_latencies_ms": latencies,
        "avg_decode_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "throughput_tok_s": 1000.0 / (sum(latencies) / len(latencies)) if latencies else 0.0,
        "kv_memory_mb": active_kv_mb,
        "allocated_buffer_mb": allocated_mb,
        "compression_ratio_vs_fp16": comp_info["compression_ratio_vs_fp16"],
        "tier_stats": {
            "sinks": s_count,
            "recent_window": rw_count,
            "heavy_hitters": hh_count,
            "pbs_tokens": pbs_count,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="TriTierCache vs Vanilla Comparison Demo")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M", help="Model name or path")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Target prompt length in tokens")
    parser.add_argument("--new-tokens", type=int, default=32, help="Number of new tokens to generate")
    parser.add_argument("--warmup", action="store_true", default=True, help="Run a brief warmup to prime CPU cache & thread pool")
    args = parser.parse_args()

    print("=" * 85)
    print("           TRITIERCACHE VS VANILLA LLAMA ATTENTION COMPARISON DEMO")
    print("=" * 85)

    model, tokenizer = load_model_and_tokenizer(args.model)

    # Build prompt
    if args.prompt is not None:
        prompt_text = args.prompt
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    else:
        sample_context = (
            "Modern deep autoregressive language models parameterize joint token distributions "
            "using multi-layer transformer blocks equipped with scaled dot-product attention. "
            "Key-value caching reduces auto-regressive decoding compute requirements from quadratic "
            "to linear time by preserving prior key and value projection tensors. "
            "Hierarchical streaming and quantized KV caches compress past context activations "
            "to mitigate high-bandwidth memory exhaustion during long-sequence generation tasks. "
        )
        tokens = tokenizer(sample_context, return_tensors="pt").input_ids
        while tokens.shape[1] < args.prompt_tokens:
            tokens = torch.cat([tokens, tokens], dim=1)
        input_ids = tokens[:, :args.prompt_tokens]
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    actual_prompt_len = input_ids.shape[1]
    print(f"Prompt Length   : {actual_prompt_len} tokens")
    print(f"Generate Length : {args.new_tokens} tokens")
    print("-" * 85)

    # Warmup
    if args.warmup:
        print("Running brief warmup (5 tokens) to prime CPU cache & OpenMP threads...")
        warmup_ids = input_ids[:, :min(64, actual_prompt_len)]
        _ = run_vanilla(model, tokenizer, warmup_ids, 5)
        _ = run_tritier(model, tokenizer, warmup_ids, 5)

    # 1. Run Vanilla Baseline
    print("Running Vanilla Hugging Face Attention...")
    v_res = run_vanilla(model, tokenizer, input_ids, args.new_tokens)

    # 2. Run TriTierCache
    print("Running TriTierCache (C++ AVX2 Fused Engine + Tiering)...")
    t_res = run_tritier(model, tokenizer, input_ids, args.new_tokens)

    # Calculate token matches
    matches = sum(1 for v, t in zip(v_res["tokens"], t_res["tokens"]) if v == t)
    agreement_pct = (matches / max(1, len(v_res["tokens"]))) * 100.0
    comp_ratio = v_res["kv_memory_mb"] / max(1e-4, t_res["kv_memory_mb"])

    # 3. Print Comparison Report
    print("\n" + "=" * 85)
    print("                              BENCHMARK & STATS REPORT")
    print("=" * 85)
    print(f"{'Metric':<38} | {'Vanilla Baseline':<20} | {'TriTierCache':<20}")
    print("-" * 85)
    print(f"{'Time-to-First-Token (Prefill TTFT)':<38} | {v_res['ttft_ms']:<17.2f} ms | {t_res['ttft_ms']:<17.2f} ms")
    print(f"{'Avg Decode Latency':<38} | {v_res['avg_decode_ms']:<17.2f} ms | {t_res['avg_decode_ms']:<17.2f} ms")
    print(f"{'Decode Throughput':<38} | {v_res['throughput_tok_s']:<14.2f} tok/s | {t_res['throughput_tok_s']:<14.2f} tok/s")
    print(f"{'Active KV Footprint (FP16 Eqv)':<38} | {v_res['kv_memory_mb']:<17.2f} MB | {t_res['kv_memory_mb']:<17.2f} MB")
    print(f"{'KV Compression Ratio (vs FP16)':<38} | {'1.00x (Baseline)':<20} | {comp_ratio:<17.2f} x")
    print(f"{'Top-1 Greedy Token Agreement':<38} | {'-':<20} | {agreement_pct:<17.2f} %")
    print("-" * 85)

    tier_stats = t_res.get("tier_stats", {})
    print("TriTier Storage Hierarchy Breakdown (Tokens in Cache):")
    print(f"  • Attention Sinks   : {tier_stats.get('sinks', 0):>4d} tokens (FP32 exact)")
    print(f"  • Recent Window     : {tier_stats.get('recent_window', 0):>4d} tokens (FP32 exact ring buffer)")
    print(f"  • Heavy Hitters     : {tier_stats.get('heavy_hitters', 0):>4d} tokens (FP32 exact dynamic)")
    print(f"  • 2-Bit Background  : {tier_stats.get('pbs_tokens', 0):>4d} tokens (2-bit packed blocks)")
    print("=" * 85)

    print("\n--- Vanilla Generated Output ---")
    print(tokenizer.decode(v_res["tokens"], skip_special_tokens=True))

    print("\n--- TriTierCache Generated Output ---")
    print(tokenizer.decode(t_res["tokens"], skip_special_tokens=True))
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
