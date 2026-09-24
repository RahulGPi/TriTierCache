#!/usr/bin/env python3
"""
benchmarks/benchmark_accuracy.py
Downstream Long-Context Task Accuracy Benchmark for TriTierCache.

Evaluates task accuracy, Exact Match (EM), Token F1, and Top-1 token agreement
comparing Vanilla Hugging Face Attention (lossless FP32) vs TriTierCache (AVX2 fused)
under context lengths >= 1024 up to 32k tokens where Tier 2 (Heavy Hitters) and
Tier 3 (2-bit PBS) are actively exercised.

Supports:
- Universal model families: LLaMA 3/3.2, SmolLM/SmolLM2, TinyLlama, Qwen 2.5/3, Mistral 7B
- Context scaling up to 32,768 tokens (32k)
- Presets: --tier 1, --tier 2, --tier 3, or --models <list>
- Export to benchmarks/results/accuracy_results.csv
- Automated update of root README.md accuracy table (--update-readme)
"""

import os
import re
import sys
import csv
import time
import math
import random
import argparse
import string
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from benchmarks.common import (
    load_model,
    set_seed,
    save_results_to_csv,
    measure_cache_bytes,
    DEFAULT_MODEL_ID,
)
from benchmarks.utils import generate_step_by_step
from benchmarks.corpus import CORPUS_PARAGRAPHS
from src.tri_tier.integration.patch_model import apply_patch, remove_patch, reset_caches


# ---------------------------------------------------------------------------
# Model Presets
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "tier1": [
        "HuggingFaceTB/SmolLM-135M",
    ],
    "tier2": [
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-1.7B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
    ],
    "tier3": [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-0.6B",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
    "all": [
        "HuggingFaceTB/SmolLM-135M",
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-1.7B",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen3-0.6B",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ],
}


# ---------------------------------------------------------------------------
# Evaluation Metrics Helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2.0 * (precision * recall) / (precision + recall)


def check_is_correct(prediction: str, ground_truth: str) -> bool:
    """Returns True if ground truth is contained in prediction or token F1 >= 0.5."""
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)
    if not norm_truth:
        return False
    if norm_truth in norm_pred:
        return True
    return compute_f1(prediction, ground_truth) >= 0.5


# ---------------------------------------------------------------------------
# Long-Context Task Generators (Scaling to 32k tokens)
# ---------------------------------------------------------------------------

QA_FACTS = [
    ("Project Zephyr was initialized in the year 2049 by Chief Scientist Dr. Evelyn Vance.",
     "Question: In what year was Project Zephyr initialized?\nAnswer:", "2049"),
    ("The primary coolant utilized across the reactor core is liquid neon-22.",
     "Question: What is the primary coolant utilized across the reactor core?\nAnswer:", "liquid neon-22"),
    ("The orbital transmission frequency for satellite Epsilon is 1420.45 MHz.",
     "Question: What is the transmission frequency for satellite Epsilon in MHz?\nAnswer:", "1420.45"),
    ("The designated emergency rendezvous station is Sector 7G on Outpost Bravo.",
     "Question: What is the designated emergency rendezvous station?\nAnswer:", "Sector 7G"),
    ("The atmospheric pressure within Habitat Dome 3 is maintained at 101.3 kilopascals.",
     "Question: What is the atmospheric pressure in Habitat Dome 3 in kilopascals?\nAnswer:", "101.3"),
    ("The backup optical encryption protocol is titled CipherX-88.",
     "Question: What is the name of the backup optical encryption protocol?\nAnswer:", "CipherX-88"),
    ("Deep-sea research vessel Poseidon reached a maximum depth of 10928 meters.",
     "Question: What maximum depth in meters was reached by vessel Poseidon?\nAnswer:", "10928"),
    ("The magnetic confinement coil operates at a target temperature of 4.2 Kelvin.",
     "Question: At what target temperature in Kelvin does the magnetic confinement coil operate?\nAnswer:", "4.2"),
    ("The quantum processor array is composed of 1024 superconducting qubits.",
     "Question: How many superconducting qubits compose the quantum processor array?\nAnswer:", "1024"),
    ("The secondary propulsion system is powered by dual Hall-effect ion thrusters.",
     "Question: What powers the secondary propulsion system?\nAnswer:", "dual Hall-effect ion thrusters"),
]

VARIABLE_TEMPLATES = [
    ("DATABASE_PORT", "5432"),
    ("CACHE_REDIS_PORT", "6379"),
    ("WEB_GATEWAY_PORT", "8080"),
    ("METRICS_COLLECTOR_PORT", "9090"),
    ("BACKUP_SFTP_PORT", "2222"),
    ("RPC_SERVICE_PORT", "50051"),
    ("DNS_RESOLVER_PORT", "5353"),
    ("KAFKA_BROKER_PORT", "9092"),
    ("STORAGE_CLUSTER_PORT", "7000"),
    ("AUTH_GATEWAY_PORT", "8443"),
]

ICL_DEMOS = [
    ("Alpha -> Apple", "Alpha", "Apple"),
    ("Beta -> Banana", "Beta", "Banana"),
    ("Gamma -> Grape", "Gamma", "Grape"),
    ("Delta -> Date", "Delta", "Date"),
    ("Epsilon -> Elderberry", "Epsilon", "Elderberry"),
    ("Zeta -> Zucchini", "Zeta", "Zucchini"),
    ("Eta -> Eggplant", "Eta", "Eggplant"),
    ("Theta -> Tangerine", "Theta", "Tangerine"),
    ("Iota -> Kiwi", "Iota", "Kiwi"),
    ("Kappa -> Kumquat", "Kappa", "Kumquat"),
    ("Lambda -> Lemon", "Lambda", "Lemon"),
    ("Mu -> Mango", "Mu", "Mango"),
    ("Nu -> Nectarine", "Nu", "Nectarine"),
    ("Xi -> Xigua", "Xi", "Xigua"),
    ("Omicron -> Orange", "Omicron", "Orange"),
    ("Pi -> Peach", "Pi", "Peach"),
    ("Rho -> Raspberry", "Rho", "Raspberry"),
    ("Sigma -> Strawberry", "Sigma", "Strawberry"),
    ("Tau -> Tomato", "Tau", "Tomato"),
    ("Upsilon -> Ugli", "Upsilon", "Ugli"),
    ("Phi -> Fig", "Phi", "Fig"),
    ("Chi -> Cherry", "Chi", "Cherry"),
    ("Psi -> Papaya", "Psi", "Papaya"),
    ("Omega -> Olive", "Omega", "Olive"),
    ("Falcon -> Aviation", "Falcon", "Aviation"),
    ("Cobalt -> Mineral", "Cobalt", "Mineral"),
    ("Argon -> NobleGas", "Argon", "NobleGas"),
    ("Orion -> Constellation", "Orion", "Constellation"),
    ("Helios -> Solar", "Helios", "Solar"),
    ("Chronos -> Temporal", "Chronos", "Temporal"),
    ("Valkyrie -> Defense", "Valkyrie", "Defense"),
    ("Nautilus -> Submarine", "Nautilus", "Submarine"),
    ("Apex -> Peak", "Apex", "Peak"),
    ("Nexus -> Connection", "Nexus", "Connection"),
    ("Vortex -> Swirl", "Vortex", "Swirl"),
    ("Zenith -> Pinnacle", "Zenith", "Pinnacle"),
    ("Cipher -> Cryptography", "Cipher", "Cryptography"),
    ("Prism -> Refraction", "Prism", "Refraction"),
    ("Solstice -> Astronomy", "Solstice", "Astronomy"),
    ("Eclipse -> Shadow", "Eclipse", "Shadow"),
]


def assemble_context(tokenizer,
                     target_token_len: int,
                     fact_text: str,
                     query_text: str,
                     depth_ratio: float = 0.5,
                     reserve_gen_tokens: int = 16) -> Tuple[str, int]:
    """
    Pads background text around the fact such that total prompt tokens <= target_token_len - reserve_gen_tokens,
    guaranteeing prompt + generation tokens stay within target_token_len (preventing RoPE boundary collapse).
    The fact is positioned approximately at depth_ratio. Supports scaling up to 32k tokens.
    """
    prompt_budget = max(64, target_token_len - reserve_gen_tokens)
    fixed_text = fact_text + "\n\n" + query_text
    fixed_tokens = len(tokenizer(fixed_text).input_ids)
    bg_budget = max(0, prompt_budget - fixed_tokens)
    prefix_budget = int(bg_budget * depth_ratio)
    suffix_budget = bg_budget - prefix_budget

    num_paras = len(CORPUS_PARAGRAPHS)
    corpus_idx = 0

    prefix_paras = []
    cur_pre = 0
    while cur_pre < prefix_budget and corpus_idx < num_paras:
        p = CORPUS_PARAGRAPHS[corpus_idx % num_paras]
        p_len = len(tokenizer(p).input_ids)
        if cur_pre + p_len > prefix_budget and prefix_paras:
            break
        prefix_paras.append(p)
        cur_pre += p_len
        corpus_idx += 1

    suffix_paras = []
    cur_suf = 0
    while cur_suf < suffix_budget and corpus_idx < num_paras * 2:
        p = CORPUS_PARAGRAPHS[corpus_idx % num_paras]
        p_len = len(tokenizer(p).input_ids)
        if cur_suf + p_len > suffix_budget and suffix_paras:
            break
        suffix_paras.append(p)
        cur_suf += p_len
        corpus_idx += 1

    parts = []
    if prefix_paras:
        parts.append("\n\n".join(prefix_paras))
    parts.append(fact_text)
    if suffix_paras:
        parts.append("\n\n".join(suffix_paras))
    parts.append(query_text)
    prompt_text = "\n\n".join(parts)

    actual_len = len(tokenizer(prompt_text).input_ids)
    # If joiner tokens cause it to slightly overshoot prompt_budget, trim prefix paragraphs
    while actual_len > prompt_budget and prefix_paras:
        prefix_paras.pop(0)
        parts = []
        if prefix_paras:
            parts.append("\n\n".join(prefix_paras))
        parts.append(fact_text)
        if suffix_paras:
            parts.append("\n\n".join(suffix_paras))
        parts.append(query_text)
        prompt_text = "\n\n".join(parts)
        actual_len = len(tokenizer(prompt_text).input_ids)

    return prompt_text, actual_len


def generate_task_samples(task: str,
                          tokenizer,
                          target_len: int,
                          num_samples: int) -> List[Dict[str, Any]]:
    """Generates a list of test instances for the given task and context length."""
    samples = []

    if task == "qa":
        depths = [0.15, 0.35, 0.50, 0.70, 0.85]
        for i in range(num_samples):
            fact_stmt, query, ground_truth = QA_FACTS[i % len(QA_FACTS)]
            depth = depths[i % len(depths)]
            prompt, actual_len = assemble_context(
                tokenizer, target_len, fact_stmt, query, depth_ratio=depth, reserve_gen_tokens=16
            )
            samples.append({
                "task": "Long-Context QA",
                "sample_id": i + 1,
                "prompt": prompt,
                "prompt_tokens": actual_len,
                "ground_truth": ground_truth,
                "max_gen_tokens": 12,
            })

    elif task == "multi_variable":
        for i in range(num_samples):
            target_var, target_val = VARIABLE_TEMPLATES[i % len(VARIABLE_TEMPLATES)]
            all_defs = [f"CONFIG_{k} = {v};" for k, v in VARIABLE_TEMPLATES]
            random.seed(42 + i)
            random.shuffle(all_defs)
            fact_block = "System Network Configuration Parameters:\n" + "\n".join(all_defs)
            query = f"Question: What is the assigned value of CONFIG_{target_var}?\nAnswer: CONFIG_{target_var} ="
            prompt, actual_len = assemble_context(
                tokenizer, target_len, fact_block, query, depth_ratio=0.5, reserve_gen_tokens=16
            )
            samples.append({
                "task": "Multi-Variable Tracking",
                "sample_id": i + 1,
                "prompt": prompt,
                "prompt_tokens": actual_len,
                "ground_truth": target_val,
                "max_gen_tokens": 8,
            })

    elif task == "many_shot_icl":
        reserve_gen = 8
        for i in range(num_samples):
            target_demo = ICL_DEMOS[i % len(ICL_DEMOS)]
            target_key, target_val = target_demo[1], target_demo[2]

            # Randomize order of demonstrations, ensuring target pair is included in the bank
            random.seed(42 + i)
            shuffled_demos = list(ICL_DEMOS)
            random.shuffle(shuffled_demos)

            demo_lines = ["Demonstrations:"]
            for d in shuffled_demos:
                demo_lines.append(f"Item: {d[1]} -> Output: {d[2]}")
            demo_block = "\n".join(demo_lines)
            query = f"Item: {target_key} -> Output:"

            # Measure tokens needed for demo block + query
            demo_query_text = demo_block + "\n\n" + query
            demo_tokens = len(tokenizer(demo_query_text).input_ids)
            needed_bg_tokens = max(0, target_len - reserve_gen - demo_tokens)

            # Pad background text BEFORE the demonstrations so demonstrations directly precede the query
            bg_paras = []
            cur_bg_tokens = 0
            c_idx = (i * 7) % len(CORPUS_PARAGRAPHS)
            while cur_bg_tokens < needed_bg_tokens and c_idx < len(CORPUS_PARAGRAPHS):
                p = CORPUS_PARAGRAPHS[c_idx % len(CORPUS_PARAGRAPHS)]
                p_len = len(tokenizer(p).input_ids)
                if cur_bg_tokens + p_len > needed_bg_tokens:
                    break
                bg_paras.append(p)
                cur_bg_tokens += p_len
                c_idx += 1

            if bg_paras:
                prompt = "\n\n".join(bg_paras) + "\n\n" + demo_query_text
            else:
                prompt = demo_query_text

            actual_len = len(tokenizer(prompt).input_ids)
            while actual_len > (target_len - reserve_gen) and bg_paras:
                bg_paras.pop(0)
                prompt = ("\n\n".join(bg_paras) + "\n\n" + demo_query_text) if bg_paras else demo_query_text
                actual_len = len(tokenizer(prompt).input_ids)

            samples.append({
                "task": "Many-Shot ICL",
                "sample_id": i + 1,
                "prompt": prompt,
                "prompt_tokens": actual_len,
                "ground_truth": target_val,
                "max_gen_tokens": 4,
            })

    return samples


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_accuracy_benchmark(model_name: str,
                           context_lens: List[int],
                           tasks: List[str],
                           samples_per_task: int = 10,
                           seed: int = 42) -> List[Dict[str, Any]]:
    set_seed(seed)
    print("=" * 80)
    print(f" Starting Downstream Accuracy Benchmark: {model_name}")
    print(f" Context Lengths: {context_lens} | Samples Per Task: {samples_per_task}")
    print("=" * 80)

    try:
        model, tokenizer = load_model(model_name)
    except Exception as e:
        print(f"ERROR: Could not load model '{model_name}': {e}. Skipping...")
        return []

    # Detect model parameters and architecture
    config = model.config
    model_type = getattr(config, "model_type", "llama")
    num_params_m = round(sum(p.numel() for p in model.parameters()) / 1e6)

    results = []

    for ctx_len in context_lens:
        # Safety check for max_position_embeddings
        max_pos = getattr(config, "max_position_embeddings", 32768)
        if ctx_len > max_pos:
            print(f"Notice: Requested context {ctx_len} exceeds model max_position_embeddings ({max_pos}). Running up to {max_pos}...")
            ctx_len = max_pos

        for task_key in tasks:
            task_samples = generate_task_samples(task_key, tokenizer, ctx_len, samples_per_task)
            if not task_samples:
                continue

            task_title = task_samples[0]["task"]
            print(f"\n--- Running Model: '{model_name}' | Task: '{task_title}' @ Context ~{ctx_len} tokens ({len(task_samples)} samples) ---")

            vanilla_correct = 0
            tritier_correct = 0
            token_matches = []
            vanilla_lats = []
            tritier_lats = []

            for idx, s in enumerate(task_samples):
                prompt = s["prompt"]
                truth = s["ground_truth"]
                max_gen = s["max_gen_tokens"]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids

                # 1. Vanilla Hugging Face Attention (FP32)
                remove_patch()
                reset_caches(model)
                v_ids, v_lats = generate_step_by_step(model, input_ids, max_new_tokens=max_gen)
                v_gen_ids = v_ids[0, input_ids.shape[1]:].tolist()
                v_text = tokenizer.decode(v_gen_ids, skip_special_tokens=True).strip()
                v_ok = check_is_correct(v_text, truth)
                if v_ok:
                    vanilla_correct += 1
                if v_lats:
                    vanilla_lats.append(sum(v_lats) / len(v_lats) * 1000.0)

                # 2. TriTierCache Attention (AVX2 Fused Kernel)
                apply_patch()
                reset_caches(model)
                t_ids, t_lats = generate_step_by_step(model, input_ids, max_new_tokens=max_gen)
                t_gen_ids = t_ids[0, input_ids.shape[1]:].tolist()
                t_text = tokenizer.decode(t_gen_ids, skip_special_tokens=True).strip()
                t_ok = check_is_correct(t_text, truth)
                if t_ok:
                    tritier_correct += 1
                if t_lats:
                    tritier_lats.append(sum(t_lats) / len(t_lats) * 1000.0)

                # Token agreement
                matches = sum(1 for v, t in zip(v_gen_ids, t_gen_ids) if v == t)
                match_pct = (matches / max_gen) * 100.0 if max_gen > 0 else 100.0
                token_matches.append(match_pct)

                print(f"  [{idx+1}/{len(task_samples)}] Truth: '{truth}' | Vanilla: '{v_text}' ({'PASS' if v_ok else 'FAIL'}) | TriTier: '{t_text}' ({'PASS' if t_ok else 'FAIL'}) | Match: {match_pct:.0f}%")

            # Summary for this configuration
            n_samples = len(task_samples)
            v_acc = (vanilla_correct / n_samples) * 100.0
            t_acc = (tritier_correct / n_samples) * 100.0
            retention = (t_acc / v_acc * 100.0) if v_acc > 0 else (100.0 if t_acc == 0 else 0.0)
            avg_match = sum(token_matches) / len(token_matches) if token_matches else 0.0
            v_lat = sum(vanilla_lats) / len(vanilla_lats) if vanilla_lats else 0.0
            t_lat = sum(tritier_lats) / len(tritier_lats) if tritier_lats else 0.0
            speedup = (v_lat / t_lat) if t_lat > 0 else 1.0

            # Exact KV cache memory accounting difference
            mem_info = measure_cache_bytes(
                total_tokens=ctx_len,
                config=model.config,
                sink_size=4,
                r_size=256,
                h_ratio=0.05,
                k_group_size=16,
                pbs_metadata_dtype="fp16",
            )
            v_mem_mb = mem_info["vanilla_fp32_mb"]
            t_mem_mb = mem_info["total_mb"]
            mem_ratio = mem_info["compression_ratio_vs_fp32"]
            mem_saved_mb = v_mem_mb - t_mem_mb

            row = {
                "model": model_name,
                "model_arch": model_type,
                "params_m": num_params_m,
                "context_length": ctx_len,
                "task": task_title,
                "samples": n_samples,
                "vanilla_acc": v_acc,
                "tritier_acc": t_acc,
                "retention_pct": retention,
                "token_match_pct": avg_match,
                "vanilla_kv_mb": round(v_mem_mb, 2),
                "tritier_kv_mb": round(t_mem_mb, 2),
                "mem_saved_mb": round(mem_saved_mb, 2),
                "mem_compression_ratio": round(mem_ratio, 2),
                "vanilla_ms_tok": v_lat,
                "tritier_ms_tok": t_lat,
                "speedup": speedup,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(row)

            print(f"==> RESULT: {model_name} | {task_title} @ {ctx_len} ctx: Vanilla={v_acc:.1f}% | TriTier={t_acc:.1f}% | Retention={retention:.1f}% | Memory={t_mem_mb:.1f}MB vs {v_mem_mb:.1f}MB ({mem_ratio:.1f}x) | Speedup={speedup:.2f}x")

    return results


# ---------------------------------------------------------------------------
# README Automation
# ---------------------------------------------------------------------------

def get_readme_path(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)
    if os.path.exists("README.md"):
        return os.path.abspath("README.md")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(script_dir), "README.md")
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath("README.md")


def update_readme_table(rows: List[Dict[str, Any]], readme_path: Optional[str] = None) -> None:
    """Inserts or updates the Downstream Long-Context Task Accuracy table in README.md."""
    resolved_path = get_readme_path(readme_path)
    if not os.path.exists(resolved_path):
        print(f"Warning: README.md not found at '{resolved_path}'. Skipping README update.")
        return

    with open(resolved_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_tag = "<!-- DOWNSTREAM_ACCURACY_START -->"
    end_tag = "<!-- DOWNSTREAM_ACCURACY_END -->"
    section_header = "### Downstream Long-Context Task Accuracy"

    # Read existing CSV rows if present to combine models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "results", "accuracy_results.csv")
    all_rows = list(rows)
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            key_map = {}
            for r in existing_rows:
                k = (r.get("model"), str(r.get("context_length")), r.get("task"))
                key_map[k] = r
            for r in rows:
                k = (r.get("model"), str(r.get("context_length")), r.get("task"))
                key_map[k] = r
            all_rows = list(key_map.values())

    STANDARD_TASK_ORDER = [
        "Long-Context QA",
        "Multi-Variable Tracking",
        "Many-Shot ICL",
    ]
    STANDARD_TASK_ABBR = {
        "Long-Context QA": "QA",
        "Multi-Variable Tracking": "Tracking",
        "Many-Shot ICL": "ICL",
    }

    # Group rows by model then context_length
    by_model: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for r in all_rows:
        model_key = str(r["model"])
        try:
            ctx = int(r["context_length"])
        except (ValueError, TypeError):
            continue
        task = str(r["task"])
        if model_key not in by_model:
            by_model[model_key] = {}
        if ctx not in by_model[model_key]:
            by_model[model_key][ctx] = {}
        by_model[model_key][ctx][task] = r

    table_lines = [
        section_header,
        "",
        "Evaluated on long-context tasks (context $\\ge 1024$ tokens) where $>75\\%$ of KV tokens reside in Tier 3 (2-bit PBS). Baseline is uncompressed FP32 Vanilla Hugging Face Attention.",
        "",
    ]

    for model_id, ctx_map in by_model.items():
        model_name = model_id.split("/")[-1]
        first_row = next(iter(next(iter(ctx_map.values())).values()))
        arch = str(first_row.get("model_arch", "llama")).upper()
        params = first_row.get("params_m", "")
        param_str = f", {params}M" if params else ""

        model_tasks_set = {t for ctx_dict in ctx_map.values() for t in ctx_dict.keys()}
        tasks = [t for t in STANDARD_TASK_ORDER if t in model_tasks_set]
        for t in sorted(model_tasks_set):
            if t not in tasks:
                tasks.append(t)

        table_lines.append(f"#### `{model_name}` ({arch} arch{param_str})")
        table_lines.append("")

        headers = ["Context"]
        for t in tasks:
            short_t = STANDARD_TASK_ABBR.get(t, t)
            headers.append(f"{short_t} (TriTier / Base)")
        headers.extend(["Retention", "Token Match", "KV Memory (TriTier / Base)", "Decode Speed"])

        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("| " + " | ".join([":---:"] * len(headers)) + " |")

        for ctx in sorted(ctx_map.keys()):
            task_data = ctx_map[ctx]
            row_cells = [f"**{ctx:,}**"]

            eval_tasks = []
            for t in tasks:
                r = task_data.get(t)
                if r:
                    eval_tasks.append(r)
                    t_acc = float(r["tritier_acc"])
                    v_acc = float(r["vanilla_acc"])
                    row_cells.append(f"**{t_acc:.1f}%** / {v_acc:.1f}%")
                else:
                    row_cells.append("-")

            if eval_tasks:
                ret = sum(float(r["retention_pct"]) for r in eval_tasks) / len(eval_tasks)
                match = sum(float(r["token_match_pct"]) for r in eval_tasks) / len(eval_tasks)
                speed = sum(float(r.get("speedup", 1.0)) for r in eval_tasks) / len(eval_tasks)
                lat = sum(float(r.get("tritier_ms_tok", 0.0)) for r in eval_tasks) / len(eval_tasks)

                # KV Memory calculation
                t_mem = None
                v_mem = None
                mem_ratio = None
                for r in eval_tasks:
                    if r.get("tritier_kv_mb") and r.get("vanilla_kv_mb"):
                        t_mem = float(r["tritier_kv_mb"])
                        v_mem = float(r["vanilla_kv_mb"])
                        mem_ratio = float(r.get("mem_compression_ratio", v_mem / max(0.001, t_mem)))
                        break
                if t_mem is None:
                    calc = measure_cache_bytes(
                        total_tokens=ctx,
                        num_layers=30 if "135" in model_name else 16,
                        num_kv_heads=3 if "135" in model_name else 4,
                        head_dim=64,
                    )
                    t_mem = calc["total_mb"]
                    v_mem = calc["vanilla_fp32_mb"]
                    mem_ratio = calc["compression_ratio_vs_fp32"]

                mem_cell = f"**{t_mem:.1f} MB** / {v_mem:.1f} MB ({mem_ratio:.1f}x)"

                row_cells.append(f"**{ret:.1f}%**")
                row_cells.append(f"{match:.1f}%")
                row_cells.append(mem_cell)
                row_cells.append(f"{speed:.2f}x ({lat:.1f} ms)")
            else:
                row_cells.extend(["-", "-", "-", "-"])

            table_lines.append("| " + " | ".join(row_cells) + " |")

        table_lines.append("")

    table_block = f"{start_tag}\n" + "\n".join(table_lines).rstrip() + f"\n{end_tag}"

    if start_tag in content and end_tag in content:
        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag, start_idx) + len(end_tag)
        content = content[:start_idx] + table_block + content[end_idx:]
    elif section_header in content:
        start_idx = content.find(section_header)
        boundaries = [content.find("\n## ", start_idx), content.find("\n---", start_idx)]
        valid_boundaries = [b for b in boundaries if b != -1]
        end_idx = min(valid_boundaries) if valid_boundaries else len(content)
        content = content[:start_idx] + table_block + "\n\n" + content[end_idx:]
    elif "### Core Benchmark Summary" in content:
        marker = "### Core Benchmark Summary"
        marker_idx = content.find(marker)
        next_sep = content.find("\n---", marker_idx)
        if next_sep != -1:
            content = content[:next_sep] + "\n---\n\n" + table_block + content[next_sep:]
        else:
            content = content[:marker_idx] + table_block + "\n\n" + content[marker_idx:]
    else:
        content += "\n\n" + table_block

    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[README Updated] -> Downstream accuracy table updated in {resolved_path}")


def save_and_merge_results_to_csv(filepath: str, new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges new benchmark rows into existing CSV without overwriting previous models."""
    if not new_rows:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        return []

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    all_rows_map = {}
    fieldnames = []
    seen_fields = set()

    # Read existing rows if present
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for fn in reader.fieldnames:
                        if fn not in seen_fields:
                            seen_fields.add(fn)
                            fieldnames.append(fn)
                for r in reader:
                    k = (str(r.get("model")), str(r.get("context_length")), str(r.get("task")))
                    all_rows_map[k] = r
        except Exception as e:
            print(f"Warning reading existing CSV {filepath}: {e}")

    # Merge new rows (updating or adding)
    for r in new_rows:
        for k in r.keys():
            if k not in seen_fields:
                seen_fields.add(k)
                fieldnames.append(k)
        key = (str(r.get("model")), str(r.get("context_length")), str(r.get("task")))
        all_rows_map[key] = r

    merged_rows = list(all_rows_map.values())

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"[CSV Saved & Merged] -> {filepath} ({len(new_rows)} new/updated, {len(merged_rows)} total rows across models)")
    return merged_rows


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TriTierCache Universal Downstream Long-Context Accuracy Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs or preset name ('tier1', 'tier2', 'tier3', 'all')")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                        help="Run predefined tier preset (1: SmolLM-135M, 2: Scaled LLaMA, 3: Qwen/Mistral/TinyLlama)")
    parser.add_argument("--context-lens", nargs="+", type=int, default=[1024, 2048],
                        help="Context lengths to benchmark up to 32k (default: 1024 2048)")
    parser.add_argument("--tasks", nargs="+", default=["qa", "multi_variable", "many_shot_icl"],
                        choices=["qa", "multi_variable", "many_shot_icl", "all"],
                        help="Tasks to evaluate (default: qa multi_variable many_shot_icl)")
    parser.add_argument("--samples-per-task", type=int, default=10,
                        help="Number of samples to run per task (default: 10)")
    parser.add_argument("--output-csv", type=str, default="benchmarks/results/accuracy_results.csv",
                        help="Path to save output results CSV")
    parser.add_argument("--update-readme", action="store_true",
                        help="Automatically update the root README.md accuracy table")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Resolve target models
    target_models = []
    if args.tier is not None:
        target_models = MODEL_PRESETS.get(f"tier{args.tier}", [])
    elif args.models:
        for m in args.models:
            if m.lower() in MODEL_PRESETS:
                target_models.extend(MODEL_PRESETS[m.lower()])
            else:
                target_models.append(m)
    else:
        target_models = [DEFAULT_MODEL_ID]

    # Deduplicate while preserving order
    seen = set()
    deduped_models = []
    for m in target_models:
        if m not in seen:
            seen.add(m)
            deduped_models.append(m)

    selected_tasks = ["qa", "multi_variable", "many_shot_icl"] if "all" in args.tasks else args.tasks

    print("Target models to evaluate:")
    for m in deduped_models:
        print(f" - {m}")

    all_results = []
    for model_id in deduped_models:
        model_results = run_accuracy_benchmark(
            model_name=model_id,
            context_lens=args.context_lens,
            tasks=selected_tasks,
            samples_per_task=args.samples_per_task,
            seed=args.seed,
        )
        all_results.extend(model_results)

    if all_results:
        merged_all = save_and_merge_results_to_csv(args.output_csv, all_results)

        if args.update_readme:
            update_readme_table(merged_all, readme_path="README.md")


if __name__ == "__main__":
    main()
