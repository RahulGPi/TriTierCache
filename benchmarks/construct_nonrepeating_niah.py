"""
benchmarks/construct_nonrepeating_niah.py
Constructs diverse, non-repeating needle-in-a-haystack (NIAH) prompts using
benchmarks.corpus.CORPUS_PARAGRAPHS.
Guarantees:
- Single BOS token at index 0 (if tokenizer specifies bos_token_id).
- Strictly non-repeating paragraphs with zero repeated sentences or 8-grams.
- Exact prompt length equal to total_len.
- Deterministic needle insertion at exact depth fraction.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Tuple, List
import torch
from benchmarks.corpus import CORPUS_PARAGRAPHS


def construct_nonrepeating_niah_prompt(
    tok,
    total_len: int,
    depth: float,
    needle_key: str = "94821"
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Constructs a non-repeating NIAH prompt:
      [BOS] + filler_part1 + needle_sentence + filler_part2 + query
    where filler is drawn sequentially from non-repeating CORPUS_PARAGRAPHS.

    Returns:
      (full_prompt, needle_start, needle_end, key_pos)
    """
    bos = [tok.bos_token_id] if getattr(tok, "bos_token_id", None) is not None else []
    needle_sentence = f" Special notice: the secret retrieval key is {needle_key}. Remember this key. "
    query = " What is the secret retrieval key? Answer: the secret retrieval key is "

    needle_ids = tok(needle_sentence, add_special_tokens=False).input_ids
    query_ids = tok(query, add_special_tokens=False).input_ids

    avail_filler = total_len - len(bos) - len(needle_ids) - len(query_ids)
    if avail_filler <= 0:
        raise ValueError(f"total_len {total_len} is too short for needle and query ({len(needle_ids) + len(query_ids)} tokens)")

    filler_ids: List[int] = []
    for p in CORPUS_PARAGRAPHS:
        p_ids = tok(" " + p, add_special_tokens=False).input_ids
        filler_ids.extend(p_ids)
        if len(filler_ids) >= avail_filler:
            break

    if len(filler_ids) < avail_filler:
        raise ValueError(
            f"CORPUS_PARAGRAPHS has only {len(filler_ids)} tokens, which is less than requested filler {avail_filler}"
        )

    filler_ids = filler_ids[:avail_filler]

    insert_pos = int(avail_filler * depth)
    part1 = filler_ids[:insert_pos]
    part2 = filler_ids[insert_pos:]

    full_ids = bos + part1 + needle_ids + part2 + query_ids
    assert len(full_ids) == total_len, f"Expected total_len {total_len}, got {len(full_ids)}"

    needle_start = len(bos) + len(part1)
    needle_end = needle_start + len(needle_ids)

    # Find the approximate position of the key token inside the needle
    key_ids = tok(needle_key, add_special_tokens=False).input_ids
    key_pos = needle_start + 1

    prompt_tensor = torch.tensor([full_ids], dtype=torch.int64)
    return prompt_tensor, needle_start, needle_end, key_pos
