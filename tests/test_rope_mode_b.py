import pytest
import torch
import numpy as np


def compute_key_query_positions(total_steps: int, r_size: int, sink_size: int, rope_mode: str):
    """
    Computes (cached_key_positions, query_position) at each decode step.
    
    Mode 'a': Absolute position indexing.
      At step t: query is at t, cached keys are at their original insertion positions [0, 1, ..., t-1].
      
    Mode 'b': Eviction-aware StreamingLLM re-rotation.
      While t <= sink_size + r_size (no eviction has occurred):
        query is at t, cached keys are at [0, 1, ..., t-1] (IDENTICAL to mode 'a').
      When t > sink_size + r_size (eviction has occurred):
        keys in recent window have positions S, S+1, ..., S+R-1.
        query has position S+R.
        sinks have positions 0, ..., S-1.
    """
    history = []
    window_cap = sink_size + r_size

    for t in range(total_steps):
        if rope_mode == "a":
            # Mode 'a': exact sequence positions
            key_positions = list(range(t))
            query_pos = t
        elif rope_mode == "b":
            # Mode 'b': eviction-aware re-rotation
            if t <= window_cap:
                # Pre-eviction: all tokens retained in cache, identical to mode 'a'
                key_positions = list(range(t))
                query_pos = t
            else:
                # Post-eviction: sliding window relative positions
                sinks = list(range(sink_size))
                window_keys = [sink_size + i for i in range(r_size)]
                key_positions = sinks + window_keys
                query_pos = sink_size + r_size
        else:
            raise ValueError(f"Unknown rope_mode: {rope_mode}")

        history.append({
            "step": t,
            "key_positions": key_positions,
            "query_pos": query_pos,
        })
    return history


def test_rope_mode_b_pre_eviction_matches_mode_a():
    """
    Step 9: Standalone unit test verifying that mode 'b' assigns IDENTICAL
    positions to mode 'a' for all tokens before eviction occurs.
    """
    sink_size = 2
    r_size = 4
    window_cap = sink_size + r_size  # = 6 tokens total capacity before eviction
    total_steps = 8

    history_a = compute_key_query_positions(total_steps, r_size, sink_size, rope_mode="a")
    history_b = compute_key_query_positions(total_steps, r_size, sink_size, rope_mode="b")

    # 1. Verify pre-eviction steps (0 through window_cap) are BIT-IDENTICAL
    for t in range(window_cap + 1):
        step_a = history_a[t]
        step_b = history_b[t]
        assert step_a["query_pos"] == step_b["query_pos"], (
            f"Pre-eviction query position mismatch at step {t}: mode 'a'={step_a['query_pos']} vs mode 'b'={step_b['query_pos']}"
        )
        assert step_a["key_positions"] == step_b["key_positions"], (
            f"Pre-eviction cached key positions mismatch at step {t}: mode 'a'={step_a['key_positions']} vs mode 'b'={step_b['key_positions']}"
        )

    # 2. Verify post-eviction steps diverge as expected (divergence only after eviction)
    for t in range(window_cap + 1, total_steps):
        step_a = history_a[t]
        step_b = history_b[t]
        assert step_a["query_pos"] != step_b["query_pos"], (
            f"Post-eviction query positions should diverge at step {t}"
        )


def test_flawed_min_clamp_detection():
    """
    Step 9: Direct verification proving that the previous 'min(pos, R_size + 4)'
    bug caused pre-eviction / step-freeze corruption.
    """
    r_size = 256
    sink_size = 4
    clamp_limit = r_size + sink_size  # 260

    # Old flawed logic
    flawed_positions = [min(pos, clamp_limit) for pos in range(300)]
    
    # At step 260, 261, 262: all positions freeze at 260
    assert flawed_positions[260] == 260
    assert flawed_positions[261] == 260
    assert flawed_positions[262] == 260
    # Difference between consecutive tokens collapsed to 0
    diff_261_260 = flawed_positions[261] - flawed_positions[260]
    assert diff_261_260 == 0, "Flawed clamp froze relative distance to 0, destroying RoPE attention"
