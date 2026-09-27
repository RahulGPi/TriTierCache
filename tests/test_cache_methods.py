"""
Unit tests for TriTierCache's core methods:
accumulate_attn_scrs, ingest_token, fetch_or_calc_thresh, route_evicted_token.

Run with: pytest test_cache_methods.py -v

compress_and_store is still a TODO in the source, so every test that
exercises route_evicted_token stubs it out with a MagicMock. This lets us
verify the routing/branching logic now, independent of whether the actual
2-bit compression exists yet. Once compress_and_store is implemented, add
a second file that tests what actually lands in Tier 3 -- these tests only
check that the right data was *handed to* compress_and_store, not what it
does with it.
"""
import math
import torch
import pytest
from unittest.mock import MagicMock

from tri_tier.cache import TriTierCache
from tri_tier.constants import UPDATE_THRESHOLD, CHUNK_SIZE, SINK_SIZE

MAX_SEQ_LEN = 128
HEAD_DIM = 16
NUM_HEADS = 4
R_SIZE = 8
H_RATIO = 0.5  # generous HH capacity so route_evicted_token tests can set it up by hand


@pytest.fixture
def cache():
    return TriTierCache(MAX_SEQ_LEN, HEAD_DIM, NUM_HEADS, R_SIZE, H_RATIO)


# ---------------------------------------------------------------------------
# accumulate_attn_scrs
# ---------------------------------------------------------------------------

class TestAccumulateAttnScrs:
    def test_single_token_decode_step_updates_correct_slice(self, cache):
        kv_len = 10
        attn = torch.rand(1, NUM_HEADS, 1, kv_len)  # [batch, heads, q_len=1, kv_len]
        cache.accumulate_attn_scrs(attn)

        expected = attn.squeeze(0).squeeze(1).mean(dim=0)
        assert torch.allclose(cache.Global_Attn_Scr[:kv_len], expected)
        assert torch.all(cache.Global_Attn_Scr[kv_len:] == 0)

    def test_accumulates_additively_not_overwriting(self, cache):
        attn = torch.rand(1, NUM_HEADS, 1, 5)
        cache.accumulate_attn_scrs(attn)
        first = cache.Global_Attn_Scr[:5].clone()
        cache.accumulate_attn_scrs(attn)
        assert torch.allclose(cache.Global_Attn_Scr[:5], first * 2)

    def test_multi_token_prefill_currently_unsupported(self, cache):
        # KNOWN LIMITATION, not a design choice: with q_len > 1 the mean
        # collapses to shape [q_len, kv_len] but Global_Attn_Scr is 1D, so
        # the in-place add_ cannot broadcast. This test documents that the
        # method only works one decode step at a time right now. If prefill
        # or speculative decoding is ever routed through this method, this
        # test should start failing -- that's your signal to replace it
        # with real multi-token support, not to delete it.
        attn = torch.rand(1, NUM_HEADS, 3, 10)  # q_len=3
        with pytest.raises(RuntimeError):
            cache.accumulate_attn_scrs(attn)


# ---------------------------------------------------------------------------
# fetch_or_calc_thresh
# ---------------------------------------------------------------------------

class TestFetchOrCalcThresh:
    def test_first_call_recalculates_regardless_of_id(self, cache):
        cache.Total_Processed_Tokens = 50
        cache.Global_Attn_Scr[:50] = torch.linspace(0, 1, 50)
        expected = torch.quantile(cache.Global_Attn_Scr[:50], 0.95)

        # id=1 is not on the UPDATE_THRESHOLD interval, but current_threshold
        # starts at 0.0, so this must still trigger a recalculation.
        result = cache.fetch_or_calc_thresh(1)
        assert torch.isclose(result, expected)

    def test_caches_between_recalculation_points(self, cache):
        cache.current_threshold = torch.tensor(-999.0)
        result = cache.fetch_or_calc_thresh(1)  # not divisible by UPDATE_THRESHOLD
        assert result.item() == -999.0

    def test_recalculates_on_update_interval(self, cache):
        cache.Total_Processed_Tokens = 50
        cache.Global_Attn_Scr[:50] = torch.linspace(0, 1, 50)
        cache.current_threshold = torch.tensor(-999.0)
        expected = torch.quantile(cache.Global_Attn_Scr[:50], 0.95)

        result = cache.fetch_or_calc_thresh(UPDATE_THRESHOLD)
        assert torch.isclose(result, expected)


# ---------------------------------------------------------------------------
# ingest_token
# ---------------------------------------------------------------------------

class TestIngestToken:
    def test_fills_ring_buffer_before_evicting(self, cache, monkeypatch):
        route_mock = MagicMock()
        monkeypatch.setattr(cache, "route_evicted_token", route_mock)

        # First SINK_SIZE tokens fill the attention sink buffer, then R_SIZE fill recent window
        for i in range(SINK_SIZE + R_SIZE):
            k = torch.full((NUM_HEADS, HEAD_DIM), float(i))
            v = torch.full((NUM_HEADS, HEAD_DIM), float(i) * 10)
            cache.ingest_token(k, v)

        assert cache.S_count == SINK_SIZE
        assert cache.RW_count == R_SIZE
        assert cache.RW_head_index == 0
        assert cache.Total_Processed_Tokens == SINK_SIZE + R_SIZE
        route_mock.assert_not_called()
        assert torch.all(cache.RW_K_Buffer[3] == float(SINK_SIZE + 3))

    def test_eviction_overwrites_oldest_slot_and_advances_head(self, cache, monkeypatch):
        route_mock = MagicMock()
        monkeypatch.setattr(cache, "route_evicted_token", route_mock)

        for i in range(SINK_SIZE + R_SIZE):
            cache.ingest_token(
                torch.full((NUM_HEADS, HEAD_DIM), float(i)),
                torch.full((NUM_HEADS, HEAD_DIM), float(i)),
            )

        new_k = torch.full((NUM_HEADS, HEAD_DIM), 999.0)
        new_v = torch.full((NUM_HEADS, HEAD_DIM), 888.0)
        cache.ingest_token(new_k, new_v)

        route_mock.assert_called_once()
        _, _, evicted_id = route_mock.call_args[0]
        assert evicted_id == SINK_SIZE  # first non-sink token ingested is the first evicted
        assert torch.all(cache.RW_K_Buffer[0] == 999.0)  # slot 0 overwritten
        assert cache.RW_head_index == 1
        assert cache.Total_Processed_Tokens == SINK_SIZE + R_SIZE + 1

    def test_evicted_token_id_formula_stays_correct_across_wraps(self, cache, monkeypatch):
        route_mock = MagicMock()
        monkeypatch.setattr(cache, "route_evicted_token", route_mock)

        # fill sinks + RW, then evict R_SIZE more times to wrap the ring buffer fully
        for i in range(SINK_SIZE + R_SIZE + R_SIZE):
            cache.ingest_token(
                torch.zeros(NUM_HEADS, HEAD_DIM), torch.zeros(NUM_HEADS, HEAD_DIM)
            )

        evicted_ids = [call.args[2] for call in route_mock.call_args_list]
        assert evicted_ids == list(range(SINK_SIZE, SINK_SIZE + R_SIZE))


# ---------------------------------------------------------------------------
# route_evicted_token
# ---------------------------------------------------------------------------

class TestRouteEvictedToken:
    def test_writes_directly_when_room_available(self, cache, monkeypatch):
        compress_mock = MagicMock()
        monkeypatch.setattr(cache, "compress_and_store", compress_mock, raising=False)
        cache.current_threshold = 0.1
        cache.Global_Attn_Scr[5] = 0.9

        cache.route_evicted_token(
            torch.ones(NUM_HEADS, HEAD_DIM), torch.ones(NUM_HEADS, HEAD_DIM), 5
        )

        compress_mock.assert_not_called()
        assert cache.HH_count == 1
        assert cache.HH_token_ids[0].item() == 5

    def test_sends_to_background_when_below_threshold(self, cache, monkeypatch):
        compress_mock = MagicMock()
        monkeypatch.setattr(cache, "compress_and_store", compress_mock, raising=False)
        cache.current_threshold = 0.9
        cache.Global_Attn_Scr[7] = 0.1

        cache.route_evicted_token(
            torch.full((NUM_HEADS, HEAD_DIM), 3.0),
            torch.full((NUM_HEADS, HEAD_DIM), 4.0),
            7,
        )

        compress_mock.assert_called_once()
        assert cache.HH_count == 0
        assert compress_mock.call_args[0][2] == 7

    def test_new_token_loses_to_existing_heavy_hitters(self, cache, monkeypatch):
        compress_mock = MagicMock()
        monkeypatch.setattr(cache, "compress_and_store", compress_mock, raising=False)
        cache.max_heavy_hitters = 3
        cache.HH_K_Buffer = torch.zeros(3, NUM_HEADS, HEAD_DIM)
        cache.HH_V_Buffer = torch.zeros(3, NUM_HEADS, HEAD_DIM)
        cache.HH_scores = torch.tensor([0.5, 0.3, 0.8])
        cache.HH_token_ids = torch.tensor([10, 11, 12])
        cache.HH_count = 3
        cache.current_threshold = 0.1
        cache.Global_Attn_Scr[50] = 0.2  # clears the threshold but loses to the weakest HH (0.3)

        cache.route_evicted_token(
            torch.full((NUM_HEADS, HEAD_DIM), 9.0),
            torch.full((NUM_HEADS, HEAD_DIM), 9.0),
            50,
        )

        assert compress_mock.call_args[0][2] == 50  # the new token was compressed, not an existing HH
        assert torch.equal(cache.HH_token_ids, torch.tensor([10, 11, 12]))  # HH buffer untouched

    def test_demote_preserves_the_original_removed_token_not_the_new_one(self, cache, monkeypatch):
        # This is the important one. removed_k/removed_v/removed_id in the
        # source are read as VIEWS into the HH buffers (no .clone()), then
        # the same buffer slot is overwritten before compress_and_store
        # would ever consume them. Without .clone(), compress_and_store
        # receives the NEW token's data twice -- the demoted token's real
        # data is silently lost. Same bug class as the ring-buffer eviction
        # two files ago, just not fixed here yet.
        compress_mock = MagicMock()
        monkeypatch.setattr(cache, "compress_and_store", compress_mock, raising=False)
        cache.max_heavy_hitters = 3
        cache.HH_K_Buffer = torch.zeros(3, NUM_HEADS, HEAD_DIM)
        cache.HH_V_Buffer = torch.zeros(3, NUM_HEADS, HEAD_DIM)
        cache.HH_scores = torch.tensor([0.5, 0.2, 0.8])
        cache.HH_token_ids = torch.tensor([10, 11, 12])
        cache.HH_count = 3
        cache.current_threshold = 0.1
        cache.Global_Attn_Scr[99] = 0.6  # beats index 1 (score 0.2, id 11)

        new_k = torch.full((NUM_HEADS, HEAD_DIM), 7.0)
        new_v = torch.full((NUM_HEADS, HEAD_DIM), 8.0)
        cache.route_evicted_token(new_k, new_v, 99)

        removed_k, removed_v, removed_id = compress_mock.call_args[0]
        assert removed_id.item() == 11, "compress_and_store should receive the OLD demoted id (11)"
        assert torch.all(removed_k == 0), "compress_and_store should receive the OLD (zero) K values, not the new token's 7.0"
        assert torch.all(removed_v == 0), "compress_and_store should receive the OLD (zero) V values, not the new token's 8.0"

        # and the slot itself should end up holding the new token, same as before
        assert cache.HH_token_ids[1].item() == 99
        assert torch.all(cache.HH_K_Buffer[1] == 7.0)
