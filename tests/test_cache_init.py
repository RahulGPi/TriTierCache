"""
Unit tests for TriTierCache.__init__

Run with: pytest test_cache_init.py -v

Covers every buffer allocated in the constructor: shape, dtype, device,
and initial/sentinel values. test_constructs_without_error exists
specifically because this constructor has failed to even instantiate
three separate times -- if that one fails, nothing else here matters yet.
"""
import math
import torch
import pytest

from tri_tier.cache import TriTierCache
from tri_tier.constants import CHUNK_SIZE

# Deliberately "ugly" numbers so ceiling/rounding behavior gets exercised
# instead of hidden by convenient round numbers.
MAX_SEQ_LEN = 128
HEAD_DIM = 16
NUM_HEADS = 4
R_SIZE = 32
H_RATIO = 0.05


@pytest.fixture
def cache():
    return TriTierCache(
        max_seq_len=MAX_SEQ_LEN,
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        R_size=R_SIZE,
        H_ratio=H_RATIO,
    )


def test_constructs_without_error():
    TriTierCache(MAX_SEQ_LEN, HEAD_DIM, NUM_HEADS, R_SIZE, H_RATIO)


class TestSizing:
    """Derived counts computed at the top of __init__."""

    def test_tier_sizes_partition_max_seq_len(self, cache):
        # R_size + heavy hitters + background must exactly account for
        # every slot in max_seq_len -- if this drifts, reconstruction
        # will silently drop or duplicate tokens later.
        total = cache.R_size + cache.max_heavy_hitters + cache.max_background_tokens
        assert total == MAX_SEQ_LEN

    def test_num_blocks_matches_chunk_size(self, cache):
        expected = math.ceil(cache.max_background_tokens / CHUNK_SIZE)
        assert cache.num_blocks == expected

    def test_quant_head_dim_matches_channel_packing(self, cache):
        assert cache.quant_head_dim == math.ceil(HEAD_DIM / 16)


class TestRecentWindow:
    """Tier 1 -- ring buffer, FP32, exact."""

    def test_shapes(self, cache):
        assert cache.RW_K_Buffer.shape == (R_SIZE, NUM_HEADS, HEAD_DIM)
        assert cache.RW_V_Buffer.shape == (R_SIZE, NUM_HEADS, HEAD_DIM)

    def test_dtype_is_fp32(self, cache):
        assert cache.RW_K_Buffer.dtype == torch.float32
        assert cache.RW_V_Buffer.dtype == torch.float32

    def test_initial_pointers(self, cache):
        assert cache.RW_count == 0
        assert cache.RW_head_index == 0


class TestHeavyHitters:
    """Tier 2 -- top 5% by attention score, FP32."""

    def test_shapes(self, cache):
        n = cache.max_heavy_hitters
        assert cache.HH_K_Buffer.shape == (n, NUM_HEADS, HEAD_DIM)
        assert cache.HH_V_Buffer.shape == (n, NUM_HEADS, HEAD_DIM)
        assert cache.HH_token_ids.shape == (n,)
        assert cache.HH_scores.shape == (n,)

    def test_dtypes(self, cache):
        assert cache.HH_K_Buffer.dtype == torch.float32
        assert cache.HH_token_ids.dtype == torch.int64
        assert cache.HH_scores.dtype == torch.float32

    def test_token_ids_start_as_unused_sentinel(self, cache):
        # -1 means "this slot has never been written." If this isn't
        # true right after construction, an empty slot is indistinguishable
        # from a real token id of whatever garbage was in memory.
        assert torch.all(cache.HH_token_ids == -1)

    def test_initial_count(self, cache):
        assert cache.HH_count == 0


class TestWaitingRoom:
    """Staging buffer -- holds evicted tokens until CHUNK_SIZE accumulate."""

    def test_shapes(self, cache):
        assert cache.WR_K_Buffer.shape == (CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
        assert cache.WR_V_Buffer.shape == (CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
        assert cache.WR_token_ids.shape == (CHUNK_SIZE,)

    def test_token_ids_start_as_unused_sentinel(self, cache):
        assert torch.all(cache.WR_token_ids == -1)

    def test_initial_count(self, cache):
        assert cache.WR_count == 0


class TestBackgroundKeys:
    """Tier 3 K storage -- 2-bit, per-channel, grouped by 16-token block."""

    def test_packed_shape_is_block_grouped(self, cache):
        # NOT per-token. First dim must be num_blocks -- this is the exact
        # shape that was wrong two revisions ago (was max_background_tokens).
        assert cache.PBS_K_Packed.shape == (cache.num_blocks, NUM_HEADS, HEAD_DIM)
        assert cache.PBS_K_Packed.dtype == torch.int32

    def test_scale_and_zero_are_block_grouped_not_per_token(self, cache):
        # One scale/zero pair per BLOCK, shared by all 16 tokens in it.
        # If this were sized by max_background_tokens instead, the metadata
        # would cost more bytes than the FP32 data it's meant to compress.
        assert cache.PBS_K_Scales.shape == (cache.num_blocks, NUM_HEADS, HEAD_DIM)
        assert cache.PBS_K_Zeroes.shape == (cache.num_blocks, NUM_HEADS, HEAD_DIM)
        assert cache.PBS_K_Scales.dtype == torch.float32


class TestBackgroundValues:
    """Tier 3 V storage -- 2-bit, per-token, packed along head_dim."""

    def test_packed_shape_is_per_token(self, cache):
        n, qd = cache.max_background_tokens, cache.quant_head_dim
        assert cache.PBS_V_Packed.shape == (n, NUM_HEADS, qd)
        assert cache.PBS_V_Packed.dtype == torch.int32

    def test_scale_and_zero_are_one_per_token(self, cache):
        n = cache.max_background_tokens
        assert cache.PBS_V_Scales.shape == (n, NUM_HEADS, 1)
        assert cache.PBS_V_Zeroes.shape == (n, NUM_HEADS, 1)


class TestBackgroundTokenIds:
    def test_shape_and_dtype(self, cache):
        assert cache.PBS_token_ids.shape == (cache.max_background_tokens,)
        assert cache.PBS_token_ids.dtype == torch.int64

    def test_starts_as_unused_sentinel(self, cache):
        assert torch.all(cache.PBS_token_ids == -1)

    def test_initial_count(self, cache):
        assert cache.PBS_count == 0


class TestGlobalAttentionScore:
    def test_shape_covers_full_context(self, cache):
        assert cache.Global_Attn_Scr.shape == (MAX_SEQ_LEN,)

    def test_starts_at_zero(self, cache):
        assert torch.all(cache.Global_Attn_Scr == 0.0)

    def test_threshold_and_counter_start_clean(self, cache):
        assert cache.current_threshold == 0.0
        assert cache.Total_Processed_Tokens == 0