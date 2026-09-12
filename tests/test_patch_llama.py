"""
Comprehensive tests for tri_tier.integration.patch_llama module without modifying patch_llama.py.
"""
import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)
from tri_tier.integration.patch_llama import (
    apply_patch,
    patched_forward,
    MAX_SEQ_LEN,
    R_SIZE,
    H_RATIO,
)
from tri_tier.cache import TriTierCache
from tri_tier.constants import SINK_SIZE, CHUNK_SIZE


@pytest.fixture(autouse=True)
def ensure_patch_applied():
    """Apply the patch before running tests."""
    apply_patch()
    assert LlamaAttention.forward == patched_forward


def create_llama_attention(num_q_heads=4, num_kv_heads=4, hidden_size=64, head_dim=16):
    """Helper to create a configured LlamaAttention module and rotary embedding."""
    config = LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=4096,
        attention_bias=False,
    )
    layer = LlamaAttention(config, layer_idx=0)
    rotary_emb = LlamaRotaryEmbedding(config=config)
    return layer, rotary_emb, config


class TestPatchApplication:
    """Test that apply_patch successfully monkey-patches LlamaAttention."""

    def test_apply_patch_replaces_forward(self):
        apply_patch()
        assert LlamaAttention.forward == patched_forward


class TestInputAssertions:
    """Verify input validation assertions for batch size and sequence length."""

    def test_batch_size_greater_than_one_raises_assertion(self):
        layer, rotary_emb, config = create_llama_attention()
        hidden_states = torch.randn(2, 1, config.hidden_size)  # bsz = 2
        pos_ids = torch.tensor([[0], [0]])
        cos, sin = rotary_emb(hidden_states, pos_ids)

        with pytest.raises(AssertionError, match="TriTierCache only supports batch_size=1"):
            layer(hidden_states, position_embeddings=(cos, sin))

    def test_seq_len_greater_than_one_raises_assertion(self):
        layer, rotary_emb, config = create_llama_attention()
        hidden_states = torch.randn(1, 4, config.hidden_size)  # q_len = 4
        pos_ids = torch.tensor([[0, 1, 2, 3]])
        cos, sin = rotary_emb(hidden_states, pos_ids)

        with pytest.raises(AssertionError, match="TriTierCache only supports one new token per forward call"):
            layer(hidden_states, position_embeddings=(cos, sin))


class TestLazyCacheInitialization:
    """Verify that TriTierCache is lazily initialized on the layer."""

    def test_cache_created_on_first_forward_call(self):
        layer, rotary_emb, config = create_llama_attention(num_q_heads=4, num_kv_heads=2, hidden_size=64, head_dim=16)
        assert not hasattr(layer, "tri_tier_cache")

        hidden_states = torch.randn(1, 1, config.hidden_size)
        pos_ids = torch.tensor([[0]])
        cos, sin = rotary_emb(hidden_states, pos_ids)

        layer(hidden_states, position_embeddings=(cos, sin))

        assert hasattr(layer, "tri_tier_cache")
        cache = layer.tri_tier_cache
        assert isinstance(cache, TriTierCache)
        assert cache.num_heads == config.num_key_value_heads  # sized by KV heads
        assert cache.head_dim == 16
        assert cache.R_size == R_SIZE
        assert cache.Total_Processed_Tokens == 1


class TestStandardMHA:
    """Test Multi-Head Attention where num_q_heads == num_kv_heads."""

    def test_single_forward_step_output_shapes_and_values(self):
        layer, rotary_emb, config = create_llama_attention(num_q_heads=4, num_kv_heads=4, hidden_size=64, head_dim=16)
        hidden_states = torch.randn(1, 1, config.hidden_size)
        pos_ids = torch.tensor([[0]])
        cos, sin = rotary_emb(hidden_states, pos_ids)

        output, attn_weights = layer(hidden_states, position_embeddings=(cos, sin))

        assert output.shape == (1, 1, config.hidden_size)
        assert attn_weights.shape == (1, 4, 1, 1)
        assert not torch.isnan(output).any()
        assert not torch.isnan(attn_weights).any()
        assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(1, 4, 1), atol=1e-5)


class TestGQA:
    """Test Grouped Query Attention where num_q_heads > num_kv_heads."""

    def test_gqa_head_expansion_and_output_shapes(self):
        num_q_heads = 8
        num_kv_heads = 2
        hidden_size = 128
        head_dim = 16
        layer, rotary_emb, config = create_llama_attention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )

        hidden_states = torch.randn(1, 1, hidden_size)
        pos_ids = torch.tensor([[0]])
        cos, sin = rotary_emb(hidden_states, pos_ids)

        output, attn_weights = layer(hidden_states, position_embeddings=(cos, sin))

        assert output.shape == (1, 1, hidden_size)
        assert attn_weights.shape == (1, num_q_heads, 1, 1)
        assert not torch.isnan(output).any()


class TestMultiStepExecutionAndTiering:
    """Test multi-step decode loop across all tiers (Sinks, RW, HH, PBS 2-bit quantization)."""

    def test_multi_step_decode_across_eviction_and_quantization(self):
        num_q_heads = 4
        num_kv_heads = 4
        hidden_size = 64
        head_dim = 16
        layer, rotary_emb, config = create_llama_attention(
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )

        num_steps = SINK_SIZE + R_SIZE + CHUNK_SIZE + 10  # 4 + 256 + 16 + 10 = 286 steps

        for step in range(num_steps):
            hidden_states = torch.randn(1, 1, hidden_size)
            pos_ids = torch.tensor([[step]])
            cos, sin = rotary_emb(hidden_states, pos_ids)

            output, attn_weights = layer(hidden_states, position_embeddings=(cos, sin))

            assert output.shape == (1, 1, hidden_size)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

            cache = layer.tri_tier_cache
            expected_active_kv = (step + 1) - cache.WR_count
            assert attn_weights.shape == (1, num_q_heads, 1, expected_active_kv)

        cache = layer.tri_tier_cache
        assert cache.Total_Processed_Tokens == num_steps
        assert cache.S_count == SINK_SIZE
        assert cache.RW_count == R_SIZE
        assert cache.PBS_count >= CHUNK_SIZE or cache.HH_count > 0


class TestFullModelEndToEnd:
    """Test complete LlamaForCausalLM model with patched attention."""

    def test_end_to_end_causal_lm_generate(self):
        config = LlamaConfig(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=256,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = LlamaForCausalLM(config)
        model.eval()

        input_ids = torch.tensor([[1]])  # single token prompt for auto-regressive decoding

        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=15, use_cache=True)

        assert output.shape == (1, 16)

        # Verify each layer has its own distinct TriTierCache instance
        layer0_cache = model.model.layers[0].self_attn.tri_tier_cache
        layer1_cache = model.model.layers[1].self_attn.tri_tier_cache

        assert layer0_cache is not layer1_cache
        assert layer0_cache.Total_Processed_Tokens == 15
        assert layer1_cache.Total_Processed_Tokens == 15
        assert layer0_cache.Global_Attn_Scr[:15].sum() > 0
