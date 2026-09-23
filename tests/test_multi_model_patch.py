import pytest
import torch
from transformers import LlamaConfig, MistralConfig, Qwen2Config
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from src.tri_tier.integration.patch_model import (
    apply_patch,
    remove_patch,
    reset_caches,
    is_patched,
    patched_forward,
)


@pytest.fixture(autouse=True)
def clean_patch_state():
    """Ensure clean unpatched state before and after each test."""
    remove_patch()
    yield
    remove_patch()


def test_apply_and_remove_patch_all_architectures():
    """Verify that apply_patch replaces forward on Llama, Mistral, and Qwen2."""
    orig_llama = LlamaAttention.forward
    orig_mistral = MistralAttention.forward
    orig_qwen2 = Qwen2Attention.forward

    assert not is_patched()

    apply_patch()
    assert is_patched()
    assert LlamaAttention.forward == patched_forward
    assert MistralAttention.forward == patched_forward
    assert Qwen2Attention.forward == patched_forward

    remove_patch()
    assert not is_patched()
    assert LlamaAttention.forward == orig_llama
    assert MistralAttention.forward == orig_mistral
    assert Qwen2Attention.forward == orig_qwen2


def test_llama_patched_forward_shapes():
    """Verify LlamaAttention executes batched prefill and single-token decode under TriTierCache."""
    cfg = LlamaConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=4, head_dim=32)
    layer = LlamaAttention(cfg, layer_idx=0)

    apply_patch()

    # 1. Batched Prefill
    bsz, q_len = 1, 16
    h_states = torch.randn(bsz, q_len, cfg.hidden_size)
    cos = torch.randn(1, q_len, cfg.head_dim)
    sin = torch.randn(1, q_len, cfg.head_dim)
    pos_emb = (cos, sin)

    out, weights = layer(h_states, position_embeddings=pos_emb)
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert hasattr(layer, "tri_tier_cache")
    assert layer.tri_tier_cache.Total_Processed_Tokens == q_len

    # 2. Single token decode step
    h_tok = torch.randn(1, 1, cfg.hidden_size)
    cos_tok = torch.randn(1, 1, cfg.head_dim)
    sin_tok = torch.randn(1, 1, cfg.head_dim)
    out_dec, _ = layer(h_tok, position_embeddings=(cos_tok, sin_tok))
    assert out_dec.shape == (1, 1, cfg.hidden_size)
    assert layer.tri_tier_cache.Total_Processed_Tokens == q_len + 1


def test_mistral_patched_forward_shapes():
    """Verify MistralAttention executes under TriTierCache (GQA, head_dim=64)."""
    cfg = MistralConfig(hidden_size=256, num_attention_heads=8, num_key_value_heads=4, head_dim=64)
    layer = MistralAttention(cfg, layer_idx=0)

    apply_patch()

    # Batched Prefill
    bsz, q_len = 1, 32
    h_states = torch.randn(bsz, q_len, cfg.hidden_size)
    cos = torch.randn(1, q_len, cfg.head_dim)
    sin = torch.randn(1, q_len, cfg.head_dim)

    out, _ = layer(h_states, position_embeddings=(cos, sin))
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert hasattr(layer, "tri_tier_cache")

    # Single token decode step
    h_tok = torch.randn(1, 1, cfg.hidden_size)
    cos_tok = torch.randn(1, 1, cfg.head_dim)
    sin_tok = torch.randn(1, 1, cfg.head_dim)
    out_dec, _ = layer(h_tok, position_embeddings=(cos_tok, sin_tok))
    assert out_dec.shape == (1, 1, cfg.hidden_size)
    assert layer.tri_tier_cache.Total_Processed_Tokens == q_len + 1


def test_qwen2_patched_forward_shapes_head_dim_128():
    """Verify Qwen2Attention executes under TriTierCache with head_dim=128."""
    cfg = Qwen2Config(hidden_size=512, num_attention_heads=4, num_key_value_heads=2, head_dim=128)
    layer = Qwen2Attention(cfg, layer_idx=0)

    apply_patch()

    # Batched Prefill
    bsz, q_len = 1, 20
    h_states = torch.randn(bsz, q_len, cfg.hidden_size)
    cos = torch.randn(1, q_len, cfg.head_dim)
    sin = torch.randn(1, q_len, cfg.head_dim)

    out, _ = layer(h_states, position_embeddings=(cos, sin))
    assert out.shape == (bsz, q_len, cfg.hidden_size)
    assert layer.tri_tier_cache.head_dim == 128

    # Single token decode step
    h_tok = torch.randn(1, 1, cfg.hidden_size)
    cos_tok = torch.randn(1, 1, cfg.head_dim)
    sin_tok = torch.randn(1, 1, cfg.head_dim)
    out_dec, _ = layer(h_tok, position_embeddings=(cos_tok, sin_tok))
    assert out_dec.shape == (1, 1, cfg.hidden_size)
    assert layer.tri_tier_cache.Total_Processed_Tokens == q_len + 1
