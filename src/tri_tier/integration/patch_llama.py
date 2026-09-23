"""
src/tri_tier/integration/patch_llama.py
LLaMA attention patcher for TriTierCache.
Re-exports universal multi-model patcher from patch_model.py for backwards compatibility.
"""

from .patch_model import (
    apply_patch,
    remove_patch,
    reset_caches,
    is_patched,
    patched_forward,
    _reference_attention_path,
    MAX_SEQ_LEN,
    R_SIZE,
    H_RATIO,
    SCORE_DECAY,
    K_GROUP_SIZE,
    PBS_METADATA_DTYPE,
    HAS_CPP_EXT,
)

__all__ = [
    "apply_patch",
    "remove_patch",
    "reset_caches",
    "is_patched",
    "patched_forward",
    "_reference_attention_path",
    "MAX_SEQ_LEN",
    "R_SIZE",
    "H_RATIO",
    "SCORE_DECAY",
    "K_GROUP_SIZE",
    "PBS_METADATA_DTYPE",
    "HAS_CPP_EXT",
]
