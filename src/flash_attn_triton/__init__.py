"""FlashAttention in Triton -- IO-aware exact attention.

Public API:
    flash_attn(q, k, v, causal=False, sm_scale=None) -> Tensor
    FlashSelfAttention(embed_dim, num_heads, causal=True)
    naive_attention / torch_sdpa  (reference implementations)
"""

from .interface import flash_attn, FlashSelfAttention, SUPPORTED_HEAD_DIMS
from .reference import naive_attention, torch_sdpa, attention_flops

__version__ = "0.1.0"
__all__ = [
    "flash_attn",
    "FlashSelfAttention",
    "naive_attention",
    "torch_sdpa",
    "attention_flops",
    "SUPPORTED_HEAD_DIMS",
]
