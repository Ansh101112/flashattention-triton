"""Public PyTorch interface: a differentiable `flash_attn` op and an nn.Module.

The autograd.Function saves only Q, K, V, O and the log-sum-exp vector L. The
N x N score matrix is never stored, which is where the memory win comes from:
activation memory for attention drops from O(B * H * N^2) to O(B * H * N * d).
"""

import math

import torch
import triton

from .kernels.fwd import _attn_fwd
from .kernels.bwd import _attn_bwd_preprocess, _attn_bwd_dkdv, _attn_bwd_dq

SUPPORTED_HEAD_DIMS = (16, 32, 64, 128)


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, sm_scale=None):
        # Shapes: (batch, heads, seqlen, head_dim)
        if q.dim() != 4:
            raise ValueError(f"expected 4D (B, H, N, D) tensors, got {q.dim()}D")
        if not (q.shape == k.shape == v.shape):
            raise ValueError("q, k, v must have identical shapes")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(f"flash_attn supports fp16/bf16, got {q.dtype}")

        B, H, N, D = q.shape
        if D not in SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"head_dim {D} unsupported; compiled for {SUPPORTED_HEAD_DIMS}"
            )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        q, k, v = (x.contiguous() for x in (q, k, v))
        o = torch.empty_like(q)
        L = torch.empty((B * H, N), device=q.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H, 1)
        _attn_fwd[grid](
            q, k, v, sm_scale, L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N,
            BLOCK_DMODEL=D,
            IS_CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)

        BLOCK = 64
        pre_grid = (triton.cdiv(N, BLOCK), B * H)
        _attn_bwd_preprocess[pre_grid](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            H, N,
            BLOCK_M=BLOCK, BLOCK_DMODEL=D,
        )

        _attn_bwd_dkdv[(triton.cdiv(N, BLOCK), B * H)](
            q, k, v, ctx.sm_scale, do, dk, dv, L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            H, N,
            BLOCK_M=BLOCK, BLOCK_DMODEL=D, BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.causal,
            num_warps=4, num_stages=2,
        )

        _attn_bwd_dq[(triton.cdiv(N, BLOCK), B * H)](
            q, k, v, ctx.sm_scale, do, dq, L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            H, N,
            BLOCK_M=BLOCK, BLOCK_DMODEL=D, BLOCK_N=BLOCK,
            IS_CAUSAL=ctx.causal,
            num_warps=4, num_stages=2,
        )

        return dq, dk, dv, None, None


def flash_attn(q, k, v, causal=False, sm_scale=None):
    """Exact scaled dot-product attention, IO-aware.

    Args:
        q, k, v: (B, H, N, D) fp16/bf16 CUDA tensors.
        causal:  apply the lower-triangular mask.
        sm_scale: softmax temperature; defaults to 1/sqrt(D).

    Returns:
        (B, H, N, D) tensor, same dtype as the inputs.
    """
    return _FlashAttention.apply(q, k, v, causal, sm_scale)


class FlashSelfAttention(torch.nn.Module):
    """Drop-in multi-head self-attention using the Triton kernel.

    Takes and returns (B, N, H*D) so it can replace a standard attention block
    without touching the surrounding model code.
    """

    def __init__(self, embed_dim, num_heads, causal=True, bias=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = flash_attn(q, k, v, causal=self.causal)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))
