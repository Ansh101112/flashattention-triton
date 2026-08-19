"""Naive PyTorch attention -- the correctness oracle and the memory baseline.

This is the O(N^2)-memory formulation the kernel replaces. Keeping it around
matters for two reasons: it defines "exact" for the numerical tests, and it is
the honest baseline for the memory plots (torch SDPA already dispatches to a
fused kernel, so comparing only against SDPA hides the actual saving).
"""

import math

import torch


def naive_attention(q, k, v, causal=False, sm_scale=None):
    """Materialises the full B x H x N x N score matrix. Reference only."""
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        N = q.shape[-2]
        mask = torch.triu(
            torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v.float()).to(q.dtype)


def torch_sdpa(q, k, v, causal=False, sm_scale=None):
    """PyTorch fused SDPA -- the performance baseline to beat (or match)."""
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale
    )


def attention_flops(batch, heads, seqlen, head_dim, causal=False, backward=False):
    """FLOPs for one attention op, used to convert latency into TFLOP/s.

    Forward is two matmuls (QK^T and PV), each 2*N*N*d MACs per head. Causal
    masking halves the useful work. Backward is conventionally counted at 2.5x
    forward (dQ, dK, dV plus the recomputed scores).
    """
    flops = 2 * 2 * batch * heads * seqlen * seqlen * head_dim
    if causal:
        flops *= 0.5
    if backward:
        flops *= 2.5
    return flops
