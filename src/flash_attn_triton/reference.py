"""Plain PyTorch attention. Used for two things.

1. As the correct answer to test against. It runs in fp32 and builds the full
   N x N matrix, so whatever it produces is what the kernel must match.

2. As the honest memory baseline. torch's own SDPA is already a fused kernel,
   so comparing only against SDPA would hide how much memory the tiling
   actually saves. This version shows the real before/after.
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
    """How many floating point ops one attention call does.

    Used to turn a millisecond number into TFLOP/s so results are comparable
    across GPUs.

    Forward is two matmuls (Q@K^T and P@V), each 2*N*N*d per head.
    Causal masking roughly halves the useful work.
    Backward is counted at 2.5x forward, which is the usual convention
    (dQ, dK, dV plus recomputing the scores).
    """
    flops = 2 * 2 * batch * heads * seqlen * seqlen * head_dim
    if causal:
        flops *= 0.5
    if backward:
        flops *= 2.5
    return flops
