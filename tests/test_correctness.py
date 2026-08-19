"""Tests that the kernel gives the same answer as plain PyTorch attention.

FlashAttention is supposed to be EXACT, not an approximation. So the only
difference allowed here is normal fp16/bf16 rounding, which the reference has
too. That's why the tolerances are dtype-level and not loose.

The sequence lengths 200 and 999 are deliberate: they aren't multiples of any
tile size, and the ragged last tile is where tiled kernels usually break.
"""

import math

import pytest
import torch

from flash_attn_triton import flash_attn, naive_attention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require CUDA"
)

DTYPES = [torch.float16, torch.bfloat16]
# 200 and 999 are deliberately not multiples of any tile size.
SEQLENS = [16, 128, 200, 512, 999, 1024]
HEAD_DIMS = [32, 64, 128]


def _tol(dtype):
    # bf16 carries 8 mantissa bits vs fp16's 11, so it needs a looser bound.
    return (2e-2, 3e-2) if dtype is torch.bfloat16 else (5e-3, 1e-2)


def _make_qkv(B, H, N, D, dtype, requires_grad=False):
    g = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(
            (B, H, N, D), device="cuda", dtype=dtype, generator=g
        ).requires_grad_(requires_grad)
        for _ in range(3)
    ]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", SEQLENS)
@pytest.mark.parametrize("D", HEAD_DIMS)
def test_forward_matches_reference(dtype, causal, N, D):
    B, H = 2, 4
    q, k, v = _make_qkv(B, H, N, D, dtype)
    out = flash_attn(q, k, v, causal=causal)
    ref = naive_attention(q, k, v, causal=causal)
    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [128, 200, 512])
def test_backward_matches_reference(dtype, causal, N):
    B, H, D = 2, 4, 64
    q, k, v = _make_qkv(B, H, N, D, dtype, requires_grad=True)
    rq, rk, rv = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    do = torch.randn_like(q)

    flash_attn(q, k, v, causal=causal).backward(do)
    naive_attention(rq, rk, rv, causal=causal).backward(do)

    atol, rtol = _tol(dtype)
    # dK/dV accumulate across every query row, so their error budget is larger.
    torch.testing.assert_close(q.grad, rq.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, rk.grad, atol=atol * 2, rtol=rtol * 2)
    torch.testing.assert_close(v.grad, rv.grad, atol=atol * 2, rtol=rtol * 2)


def test_causal_row_zero_attends_only_to_itself():
    """Regression guard for the diagonal tile: row 0 must ignore all of K[1:]."""
    B, H, N, D = 1, 1, 256, 64
    q, k, v = _make_qkv(B, H, N, D, torch.float16)
    out_full = flash_attn(q, k, v, causal=True)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, 1:] = 0
    v2[:, :, 1:] = 0
    out_trunc = flash_attn(q, k2, v2, causal=True)
    torch.testing.assert_close(out_full[:, :, 0], out_trunc[:, :, 0], atol=1e-3, rtol=1e-3)


def test_custom_softmax_scale():
    B, H, N, D = 2, 2, 256, 64
    q, k, v = _make_qkv(B, H, N, D, torch.float16)
    scale = 0.137
    torch.testing.assert_close(
        flash_attn(q, k, v, sm_scale=scale),
        naive_attention(q, k, v, sm_scale=scale),
        atol=5e-3, rtol=1e-2,
    )


def test_no_quadratic_memory():
    """The whole point: peak memory must not scale with N^2.

    Doubling the sequence length should roughly double -- not quadruple -- the
    memory the op adds on top of its inputs.
    """
    B, H, D = 1, 8, 64

    def peak_for(N):
        q, k, v = _make_qkv(B, H, N, D, torch.float16, requires_grad=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        flash_attn(q, k, v, causal=True).sum().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - base

    small = peak_for(1024)
    large = peak_for(2048)
    # Linear scaling would give ~2x; allow slack for autotuner scratch space.
    assert large < small * 3, f"memory grew {large / small:.1f}x for a 2x seqlen"


def test_rejects_fp32():
    q, k, v = _make_qkv(1, 1, 64, 64, torch.float32)
    with pytest.raises(TypeError):
        flash_attn(q, k, v)


def test_rejects_unsupported_head_dim():
    q, k, v = _make_qkv(1, 1, 64, 48, torch.float16)
    with pytest.raises(ValueError):
        flash_attn(q, k, v)
