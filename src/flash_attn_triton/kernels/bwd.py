"""FlashAttention backward kernels in Triton.

Backward is the part that makes the memory saving real: a naive implementation
would need the N x N probability matrix P again to form dQ/dK/dV. Instead we
*recompute* P tile-by-tile from Q, K and the saved log-sum-exp vector L, so the
backward pass keeps the same O(N) activation footprint as the forward.

Gradients (S = scale * Q K^T, P = softmax(S), O = P V):

    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - delta),   delta_i = sum_j P_ij dP_ij = sum_d dO_id O_id
    dQ = scale * dS  K
    dK = scale * dS^T Q

delta is the cheap trick here: the row-wise softmax Jacobian term collapses to a
single dot product of dO and O, which a small pre-pass computes in O(N*d).

Two separate kernels are used rather than one fused kernel:
  * `_attn_bwd_dkdv` parallelises over K/V tiles and loops over Q tiles.
  * `_attn_bwd_dq`   parallelises over Q tiles and loops over K/V tiles.
This avoids cross-block atomics on dQ at the cost of recomputing QK^T twice. On
Ampere that trade is favourable because fp32 atomics on dQ serialise badly.
"""

import triton
import triton.language as tl

LOG2E: tl.constexpr = 1.44269504089


@triton.jit
def _attn_bwd_preprocess(
    Out, DO, Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Delta[i] = sum_d dO[i, d] * O[i, d] -- the softmax-Jacobian row term."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    o_ptrs = (Out + off_z * stride_oz + off_h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    do_ptrs = (DO + off_z * stride_doz + off_h * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_don)

    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + offs_m, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dkdv(
    Q, K, V, sm_scale, DO, DK, DV, L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_n = offs_n < N_CTX

    k_ptrs = (K + off_z * stride_kz + off_h * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = (V + off_z * stride_vz + off_h * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # Under causal masking, query row m only attends to key col n <= m, so K tile
    # `start_n` is only touched by Q tiles at or after it.
    if IS_CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    start_m = lo
    while start_m < N_CTX:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX

        q_ptrs = (Q + off_z * stride_qz + off_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        do_ptrs = (DO + off_z * stride_doz + off_h * stride_doh
                   + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok)
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # Recompute the probability tile from Q, K and the saved LSE.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        valid = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)
        p = tl.math.exp2((qk - l_i[:, None]) * LOG2E)
        p = tl.where(valid, p, 0.0)

        # dV = P^T dO
        dv += tl.dot(tl.trans(p).to(do.dtype), do, out_dtype=tl.float32)

        # dS = P * (dP - delta);  dK = scale * dS^T Q
        delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32)

        start_m += BLOCK_M

    dk_ptrs = (DK + off_z * stride_kz + off_h * stride_kh
               + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    dv_ptrs = (DV + off_z * stride_vz + off_h * stride_vh
               + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def _attn_bwd_dq(
    Q, K, V, sm_scale, DO, DQ, L, Delta,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < N_CTX

    q_ptrs = (Q + off_z * stride_qz + off_h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    do_ptrs = (DO + off_z * stride_doz + off_h * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok)
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(L + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + off_hz * N_CTX + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    start_n = 0
    while start_n < hi:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        k_ptrs = (K + off_z * stride_kz + off_h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = (V + off_z * stride_vz + off_h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        valid = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.math.exp2((qk - l_i[:, None]) * LOG2E)
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k, out_dtype=tl.float32)

        start_n += BLOCK_N

    dq_ptrs = (DQ + off_z * stride_qz + off_h * stride_qh
               + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])
