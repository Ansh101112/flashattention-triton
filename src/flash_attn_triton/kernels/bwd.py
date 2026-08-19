"""FlashAttention backward pass, in Triton.

The forward pass avoided storing the N x N matrix. If backward just rebuilt and
stored it, we would have gained nothing -- so this is the part that actually
makes the memory saving real.

The trick: instead of storing P (the softmax probabilities), we RECOMPUTE each
tile of it from Q, K and the saved L vector, one tile at a time. That costs an
extra matmul but saves a factor of N in memory.

The gradients we need (S = scale * Q @ K^T, P = softmax(S), O = P @ V):

    dV = P^T @ dO
    dP = dO @ V^T
    dS = P * (dP - delta)      where delta_i = sum over d of dO[i,d] * O[i,d]
    dQ = scale * dS  @ K
    dK = scale * dS^T @ Q

About `delta`: differentiating softmax normally gives an N x N Jacobian per row,
which would be horrible. But when you contract it with dP the whole thing
collapses to one number per row, and that number happens to equal the row dot
product of dO and O. So a tiny O(N*d) pre-pass computes it and the expensive
kernels just read it. That's `_attn_bwd_preprocess`.

Why two kernels instead of one:
  dQ sums over key tiles, but dK and dV sum over query tiles. One fused kernel
  would need atomics on whichever one it didn't parallelise over. So:
    _attn_bwd_dkdv - one program per K/V tile, loops over Q tiles
    _attn_bwd_dq   - one program per Q tile, loops over K/V tiles
  This recomputes Q @ K^T twice, but on Ampere that's still faster than fp32
  atomics on dQ, which serialise badly.
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
    """Computes Delta[i] = sum over d of dO[i,d] * O[i,d], one value per row.

    Cheap O(N*d) pass. See the module docstring for why this one number replaces
    the whole softmax Jacobian.
    """
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

    # With a causal mask, query row m only looks at key column n <= m. So this
    # K tile is only used by Q tiles at or after it, and we can start the loop
    # there instead of at 0.
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

        # Rebuild this tile of P from scratch: P = exp(scores - L).
        # This is the recomputation that lets us skip storing P entirely.
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
