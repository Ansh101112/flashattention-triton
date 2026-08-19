"""FlashAttention-2 forward kernel in Triton.

Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness" (2022) and "FlashAttention-2" (2023).

Key ideas implemented here:
  * Tiling: Q is split into blocks of BLOCK_M rows; K/V into blocks of BLOCK_N.
    A Q tile and the running output accumulator stay in SRAM/registers for the
    whole inner loop, so the O(N^2) attention matrix is never materialised in HBM.
  * Online softmax: the running max `m_i` and running sum `l_i` are updated per
    K/V tile, and the accumulator is rescaled by exp(m_old - m_new). This gives
    the exact softmax in a single pass over K/V.
  * FlashAttention-2 deferred normalisation: we accumulate the *unnormalised*
    output and divide by `l_i` once, after the inner loop, instead of rescaling
    by 1/l on every tile. That removes one divide per inner iteration.
  * Causal masking is split into a "full" range of K tiles that need no mask and
    a short "diagonal" range that does, so the masked branch is only taken for
    the one tile that straddles the diagonal.

The log-sum-exp statistic L = m + log(l) is saved for the backward pass, which
lets backward recompute the softmax probabilities without storing the S matrix.
"""

import triton
import triton.language as tl


def _fwd_configs():
    configs = []
    for BM in (64, 128):
        for BN in (32, 64):
            for s in (2, 3, 4):
                for w in (4, 8):
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN},
                            num_stages=s,
                            num_warps=w,
                        )
                    )
    return configs


@triton.autotune(configs=_fwd_configs(), key=["N_CTX", "BLOCK_DMODEL", "IS_CAUSAL"])
@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = Out + off_z * stride_oz + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=q_base, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
    )
    # K is loaded transposed so the QK^T dot maps onto the tensor-core layout.
    K_block_ptr = tl.make_block_ptr(
        base=k_base, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn),
        offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_base, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk),
        offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=o_base, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Running softmax state, kept in registers for the whole inner loop.
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Fold log2(e) into the scale so we can use the faster exp2 intrinsic.
    qk_scale = sm_scale * 1.44269504089
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(q.dtype)

    if IS_CAUSAL:
        # Only K tiles up to and including the diagonal contribute.
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        # Tiles strictly below the diagonal need no mask.
        n_full = (start_m * BLOCK_M) // BLOCK_N * BLOCK_N
    else:
        hi = N_CTX
        n_full = hi

    start_n = 0
    while start_n < hi:
        needs_mask = start_n >= n_full

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k, out_dtype=tl.float32)

        # Out-of-range columns (ragged tail) must not win the max.
        cols = start_n + offs_n
        qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))
        if IS_CAUSAL:
            if needs_mask:
                qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -float("inf"))

        # --- online softmax update (FlashAttention-2 form) ---
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)          # rescale factor for old state
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        start_n += BLOCK_N

    # Deferred normalisation: one divide per query row, not per K tile.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    # log-sum-exp in natural log, consumed by the backward pass.
    lse = m_i * 0.69314718055994530942 + tl.log(l_safe)
    lse = tl.where(l_i == 0.0, -float("inf"), lse)
    tl.store(L + off_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
