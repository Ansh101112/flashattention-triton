"""FlashAttention-2 forward pass, written in Triton.

Papers: arXiv:2205.14135 (FlashAttention) and arXiv:2307.08691 (FlashAttention-2).

Normal attention computes the full N x N score matrix and writes it to GPU main
memory (HBM). At 8K context that matrix is 128 MB per head, and moving it back
and forth is what makes attention slow -- not the maths.

What this kernel does instead:

1. TILING
   Split Q into blocks of BLOCK_M rows and K/V into blocks of BLOCK_N rows.
   One program handles one Q tile and loops over all the K/V tiles. The Q tile
   and the running output stay in fast on-chip memory the whole time, so the
   N x N matrix is never written to HBM at all.

2. ONLINE SOFTMAX
   Problem: softmax needs the sum over the whole row, but we only ever see one
   tile of the row at a time.
   Fix: keep a running max (m_i) and running sum (l_i). When a new tile has a
   bigger max, rescale what we already accumulated by exp(m_old - m_new).
   This is exact algebra, not an approximation -- the answer is identical to
   normal attention.

3. NORMALISE AT THE END (this is the FlashAttention-2 part)
   Version 1 divided by l on every tile. Here we accumulate the un-divided
   output and divide once, after the loop. Saves a division per tile per row.

4. CAUSAL MASK SPLIT
   With a causal mask, most K tiles are either fully visible or fully hidden.
   Only the one tile sitting on the diagonal needs an actual mask. So we loop
   over the fully-visible tiles first and only apply tl.where on that last one.

At the end we save L = m + log(l) (one float per row, not N x N). The backward
pass uses it to rebuild the softmax values without having stored them.
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
    # K is loaded already transposed. We need K^T for Q @ K^T anyway, and this
    # layout is the one the tensor cores want, so we avoid a separate transpose.
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

    # The running softmax state. These live in registers for the whole loop.
    #   m_i = biggest score seen so far in each row
    #   l_i = running sum of exp(score - m_i)
    #   acc = running output, not yet divided by l_i
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # exp2 is a single hardware instruction, exp is not. Since
    # exp(x) == exp2(x * log2(e)), we bake log2(e) into the scale here once and
    # then use exp2 everywhere in the loop.
    qk_scale = sm_scale * 1.44269504089
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(q.dtype)

    if IS_CAUSAL:
        # A query at row m can only see keys at column n <= m, so we stop at
        # the diagonal instead of looping over the whole sequence.
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        # Everything before n_full is entirely below the diagonal, so it needs
        # no mask at all. Only tiles from n_full onwards do.
        n_full = (start_m * BLOCK_M) // BLOCK_N * BLOCK_N
    else:
        hi = N_CTX
        n_full = hi

    start_n = 0
    while start_n < hi:
        needs_mask = start_n >= n_full

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k, out_dtype=tl.float32)

        # If N isn't a multiple of BLOCK_N the last tile has junk columns.
        # Set them to -inf so they can't become the row max and can't add
        # anything to the sum.
        cols = start_n + offs_n
        qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))
        if IS_CAUSAL:
            if needs_mask:
                qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -float("inf"))

        # --- the online softmax update ---
        # m_ij is the new max after seeing this tile.
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # A fully masked row would leave m_ij at -inf and produce NaN below.
        m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        # exp of this tile's scores, relative to the new max.
        p = tl.math.exp2(qk - m_ij[:, None])
        # alpha corrects everything we accumulated under the OLD max.
        # If the max didn't change, alpha is 1 and nothing is rescaled.
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_ij

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        start_n += BLOCK_N

    # Now that the loop is done, divide once. This is the deferred
    # normalisation -- one divide per row instead of one per tile.
    # l_safe guards against dividing by zero on a fully masked row.
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]

    # Save L = m + log(l) for the backward pass. Converting back to natural log
    # here because that's what the backward kernels expect.
    lse = m_i * 0.69314718055994530942 + tl.log(l_safe)
    lse = tl.where(l_i == 0.0, -float("inf"), lse)
    tl.store(L + off_hz * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))
