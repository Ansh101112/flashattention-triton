# Derivation: why tiling attention is exact, and why it is faster

This is the reasoning the kernels implement. It is written out because the two
things people usually get wrong about FlashAttention are (a) assuming it is an
approximation and (b) assuming the speedup comes from doing fewer FLOPs.
Neither is true.

## 1. The problem is IO, not arithmetic

Standard attention computes

```
S = Q Kᵀ / √d        (N × N)
P = softmax(S)       (N × N)
O = P V              (N × d)
```

A GPU has a memory hierarchy with wildly different bandwidths. On an A100:

| Level        | Size        | Bandwidth   |
|--------------|-------------|-------------|
| SRAM (SMEM)  | ~192 KB/SM  | ~19 TB/s    |
| HBM          | 40–80 GB    | ~1.5–2 TB/s |

The naive implementation writes `S` to HBM, reads it back to softmax it, writes
`P`, reads it back to multiply by `V`. That is **Θ(N²)** HBM traffic on top of
the Θ(Nd) needed for the inputs. For N = 8192 the score matrix alone is 128 MiB
per head in fp16 — it cannot live in SRAM, so every element makes several
round trips over the slowest link in the system.

Counting HBM accesses instead of FLOPs:

| | HBM accesses |
|---|---|
| Standard attention | Θ(N² + N d) |
| FlashAttention | Θ(N² d² / M), M = SRAM size |

With d = 64 and M ≈ 100 KB, `d²/M` is roughly 1/25, so the tiled version moves
about an order of magnitude less data. The arithmetic is identical — in fact
FlashAttention does *more* arithmetic in the backward pass, because it
recomputes `P`. It still wins, because attention at these shapes is
memory-bound, not compute-bound.

## 2. Online softmax makes tiling exact

The obstacle to tiling is that softmax is not local: the denominator needs every
score in the row. The fix is a running-statistics formulation.

For numerical safety softmax is always computed with the max subtracted:

```
softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m),   m = max_j x_j
```

Split a row `x` into two blocks `x⁽¹⁾, x⁽²⁾`. Let

```
m⁽¹⁾ = max(x⁽¹⁾)                l⁽¹⁾ = Σ exp(x⁽¹⁾ − m⁽¹⁾)
m    = max(m⁽¹⁾, max(x⁽²⁾))     
```

Then the corrected statistics for the merged row are

```
l = e^(m⁽¹⁾ − m) · l⁽¹⁾  +  Σ exp(x⁽²⁾ − m)
```

and the same correction factor `α = e^(m_old − m_new)` rescales the partial
output accumulator:

```
O_acc ← α · O_acc + exp(S⁽²⁾ − m_new) V⁽²⁾
```

Every update is an exact algebraic identity, so after the last block `O_acc / l`
is the exact attention output — bit-for-bit up to floating-point rounding, which
the reference implementation incurs too. This is why `tests/test_correctness.py`
asserts against fp32 reference values with only dtype-level tolerances, not
approximation tolerances.

## 3. FlashAttention-2: what changed

FlashAttention-1 rescaled the output by `1/l` on every inner iteration.
FlashAttention-2 makes three changes, all implemented here:

1. **Defer the normalisation.** Accumulate unnormalised, divide once at the end
   (`fwd.py`, after the inner loop). Saves one divide per tile per row.
2. **Loop order.** Queries in the *outer* loop, keys/values in the inner loop,
   so the output accumulator and softmax state stay resident in registers and
   never round-trip through shared memory.
3. **Parallelise over the sequence dimension.** The grid is
   `(ceil(N / BLOCK_M), B·H)` rather than just `B·H`. At long context with small
   batch, `B·H` alone leaves most SMs idle; splitting queries across programs
   keeps the GPU occupied.

There is a fourth idea worth noting because it shows up in the causal path:
for a causal mask, a Q tile at row block `i` only needs K tiles `0..i`. Rather
than masking every tile, `fwd.py` splits the inner loop into a fully-unmasked
prefix and a single diagonal tile that carries the mask. Roughly half the work
disappears and the `tl.where` executes on one tile instead of all of them.

## 4. Backward without storing P

The backward pass needs `P` to form the gradients:

```
dV = Pᵀ dO
dP = dO Vᵀ
dS = P ∘ (dP − δ)        where δ_i = Σ_j P_ij dP_ij
dQ = dS K / √d,   dK = dSᵀ Q / √d
```

Storing `P` would reintroduce the Θ(N²) memory we just removed. Instead the
forward pass saves only the log-sum-exp vector

```
L_i = m_i + log(l_i)        (N floats per head, not N²)
```

and backward recomputes each probability tile as `P_ij = exp(S_ij − L_i)`. The
recomputation costs an extra QKᵀ per tile; the memory saving is a factor of N.

The `δ` term deserves a note. The softmax Jacobian is
`∂P_i/∂S_i = diag(P_i) − P_i P_iᵀ`, which naively implies an N×N Jacobian per
row. Contracting it with `dP` collapses to `P ∘ (dP − δ)` where `δ` is a scalar
per row — and by the chain rule through `O = PV`, that scalar equals
`Σ_d dO_id O_id`. So `_attn_bwd_preprocess` computes it in one cheap O(Nd) pass
and the expensive kernels just read it.

## 5. Why two backward kernels

`dQ` accumulates over key tiles; `dK`/`dV` accumulate over query tiles. One
fused kernel therefore needs atomics on whichever gradient it does not
parallelise over. On Ampere, fp32 atomic contention on `dQ` measurably hurts.
Splitting into `_attn_bwd_dkdv` (grid over N tiles) and `_attn_bwd_dq` (grid
over M tiles) recomputes `QKᵀ` twice but removes the atomics entirely.

## 6. What is *not* implemented

Stated explicitly so the scope is honest:

- No Hopper/TMA or warp-specialisation path (FlashAttention-3). Ampere-targeted.
- No variable-length / packed-sequence (`cu_seqlens`) API yet.
- No dropout inside the kernel, no ALiBi, no sliding-window mask.
- No MQA/GQA head broadcasting — K and V must have the same head count as Q.
- Head dims limited to {16, 32, 64, 128}.

## References

Primary sources, ordered by which section of this document they support.

**§1 — IO as the bottleneck**

1. Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact
   Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135
   — Source of the Θ(N²d²/M) HBM-access bound quoted above, and of the
   SRAM/HBM bandwidth figures.
2. Ivanov, Dryden, Ben-Nun, Li, Hoefler. *Data Movement Is All You Need: A Case
   Study on Optimizing Transformers.* MLSys 2021. arXiv:2007.00072
   — Measures the split directly: a large fraction of transformer runtime is
   data movement, not arithmetic. The empirical case for the whole approach.
3. Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
   Model for Multicore Architectures.* Communications of the ACM, 2009.
   — The arithmetic-intensity framing that makes "memory-bound" precise; it is
   what `bench/bench_attention.py` is implicitly measuring against.

**§2 — online softmax**

4. Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018.
   arXiv:1805.02867
   — The running-max/running-sum recurrence, and the proof it is exact.
5. Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021.
   arXiv:2112.05682
   — Independent derivation of chunked attention with lazy normalisation;
   establishes the O(log n) / O(1) memory result the tiling relies on.

**§3 — FlashAttention-2 changes**

6. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work
   Partitioning.* ICLR 2024. arXiv:2307.08691
   — Deferred normalisation, Q-outer loop order, and parallelising over the
   sequence dimension: the three changes implemented in `kernels/fwd.py`.
7. Tillet, Kung, Cox. *Triton: An Intermediate Language and Compiler for Tiled
   Neural Network Computations.* MAPL 2019.
   — The block-pointer / tiled programming model these kernels are written in,
   and why `BLOCK_M`/`BLOCK_N`/`num_warps`/`num_stages` are the tuning surface.

**§4–5 — backward pass**

8. Dao et al. (ref. 1), Appendix B — the dQ/dK/dV derivation and the δ identity
   δᵢ = Σ_d dO_id O_id used by `_attn_bwd_preprocess`.
9. Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin.
   *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762
   — The scaled dot-product definition being differentiated.

**§6 — what is out of scope, and why**

10. Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and
    Accurate Attention with Asynchrony and Low-precision.* NeurIPS 2024.
    arXiv:2407.08608 — Hopper TMA and warp specialisation; requires SM 90.
11. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019.
    arXiv:1911.02150 — MQA.
12. Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai. *GQA: Training
    Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*
    EMNLP 2023. arXiv:2305.13245 — GQA; both need K/V head broadcasting, which
    the current kernel does not implement.
13. Press, Smith, Lewis. *Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation.* ICLR 2022. arXiv:2108.12409 — ALiBi.
14. Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer.* 2020.
    arXiv:2004.05150 — sliding-window masking.
