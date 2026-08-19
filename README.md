# FlashAttention in Triton

An exact, IO-aware attention kernel written from the paper — FlashAttention-2
forward and backward in Triton, with a fp32 reference oracle, a numerical
parity suite, and a benchmark harness that generates every number in this repo.

The point of the project is not "attention, but faster." It is that attention at
long context is **memory-bound, not compute-bound**, and that you can get a large
speedup while doing *more* arithmetic, purely by changing what crosses the HBM
boundary. The full derivation is in [`docs/DERIVATION.md`](docs/DERIVATION.md).

```
Q,K,V in HBM
   │
   ├── tile Q into BLOCK_M rows ────────────┐   outer loop (one program per tile)
   │                                        │
   │   ┌── tile K,V into BLOCK_N rows ───┐  │   inner loop, stays in SRAM
   │   │   S_ij = Q_i K_jᵀ · scale       │  │
   │   │   online softmax: m_i, l_i, α   │  │   exact, no N×N matrix
   │   │   acc ← α·acc + P_ij V_j        │  │
   │   └─────────────────────────────────┘  │
   │                                        │
   └── O_i = acc / l_i,  save L_i = m+log l ┘   L is O(N), replaces storing P
```

## What is implemented

| | Status |
|---|---|
| Forward kernel (FA-2: deferred normalisation, Q-outer loop, seq-parallel grid) | ✅ |
| Backward kernels (`dQ`, `dK`, `dV`) with P recomputed from the saved LSE | ✅ |
| Causal masking with a split unmasked / diagonal inner loop | ✅ |
| fp16 and bf16 | ✅ |
| Ragged sequence lengths (not a multiple of the tile size) | ✅ |
| Autotuned `BLOCK_M`/`BLOCK_N`/warps/stages | ✅ |
| `torch.autograd.Function` + drop-in `FlashSelfAttention` module | ✅ |
| Parity tests vs an fp32 oracle across shapes, dtypes, masking | ✅ |
| Benchmark harness (latency, TFLOP/s, peak memory, OOM point) | ✅ |
| MQA/GQA, dropout, ALiBi, sliding window, `cu_seqlens` | ❌ not yet |
| Hopper / TMA path (FlashAttention-3) | ❌ out of scope |

## Install

Requires an NVIDIA GPU (SM 80+ recommended), CUDA PyTorch, and Triton. Triton
does not support Windows natively — use Linux or WSL2.

```bash
pip install -r requirements.txt
pip install -e .
```

## Use

```python
import torch
from flash_attn_triton import flash_attn, FlashSelfAttention

# (batch, heads, seqlen, head_dim), fp16 or bf16, on CUDA
q, k, v = (torch.randn(4, 16, 8192, 64, device="cuda", dtype=torch.float16) for _ in range(3))
out = flash_attn(q, k, v, causal=True)          # differentiable

# or drop it into a model in place of an attention block
attn = FlashSelfAttention(embed_dim=1024, num_heads=16, causal=True).cuda().half()
y = attn(torch.randn(4, 8192, 1024, device="cuda", dtype=torch.float16))
```

## Correctness

FlashAttention is *exact*, so the tests assert against a fp32 reference with
dtype-level tolerances only — no approximation slack.

```bash
pytest tests/ -v
```

Covered: fp16 and bf16 · causal and non-causal · head dims 32/64/128 · sequence
lengths that deliberately do not divide the tile size (200, 999) · custom
softmax scale · a causal regression test that row 0 cannot see `K[1:]` · a
memory-scaling test asserting peak memory grows ~linearly, not quadratically,
with sequence length.

## Benchmarks

Every number is produced by the harness — nothing is hard-coded.

```bash
python bench/bench_attention.py --causal --dtype fp16 --backward --out docs/RESULTS.md
```

It reports, per shape, the median latency and achieved TFLOP/s for this kernel,
for `torch.nn.functional.scaled_dot_product_attention`, and for the explicit
`softmax(QKᵀ)V` baseline, plus peak memory and the sequence length at which the
quadratic baseline runs out of memory.

Results for the machine you run it on land in
[`docs/RESULTS.md`](docs/RESULTS.md) and `docs/results.json`.

> Results are intentionally not committed for a machine this repo was not
> benchmarked on. Run the harness and the table fills itself in; the numbers are
> strongly hardware-dependent (SM count and HBM bandwidth both move them).

## Layout

```
src/flash_attn_triton/
  kernels/fwd.py     FlashAttention-2 forward, autotuned
  kernels/bwd.py     preprocess (δ) + dK/dV kernel + dQ kernel
  interface.py       autograd.Function, flash_attn(), FlashSelfAttention
  reference.py       fp32 oracle, torch SDPA baseline, FLOP counter
bench/               latency / TFLOP-s / memory harness
tests/               numerical parity + memory-scaling suite
docs/DERIVATION.md   online softmax, IO complexity, backward without P
```

## References

Each paper's contribution and where it lands in this code is written out in
[`docs/DERIVATION.md`](docs/DERIVATION.md).

**The algorithm**
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness.* NeurIPS 2022 — arXiv:2205.14135
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning.* ICLR 2024 — arXiv:2307.08691
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and
  Accurate Attention with Asynchrony and Low-precision.* NeurIPS 2024 —
  arXiv:2407.08608 *(Hopper-specific; out of scope here, see DERIVATION §6)*

**The softmax trick that makes tiling exact**
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018 —
  arXiv:1805.02867
- Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021 —
  arXiv:2112.05682

**Why the bottleneck is IO**
- Ivanov, Dryden, Ben-Nun, Li, Hoefler. *Data Movement Is All You Need: A Case
  Study on Optimizing Transformers.* MLSys 2021 — arXiv:2007.00072
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM 2009

**The compiler this is written against**
- Tillet, Kung, Cox. *Triton: An Intermediate Language and Compiler for Tiled
  Neural Network Computations.* MAPL 2019

**Context for the head-dim / KV constraints**
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017 — arXiv:1706.03762
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019 —
  arXiv:1911.02150 *(MQA — not yet supported, see DERIVATION §6)*
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models.*
  EMNLP 2023 — arXiv:2305.13245 *(likewise)*

## License

MIT
