# FlashAttention in Triton

I implemented the FlashAttention-2 forward and backward kernels in Triton,
working from the papers. It comes with a plain-PyTorch reference to test
against and a benchmark script that measures speed and memory.

The thing I wanted to understand with this project: attention at long context
is **slow because of memory movement, not because of maths**. This kernel
actually does *more* arithmetic than normal attention in the backward pass, and
it is still much faster, purely because of what it does or doesn't write to GPU
main memory.

There's an interactive page at
**[ansh101112.github.io/flashattention-triton/docs](https://ansh101112.github.io/flashattention-triton/docs/)**
that runs the tiled inner loop in the browser, so you can watch the blocks go
through and check for yourself that the online softmax comes out exact.

Full walkthrough of the maths is in [`docs/DERIVATION.md`](docs/DERIVATION.md).

## The idea in one diagram

```
Q, K, V sitting in HBM (slow GPU main memory)
   │
   ├── split Q into blocks of BLOCK_M rows ──┐  outer loop, one program per block
   │                                         │
   │   ┌── split K,V into BLOCK_N rows ───┐  │  inner loop, stays in fast memory
   │   │   scores = Q_i @ K_j^T * scale   │  │
   │   │   update running max + sum       │  │  exact, no N×N matrix anywhere
   │   │   acc = acc * alpha + P_ij @ V_j │  │
   │   └─────────────────────────────────┘  │
   │                                        │
   └── divide by the sum, save L = m+log(l) ┘  L is N floats, not N×N
```

The N×N score matrix never exists in main memory. That's the whole trick.

## Why this is faster

A normal attention implementation writes the N×N score matrix to HBM, reads it
back to run softmax, writes it again, then reads it once more to multiply by V.
At 8K context that matrix is 128 MB per head in fp16.

The GPU memory hierarchy is very lopsided (numbers for an A100):

| | Size | Bandwidth |
|---|---|---|
| On-chip SRAM | ~192 KB per SM | ~19 TB/s |
| HBM (main memory) | 40–80 GB | ~1.5–2 TB/s |

So every trip to HBM is roughly 10x slower than staying on-chip. Tiling keeps
the working set small enough to stay on-chip for the whole inner loop, which is
where the speedup comes from.

## What's implemented

| | |
|---|---|
| Forward kernel (FA-2 style: normalise at the end, Q in the outer loop, grid split over sequence) | done |
| Backward kernels for dQ, dK, dV, rebuilding P from the saved L | done |
| Causal masking, with the mask only applied on the diagonal tile | done |
| fp16 and bf16 | done |
| Sequence lengths that aren't a multiple of the tile size | done |
| Autotuning over BLOCK_M / BLOCK_N / warps / stages | done |
| `autograd.Function` + a drop-in `FlashSelfAttention` module | done |
| Tests against fp32 PyTorch across shapes, dtypes and masking | done |
| Benchmark script (speed, TFLOP/s, peak memory, OOM point) | done |
| MQA/GQA, dropout, ALiBi, sliding window, variable-length batches | not yet |
| Hopper / FlashAttention-3 path | out of scope |

## Setup

Needs an NVIDIA GPU (SM 80+ is what I targeted), CUDA PyTorch, and Triton.
**Triton does not run on Windows** — use Linux, WSL2, or a Colab notebook.

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import torch
from flash_attn_triton import flash_attn, FlashSelfAttention

# shape is (batch, heads, seqlen, head_dim), fp16 or bf16, on CUDA
q, k, v = (torch.randn(4, 16, 8192, 64, device="cuda", dtype=torch.float16) for _ in range(3))
out = flash_attn(q, k, v, causal=True)   # gradients work as normal

# or swap it into a model in place of an attention block
attn = FlashSelfAttention(embed_dim=1024, num_heads=16, causal=True).cuda().half()
y = attn(torch.randn(4, 8192, 1024, device="cuda", dtype=torch.float16))
```

## Tests

```bash
pytest tests/ -v
```

FlashAttention is exact, not an approximation, so these tests compare against
fp32 PyTorch attention with only normal fp16/bf16 rounding tolerance.

They cover fp16 and bf16, causal and non-causal, head dims 32/64/128, sequence
lengths that deliberately don't divide the tile size (200 and 999), a custom
softmax scale, a check that row 0 under a causal mask genuinely can't see
anything after it, and a check that peak memory grows linearly rather than
quadratically with sequence length.

## Benchmarks

```bash
python bench/bench_attention.py --causal --dtype fp16 --backward --out docs/RESULTS.md
```

This prints, for each shape, the time and TFLOP/s for this kernel, for torch's
own `scaled_dot_product_attention`, and for plain `softmax(Q@K^T)@V` — plus peak
memory for each, and the sequence length where the plain version runs out of
memory.

> **[`docs/RESULTS.md`](docs/RESULTS.md) is empty on purpose.** I haven't
> committed numbers from a machine I didn't run this on. Run the script and it
> fills the file in itself. Results depend heavily on the GPU (SM count and
> memory bandwidth both move them a lot), so the script records the GPU name and
> the torch/triton versions next to every number.

On a small GPU, use a smaller sweep so it doesn't OOM:

```bash
python bench/bench_attention.py --batch 1 --heads 8 --seqlens 512 1024 2048 4096
```

## Files

```
src/flash_attn_triton/
  kernels/fwd.py     forward kernel, autotuned
  kernels/bwd.py     delta pre-pass, then the dK/dV kernel and the dQ kernel
  interface.py       autograd.Function, flash_attn(), FlashSelfAttention
  reference.py       plain PyTorch attention, torch SDPA, FLOP counter
bench/               speed and memory measurement
tests/               correctness + memory scaling
docs/DERIVATION.md   the maths: online softmax, why it's exact, backward pass
```

## Papers I used

Full notes on which paper contributed what are in
[`docs/DERIVATION.md`](docs/DERIVATION.md).

**The algorithm**
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness.* NeurIPS 2022 — arXiv:2205.14135
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning.* ICLR 2024 — arXiv:2307.08691
- Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and
  Low-precision.* NeurIPS 2024 — arXiv:2407.08608 *(Hopper only, not implemented here)*

**Why tiling softmax is still exact**
- Milakov, Gimelshein. *Online normalizer calculation for softmax.* 2018 —
  arXiv:1805.02867
- Rabe, Staats. *Self-attention Does Not Need O(n²) Memory.* 2021 — arXiv:2112.05682

**Why memory movement is the bottleneck**
- Ivanov et al. *Data Movement Is All You Need: A Case Study on Optimizing
  Transformers.* MLSys 2021 — arXiv:2007.00072
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance
  Model for Multicore Architectures.* CACM 2009

**The compiler**
- Tillet, Kung, Cox. *Triton: An Intermediate Language and Compiler for Tiled
  Neural Network Computations.* MAPL 2019

**Background / features not implemented**
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017 — arXiv:1706.03762
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019 —
  arXiv:1911.02150 *(MQA)*
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models.*
  EMNLP 2023 — arXiv:2305.13245

## License

MIT
