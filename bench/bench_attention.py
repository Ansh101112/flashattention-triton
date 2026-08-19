"""Measures speed and memory. Run this to fill in docs/RESULTS.md.

No number in this repo is hardcoded -- everything comes out of this script.

    python bench/bench_attention.py --causal --dtype fp16 --out docs/RESULTS.md

Compares three implementations:
    flash   the Triton kernel in this repo
    sdpa    torch.nn.functional.scaled_dot_product_attention (already fused,
            so this is the real bar to clear)
    naive   plain softmax(Q@K^T)@V, builds the whole N x N matrix

naive runs out of memory long before the other two. The script catches that and
records "OOM" instead of crashing, because the sequence length where the naive
version dies is itself one of the more interesting results.
"""

import argparse
import json
import platform
import sys

import torch
import triton

sys.path.insert(0, "src")
from flash_attn_triton import flash_attn, naive_attention, torch_sdpa, attention_flops  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def measure(fn, backward=False):
    """Times fn and records how much memory it needed.

    Returns (milliseconds, peak MiB), or (None, None) if it ran out of memory.
    Median is used rather than mean so one slow first run doesn't skew it.
    """
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

        if backward:
            def run():
                out = fn()
                out.backward(torch.ones_like(out), retain_graph=True)
        else:
            run = fn

        ms = triton.testing.do_bench(run, warmup=25, rep=100, quantiles=[0.5])
        if isinstance(ms, (list, tuple)):
            ms = ms[0]
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        return float(ms), float(peak)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return None, None


def bench_one(B, H, N, D, dtype, causal, backward, include_naive):
    rg = backward
    q, k, v = [
        torch.randn((B, H, N, D), device="cuda", dtype=dtype, requires_grad=rg)
        for _ in range(3)
    ]
    flops = attention_flops(B, H, N, D, causal=causal, backward=backward)

    impls = {
        "flash": lambda: flash_attn(q, k, v, causal=causal),
        "sdpa": lambda: torch_sdpa(q, k, v, causal=causal),
    }
    if include_naive:
        impls["naive"] = lambda: naive_attention(q, k, v, causal=causal)

    row = {"B": B, "H": H, "N": N, "D": D}
    for name, fn in impls.items():
        ms, mem = measure(fn, backward=backward)
        row[name] = {
            "ms": ms,
            "mem_mib": mem,
            "tflops": (flops / (ms * 1e-3) / 1e12) if ms else None,
            "oom": ms is None,
        }
    # No `del q, k, v` here: the lambdas above close over those names, and a
    # `del` in the same scope leaves the closures referring to unbound names.
    # Returning drops the last references anyway; the caller reclaims the
    # blocks once this frame is gone.
    return row


def fmt(v, spec=".3f"):
    return "OOM" if v is None else format(v, spec)


def to_markdown(rows, meta):
    lines = [
        f"# Benchmark results",
        "",
        f"- GPU: `{meta['gpu']}`  ({meta['sm_count']} SMs, {meta['vram_gib']:.1f} GiB)",
        f"- torch `{meta['torch']}` / triton `{meta['triton']}` / {meta['platform']}",
        f"- dtype `{meta['dtype']}`, causal={meta['causal']}, pass={meta['pass']}",
        "",
        "| B | H | N | D | flash ms | sdpa ms | naive ms | flash TFLOP/s | sdpa TFLOP/s |"
        " flash MiB | naive MiB | speedup vs naive | speedup vs sdpa |",
        "|--:|--:|--:|--:|---------:|--------:|---------:|--------------:|-------------:|"
        "----------:|----------:|-----------------:|----------------:|",
    ]
    for r in rows:
        f, s = r["flash"], r["sdpa"]
        n = r.get("naive", {"ms": None, "mem_mib": None, "tflops": None})
        sp_naive = f"{n['ms'] / f['ms']:.2f}x" if n["ms"] and f["ms"] else "-"
        sp_sdpa = f"{s['ms'] / f['ms']:.2f}x" if s["ms"] and f["ms"] else "-"
        lines.append(
            f"| {r['B']} | {r['H']} | {r['N']} | {r['D']} "
            f"| {fmt(f['ms'])} | {fmt(s['ms'])} | {fmt(n['ms'])} "
            f"| {fmt(f['tflops'], '.1f')} | {fmt(s['tflops'], '.1f')} "
            f"| {fmt(f['mem_mib'], '.1f')} | {fmt(n['mem_mib'], '.1f')} "
            f"| {sp_naive} | {sp_sdpa} |"
        )
    lines += ["", "`OOM` means that implementation could not allocate at this shape.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--seqlens", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--backward", action="store_true", help="benchmark fwd+bwd")
    p.add_argument("--no-naive", action="store_true", help="skip the O(N^2) baseline")
    p.add_argument("--out", default="docs/RESULTS.md")
    p.add_argument("--json-out", default="docs/results.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required.")

    props = torch.cuda.get_device_properties(0)
    meta = {
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "vram_gib": props.total_memory / 2**30,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "platform": platform.platform(),
        "dtype": args.dtype,
        "causal": args.causal,
        "pass": "fwd+bwd" if args.backward else "fwd",
    }

    rows = []
    for N in args.seqlens:
        print(f"  N={N} ...", flush=True)
        rows.append(bench_one(
            args.batch, args.heads, N, args.head_dim,
            DTYPES[args.dtype], args.causal, args.backward, not args.no_naive,
        ))
        # bench_one's frame is gone, so its tensors are unreachable. Hand the
        # blocks back to the driver before allocating the next, larger shape --
        # on a small card this is the difference between finishing and OOMing.
        torch.cuda.empty_cache()

    md = to_markdown(rows, meta)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "rows": rows}, fh, indent=2)
    print(md)
    print(f"\nwrote {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
