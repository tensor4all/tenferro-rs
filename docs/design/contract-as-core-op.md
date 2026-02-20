# Why `Contract` Is a Core Operation

## Context

`PrimDescriptor::Contract` fuses permutation and GEMM into a single primitive,
matching cuTENSOR's `cutensorContract`. It was initially an extended operation
(dynamically queried) but was promoted to core (every backend must implement)
based on performance findings from strided-rs benchmarks.

## The Problem: Permute + BatchedGemm Is Suboptimal

When an einsum contraction tree is executed step by step, each step typically
involves:

1. Permuting input operands into GEMM-compatible layout
2. Executing GEMM
3. Permuting the output for the next step

If the einsum layer decomposes this into separate `Permute` + `BatchedGemm`
calls, the `Permute` primitive must fully materialize the permuted tensor. This
is suboptimal because:

- **Unnecessary copies**: The contraction backend can often skip the copy
  entirely when strides are already compatible (`try_fuse_group` in strided-rs).
  A separate `Permute` cannot know this.

- **Wrong copy strategy**: The optimal copy strategy depends on cache state.
  Between contraction steps, source data is typically cold in cache (evicted by
  the previous GEMM). Source-stride-order iteration (sequential reads,
  scattered writes) outperforms HPTT (destination-stride-order with cache
  blocking) for cold-cache sources. But a standalone `Permute` cannot know the
  cache state — only the contraction backend has this context.

## Benchmark Evidence

Experiments on strided-rs (branch `perf/eager-hptt-permute`) compared:
- **Lazy permutation**: metadata-only reorder, backend handles copy internally
- **Eager HPTT**: always materialize via `Permute` (HPTT) before GEMM

Results on AMD EPYC 7713P (faer backend):

| Instance | Lazy 1T | Eager 1T | Regression |
|---|---|---|---|
| mera_open (opt_flops) | 918 ms | 1199 ms | **+31%** |
| mera_open (opt_size) | 918 ms | 1159 ms | **+26%** |
| tensor network instances | ~285 ms | ~287 ms | ~0% |

The `mera_open` regression is caused by eager permutation forcing copies at
every step, even when `try_fuse_group` would have skipped them.

See `strided-rs/docs/eager-hptt-experiment.md` for full results.

## Design Decision

Making `Contract` a **core operation** means:

1. **The einsum layer always emits `Contract`** — no fallback to
   `Permute` + `BatchedGemm` needed.

2. **Each backend controls internal data movement** — CPU backend can use
   source-stride-order copy, try_fuse_group elision, buffer pooling, etc.
   GPU backend delegates to `cutensorContract`.

3. **No hints needed on `Permute`** — `Permute` remains a simple standalone
   operation (for final output permutation, etc.) without cache-state hints.
   The performance-critical path goes through `Contract`.

## Relationship to `BatchedGemm`

`BatchedGemm` remains a separate core operation for cases where the data is
already in GEMM-ready layout (pre-packed, contiguous batch slices). `Contract`
is the general-purpose contraction that handles arbitrary mode labels. A
backend may implement `Contract` by internally calling its `BatchedGemm` after
preparing operands.
