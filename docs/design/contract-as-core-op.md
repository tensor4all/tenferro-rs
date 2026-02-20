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

- **Wrong copy strategy**: When a copy is needed, there are two iteration
  orders — source-stride-order and destination-stride-order (HPTT). The
  optimal choice depends on cache state, which a standalone `Permute` cannot
  know.

### Source-stride-order vs HPTT (destination-stride-order)

A permutation copy must traverse the same elements in two different stride
orders (source and destination). The question is which order to follow:

| | Source-stride-order | HPTT (dst-stride-order) |
|---|---|---|
| Reads | Sequential (hardware prefetcher effective) | Scattered |
| Writes | Scattered (absorbed by write-combining buffers) | Sequential + cache-blocked |

**Source-stride-order** iterates in ascending source stride, giving
sequential reads exploited by the hardware prefetcher. Scattered
destination writes are absorbed by write-combining buffers.

**HPTT (destination-stride-order)** gives sequential writes with
cache-blocked reads, which is better when source data is warm in cache.

In theory, the choice depends on whether the source data is cache-hot
(just computed) or cache-cold (evicted by intervening work). In a
depth-first contraction tree, the right operand is typically warm (used
immediately after computation), while the left operand may or may not
be cold depending on the right subtree's size.

**In practice**, a direct comparison (branch `perf/src-vs-dst-order`) with
copy elision disabled (`force-copy` feature) showed that **HPTT is faster
on most workloads** (16–43% faster on lm_*, str_*, mera_closed), while
**source-stride-order is faster only for degenerate cases with many small
binary dimensions** (tn_focus/tn_light: 27–30% faster). On `mera_open`,
the two strategies perform identically (±2%), confirming that the 26–31%
regression in the eager-HPTT experiment was entirely due to copy elision loss.

The dominant optimization is copy elision. When copies cannot be avoided,
HPTT is the better default thanks to cache-blocked tiling.

See `strided-rs/docs/src-vs-dst-order-experiment.md` for full results.

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

## CPU Backend Implementation Strategy

`Contract::execute` receives `&[&StridedView<T>]` inputs — these may have
arbitrary strides from lazy permutation in the einsum layer. The CPU backend
should follow this priority order:

1. **Skip the copy** (`try_fuse_group`): Check if each input's dimension
   groups are already contiguous enough for GEMM. If so, pass the raw
   pointers and strides directly — zero-cost. This is the most impactful
   optimization (responsible for the mera_open 26–31% gap).

2. **HPTT (destination-stride-order) copy**: When materialization is
   needed, use HPTT's cache-blocked tiling. The `perf/src-vs-dst-order`
   experiment showed HPTT outperforms source-stride-order on most
   workloads (16–43% faster). Exception: for tensors with many small
   dimensions (e.g., 24 binary dims of size 2), source-stride-order
   is 27–30% faster due to HPTT's recursion degenerating.

3. **GEMM**: Call `BatchedGemm` on the prepared contiguous operands.

### Copy strategy experiments summary

Three experiments characterize the copy strategy landscape:

1. **Flatten HPTT recursion** (`perf/flatten-hptt-recursion`): Replaced
   recursive ComputeNode traversal with flat odometer — **no improvement**
   (±3% noise). Recursion overhead is not a bottleneck.

2. **Source-order vs HPTT** (`perf/src-vs-dst-order`): With copy elision
   disabled, HPTT is 16–43% faster on most workloads. Source-order is
   27–30% faster only for many-small-dims cases (tn_focus/tn_light).

3. **Eager HPTT** (`perf/eager-hptt-permute`): Eagerly materializing all
   permutations via HPTT causes 26–31% regression on `mera_open` — entirely
   due to copy elision loss, not copy strategy.

**Conclusion**: Copy elision is the dominant optimization. When copies are
unavoidable, HPTT is the better default. An adaptive strategy switching to
source-order for degenerate many-small-dims cases could provide the best of
both worlds.

## Relationship to `BatchedGemm`

`BatchedGemm` remains a separate core operation for cases where the data is
already in GEMM-ready layout (pre-packed, contiguous batch slices). `Contract`
is the general-purpose contraction that handles arbitrary mode labels. A
backend may implement `Contract` by internally calling its `BatchedGemm` after
preparing operands.
