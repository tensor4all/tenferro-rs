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

**In practice**, the flatten-HPTT experiment (`perf/flatten-hptt-recursion`)
showed no measurable difference between source-order and HPTT-order
iteration when both use the same flat odometer structure (±5% noise).
The copy strategy is not the bottleneck — **copy elision is**.

See `strided-rs/docs/permutation-optimization.md` for detailed analysis.

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

2. **Source-stride-order copy**: When materialization is needed, iterate
   in ascending source stride order. This gives sequential reads that
   exploit the hardware prefetcher. The destination writes are scattered
   but absorbed by write-combining buffers. Note: whether this actually
   outperforms HPTT (destination-stride-order) depends on cache state —
   in the flatten-HPTT experiment, the two strategies showed no
   measurable difference. Source-stride-order is kept as the default
   for simplicity, not for a proven performance advantage.

3. **GEMM**: Call `BatchedGemm` on the prepared contiguous operands.

### Why not improve HPTT instead?

An experiment (branch `perf/flatten-hptt-recursion`) replaced HPTT's
recursive ComputeNode traversal with a flat iterative odometer loop,
eliminating all function-call overhead. Results showed **no improvement**
(±3% noise), confirming that:

- The recursion overhead is not the bottleneck
- The performance difference is fundamentally about **copy elision**
  (`try_fuse_group`), not copy strategy
- Source-stride-order is a reasonable default when copies cannot be skipped,
  but the primary win comes from not copying at all

## Relationship to `BatchedGemm`

`BatchedGemm` remains a separate core operation for cases where the data is
already in GEMM-ready layout (pre-packed, contiguous batch slices). `Contract`
is the general-purpose contraction that handles arbitrary mode labels. A
backend may implement `Contract` by internally calling its `BatchedGemm` after
preparing operands.
