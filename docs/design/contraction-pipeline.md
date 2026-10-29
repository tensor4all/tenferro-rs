# Binary Contraction Pipeline

This document describes the current binary step used inside an N-ary einsum
contraction tree. It is implemented through `tenferro-einsum` planning and the
`TensorBackend::dot_general` execution surface in `tenferro-tensor`.

See [einsum.md](./einsum.md) for the N-ary API and
[tensor-prims.md](./tensor-prims.md) for the current tensor backend protocol.

---

## Step Inputs

Each pairwise step receives:

- left operand labels,
- right operand labels,
- output labels for this intermediate or final result,
- label sizes from the tree's size dictionary.

Repeated labels are normalized before ordinary pairwise contraction. Strict
binary lowering only accepts non-repeated labels; repeated-label cases fall back
to the general path.

## Label Classification

For a binary contraction, labels are classified into:

| Class | Meaning |
| --- | --- |
| left free | appears only in the left operand and survives |
| right free | appears only in the right operand and survives |
| batch | appears in both operands and survives |
| contraction | appears in both operands and does not survive |
| pre-reduced | appears in one operand and does not survive |

Pre-reduced labels are reduced before the `dot_general` step.

## Column-Major Layout

tenferro uses column-major storage. The canonical `DotGeneral` output order is:

```text
lhs_free..., rhs_free..., batch...
```

The strict binary lowering tracks GEMM-like metadata with compute dimensions on
the left and batch dimensions on the right:

```text
lhs matrix dims: lhs_free..., contract...
rhs matrix dims: contract..., rhs_free...
output dims:     lhs_free..., rhs_free..., batch...
```

Batch-last placement means each batch slice occupies a contiguous block in
column-major layout, which is the convention used by the CPU GEMM path.

## Strict Binary Lowering

`crates/tenferro-einsum/src/planning/strict_binary.rs` detects dense binary
contractions that can be represented as one canonical `DotGeneral` plus a final
output transpose if needed.

The strict plan records:

- left and right free labels,
- contraction labels,
- batch labels,
- operand permutations,
- `m`, `k`, and `n`,
- canonical output dimensions,
- final output permutation.

It returns `None` for patterns it should not own, including repeated labels.
That is not an error; the general path handles those semantics.

## General Path

The eager executor and graph builder use the general path when strict binary
lowering does not apply:

1. Extract diagonals for repeated input labels.
2. Reduce labels that no longer survive.
3. Broadcast and multiply for pure outer/Hadamard-like cases.
4. Use `dot_general` for ordinary contractions.
5. Embed repeated output labels with `embed_diagonal`.
6. Transpose to the requested output label order.

This path favors correctness and broad semantics. Backend-specific performance
comes from the lower `TensorBackend` implementation of `dot_general`, reduction,
transpose, and diagonal operations.

## CPU Execution

CPU execution is handled by `CpuBackend`:

- elementwise/reduction/structural work uses the split strided-basic,
  strided-kernel and strided-fused implementations plus dedicated CPU
  semantics, with automatic fusion retaining the enclosing context and pool,
- `dot_general`, grouped GEMM, and N-ary concrete einsum delegate to cpueinsum
  prepared plans, and CPU linalg delegates to tlinalg / tlinalg-blas,
- the lower library runs inside the selected engine pool through
  `CpuExecutionContext`, with `Sequential` for one-thread contexts and the
  selected pool plus a thread budget otherwise,
- vendor BLAS/LAPACK threading remains vendor-owned; tenferro guarantees only
  the coordinator thread's mask.

Provider selection is compile-time (`native` or `blas`); there is no runtime
provider bundle or override hook. Unsupported dtypes and shapes map to typed
errors.

### Contraction plan cache and the eager slot

The session's `GemmAnalysisCache` retains the prepared numerical plans, bounded
by a slot capacity (1024 by default), and reports the retained entries and
hit/miss counters through `runtime_cache_stats`. A `cache_slot` names one slot of
that cache.

A compiled program keys a plan by the instruction index of the `dot_general` it
prepares: a program has exactly one such index per operation, so the same slot
denotes the same geometry on every execution. **An eager call has no instruction
index**, so the eager entries (`dot_general_read`, `dot_general_read_into_accum`,
`dot_general_with_conj`) derive the slot from the request instead: a hash of the
dtype, the two operand shapes, the four axis lists and the conjugation flags,
reduced modulo the cache capacity.

The slot only *routes* a request. `contraction::BinaryCache` compares the full
plan key - dtype, both operand layouts including strides and offsets, the output
layout, the conjugation flags - and the `DotGeneralConfig` before reusing a plan,
so a fingerprint collision rebuilds instead of returning the wrong plan. The
fingerprint deliberately omits the operand strides: reading them here would cost
the allocation the cache exists to avoid, and including them is not needed for
correctness. The consequence to keep in mind is a **hit-rate**, not a
correctness, one: two calls of the same shape reached through differently shaped
views of the same storage land in the same slot and one of them misses.

This changed what the eager path retains. Before, an eager `dot_general` built
its plan, used it once, and dropped it; now it is retained until the cache is
cleared, the capacity shrinks, or the slot is reused by a different geometry.
That is the intended trade of #2141 item 1 and it is reported through
`runtime_cache_stats` rather than being hidden. Measured on one AMD EPYC 7713P
thread (`CpuBackend::with_threads(1)`) with a warm output pool and an f64 64x64
`matmul`: **84 -> 0 heap allocations per repeated call** (415 on the cold call,
which also warms the provider) and **22.4-23.6 us -> 16.5-18.5 us per call**.

A miss on a slot that already holds an entry allocates no new `Arc`: the
`BinaryCache` rebuilds the prepared plan in the slot's own storage when the
stored `Arc` is exclusively owned (no caller is still running it) and has the
requested scalar type. A plan another caller holds, or another scalar type,
falls back to a fresh `Arc` - that plan keeps working for its holder - and a
failed preparation returns the error before touching the entry, so only the
miss counter moves. Requests without a retained slot (no slot, or one at or
beyond the capacity) have no storage to reuse and still allocate. This is what
keeps a caller that streams GEMMs of different shapes through one slot from
paying an allocation per call once warm (#2314): with the native provider and
f64, three alternating shapes and an eight-block recoupling sequence measured
**3 -> 0** and **8 -> 0** allocations per warm round with the slot capacity
forced to one. The rebuilt lower plan itself may still allocate: `cpueinsum-blas`
planning builds vendor metadata on every miss, and large packed native plans
allocate their own state.

## CubeCL/CUDA Execution

When a compiled program is evaluated with `CudaBackend`, the same graph-level
ops run on GPU if supported:

- `dot_general` can route through CubeCL/CUDA and cuTENSOR/cuBLAS-backed paths,
- structural and reduction steps use CubeCL kernels where implemented,
- unsupported dtypes or operations return `BackendFailure`.

There is no current separate direct `CudaBackend` contraction pipeline. Future
GPU contraction improvements should extend `CudaBackend` and preserve the
explicit placement policy described in [gpu-backend-design.md](./gpu-backend-design.md).

## Planner Safety

Contraction planning must not rely on debug-only assertions for semantic
validity. Missing label sizes, invalid explicit paths, duplicate live operands,
and rank/shape mismatches are normal `Result` errors so debug and release builds
behave the same.
