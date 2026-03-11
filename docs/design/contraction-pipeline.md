# Binary Contraction Pipeline

This document details the binary contraction pipeline: how two tensors are
contracted in a single step of an N-ary contraction tree. It covers copy
elision, copy strategy experiments, and the recommended decomposition into
core primitives.

See [tensor-prims.md](./tensor-prims.md) for the semiring-core / fast-path
traits and
[einsum.md](./einsum.md) for how binary contractions fit into the N-ary
einsum engine.

---

## Historical Problem: Eager Reorder + BatchedGemm Is Suboptimal

When an einsum contraction tree is executed step by step, each step typically
involves:

1. Reordering input operands into GEMM-compatible layout
2. Executing GEMM
3. Reordering the output for the next step

In the old design, decomposing this into separate eager reordering and
`SemiringCoreDescriptor::BatchedGemm` calls forced materialization before the backend
could reason about layout. This was suboptimal because:

- **Unnecessary copies**: The contraction backend can often skip the copy
  entirely when strides are already compatible (`try_fuse_group` in strided-rs).
  A standalone reorder primitive cannot know this.

- **Wrong copy strategy**: When a copy is needed, there are two iteration
  orders — source-stride-order and destination-stride-order (HPTT). The
  optimal choice depends on cache state, which a standalone reorder primitive
  cannot know.

---

## Two Approaches

### Approach A: `Contract` Extended Operation

`SemiringFastPathDescriptor::Contract` fuses permutation and GEMM into a single primitive,
matching cuTENSOR's `cutensorContract`. It is an **extended operation**
(dynamically queried via `has_extension_for`) because not all backends need to
implement it.

The einsum layer queries `TensorSemiringFastPath::has_fast_path(...)`:
- **Available**: emit `Contract`, backend handles everything internally.
- **Not available**: fall back to core ops decomposition (Approach B).

Backend advantages:
- CPU: `try_fuse_group` copy elision, HPTT copy, global allocator reuse
- GPU: maps directly to `cutensorContract` (single kernel launch)

### Approach B: `permute` view + `MakeContiguous` + `BatchedGemm`

The argument for `Contract` assumes that decomposing reordering +
`BatchedGemm` forces materialization. This assumption breaks if we
distinguish **two kinds of permutation**:

- `Tensor::permute()`: metadata-only reordering (zero-copy, Tensor layer)
- `SemiringCoreDescriptor::MakeContiguous`: explicit materialization when a packed
  buffer is actually required

If the einsum layer uses `permute` views to reorder axes, then `BatchedGemm`
receives a `StridedView` with arbitrary strides. A `MakeContiguous` prim
bridges this gap.

---

## Recommended Pipeline (Approach B, for CPU)

```
1. Axis classification (einsum layer)
   Classify modes into batch, lo (left-output), ro (right-output), sum (contracted).

2. permute (Tensor layer, zero-copy)
   Reorder to canonical layout:
   A → [lo, sum, batch], B → [sum, ro, batch], C → [lo, ro, batch]
   Metadata-only: strides are reordered, no data movement.

3. MakeContiguous (Prims layer, conditional copy)
   Check full-tensor contiguity via try_fuse on all dimensions.
   - Contiguous (col-major or row-major): no-op, pass through.
   - Not contiguous: copy to col-major buffer.

4. BatchedGemm (Prims layer, contiguous input)
   Inputs are guaranteed contiguous. N/T is determined by stride layout:
   - stride[0] = 1 → col-major → CblasNoTrans
   - stride[-1] = 1 → row-major → CblasTrans
```

### `MakeContiguous` Prim Descriptor

```rust
SemiringCoreDescriptor::MakeContiguous
```

No parameters needed. The operation:
1. Checks if all elements are packed without gaps (either col-major or
   row-major)
2. If yes: no-op (the backend returns immediately)
3. If no: copies to col-major layout (tenferro convention)

Since `permute` already reorders axes into canonical order, full-tensor
contiguity implies group-level fusability. There is no need for per-group
`try_fuse_group` checks — a single full-tensor check suffices.

### Why This Works Without Copy Elision Loss

The key insight is that `permute` produces a non-contiguous `StridedView`
with reordered strides, not a materialized copy. `MakeContiguous` then checks
whether the data is already packed:

- **Intermediate results** from previous contraction steps are typically
  allocated as col-major buffers. After `permute`, the strides change but
  the data is still packed (just in row-major order relative to the new axis
  names). `MakeContiguous` detects this and skips the copy. This is equivalent
  to `try_fuse_group` succeeding in the current strided-einsum2 design.

- **Original input tensors** may have arbitrary strides (from slicing,
  broadcasting, etc.). These genuinely need copying, which `MakeContiguous`
  handles.

---

## GPU: Use `Contract` Extended Op

For GPU backends, `cutensorContract` accepts arbitrary strides natively
and fuses permutation + GEMM in a single kernel launch. The decomposed path
(Approach B) would require a separate contiguify kernel followed by GEMM —
two kernel launches with a synchronization point. For GPU backends, `Contract`
as an extended op remains preferable.

---

## Trade-offs

| Aspect | `Contract` (Approach A) | `permute` + `MakeContiguous` + `BatchedGemm` (Approach B) |
|---|---|---|
| Copy elision | Backend-internal `try_fuse_group` per group | Full-tensor contiguity check (slightly conservative) |
| Backend complexity | Must implement full contraction pipeline | Only `MakeContiguous` + `BatchedGemm` (simpler) |
| GPU compatibility | Maps directly to `cutensorContract` | Requires 2 kernel launches (contiguify + GEMM) |
| API surface | Extended op, dynamically queried | Core ops only, no extension mechanism needed |

**Conservative case**: When dimension groups are independently fusable but the
full tensor is not contiguous (gap between groups), `MakeContiguous` copies
unnecessarily. In practice this rarely occurs because intermediate results are
col-major allocated and `permute` preserves packed layout.

### Recommendation

- **CPU backend**: Use `permute` + `MakeContiguous` + `BatchedGemm`.
  Simpler implementation, no extended op needed, minimal performance gap.
- **GPU backend**: Use `Contract` extended op mapping to `cutensorContract`.
  Single kernel launch, no intermediate buffer.

---

## strided-einsum2 Six-Step Pipeline (Reference)

The current strided-einsum2 implementation uses a six-step pipeline for the
`Contract` extended operation on CPU:

```
1. Trace pre-reduction
   Sum out axes appearing only in one operand before GEMM.
   Conjugation materialized during reduce (conj flag -> false for GEMM).

2. Permutation to canonical order
   A[left, contracted, batch], B[contracted, right, batch], C[left, right, batch]
   Batch-last for column-major contiguity.

3. Element-wise bypass
   If contracted, left, and right are all empty (pure Hadamard product),
   call zip_map2_into instead of GEMM.

4. Fusability check (try_fuse_group)
   Test whether dimension groups can be fused into a single contiguous
   dimension without copying. Sorts (dim, stride) pairs by |stride|
   ascending and verifies stride[i] * dim[i] == stride[i+1].
   If fusable → zero-copy metadata extraction.
   If not → allocate col-major buffer and copy.

5. GEMM dispatch
   Call selected backend: faer::bgemm or cblas::dgemm/zgemm.
   Naive loop fallback for non-Scalar types (integers, tropical).

6. Copy-back
   If output was non-contiguous, copy from internal buffer back to
   the original strided destination.
```

Steps 1-4 are analyzed during `plan_contraction`. Steps 5-6 are executed
during `contract`.

### CPU Contraction Plan (for Contract extended op)

```rust
pub struct CpuContractionPlan {
    a_perm: Vec<usize>,
    b_perm: Vec<usize>,
    c_perm: Vec<usize>,
    a_fusable: bool,
    b_fusable: bool,
    batch_size: usize,
    left_size: usize,
    right_size: usize,
    contract_size: usize,
    gemm: GemmDispatch,  // Faer | Cblas | Naive
    workspace_size: usize,
    elementwise_bypass: bool,
    blocking: Option<BlockingPlan>,
}
```

---

## Copy Strategy Experiments

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

### Experiment Results

Three experiments characterize the copy strategy landscape:

1. **Flatten HPTT recursion** (`perf/flatten-hptt-recursion`): Replaced
   recursive ComputeNode traversal with flat odometer — **no improvement**
   (±3% noise). Recursion overhead is not a bottleneck.

2. **Source-order vs HPTT** (`perf/src-vs-dst-order`): With copy elision
   disabled, HPTT is 16–43% faster on most workloads (lm_*, str_*,
   mera_closed). Source-order is 27–30% faster only for degenerate
   many-small-dims cases (tn_focus/tn_light). On `mera_open`, the two
   strategies perform identically (±2%).

3. **Eager HPTT** (`perf/eager-hptt-permute`): Eagerly materializing all
   permutations via HPTT causes 26–31% regression on `mera_open` — entirely
   due to copy elision loss, not copy strategy.

### Benchmark Evidence (Lazy vs Eager)

Results on AMD EPYC 7713P (faer backend):

| Instance | Lazy 1T | Eager 1T | Regression |
|---|---|---|---|
| mera_open (opt_flops) | 918 ms | 1199 ms | **+31%** |
| mera_open (opt_size) | 918 ms | 1159 ms | **+26%** |
| tensor network instances | ~285 ms | ~287 ms | ~0% |

The `mera_open` regression is caused by eager permutation forcing copies at
every step, even when `try_fuse_group` would have skipped them.

### Conclusion

**Copy elision is the dominant optimization.** When copies are unavoidable,
HPTT is the better default thanks to cache-blocked tiling. An adaptive
strategy switching to source-order for degenerate many-small-dims cases
could provide the best of both worlds.

See `strided-rs/docs/src-vs-dst-order-experiment.md` and
`strided-rs/docs/eager-hptt-experiment.md` for full results.
