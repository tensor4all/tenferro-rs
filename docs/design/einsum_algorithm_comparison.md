# Einsum Algorithm Comparison: strided-rs vs omeinsum-rs

This document compares the contraction and permutation algorithm optimizations in
**strided-rs** (strided-einsum2 + strided-opteinsum) and **omeinsum-rs**, to guide
the "best of both" merge into **tenferro-prims** (binary contraction pipeline)
and **tenferro-einsum** (N-ary engine, algebra dispatch).

## 1. Binary Contraction Pipeline

### strided-rs (strided-einsum2)

Six-step reshape-to-GEMM pipeline:

1. **Trace reduction** — axes appearing only in one operand are summed out
   *before* GEMM via `reduce_trace_axes()`. Conjugation is materialized during
   the reduce when needed, so the conj flag passed to GEMM becomes `false`.

2. **Permutation to canonical order** — operands are reordered to
   ~~`A[batch, lo, sum]`, `B[batch, sum, ro]`, `C[batch, lo, ro]`~~
   `A[lo, sum, batch]`, `B[sum, ro, batch]`, `C[lo, ro, batch]` via
   `Einsum2Plan::new()`. (Fixed: batch-last for col-major contiguity.)

3. **Element-wise fast path** — when sum, lo, and ro are all empty (pure
   Hadamard product), bypass GEMM entirely and call `zip_map2_into()`.

4. **Fusability check** (`try_fuse_group`) — tests whether dimension groups
   (lo+sum, sum+ro) can be fused into a single contiguous dimension without
   copying. Sorts (dim, stride) pairs by |stride| ascending and verifies
   `stride[i] * dim[i] == stride[i+1]`. If fusable → zero-copy metadata
   extraction; if not → allocate col-major buffer and copy.

5. **GEMM dispatch** — calls `ActiveBackend::bgemm_contiguous_into()`.
   Backend is selected at compile time (faer, CBLAS, or naive).

6. **Copy-back** — if the output was non-contiguous, `finalize_into()` copies
   from the internal buffer back to the original strided view.

**Key features:**
- Fusability checking avoids copies when strides are contiguous.
- Owned-input path (`einsum2_into_owned`) transfers ownership to the operand
  buffer, avoiding a separate allocation when the array is already fusable.
- `BackendConfig` trait declares per-backend requirements
  (`MATERIALIZES_CONJ`, `REQUIRES_UNIT_STRIDE`) so operand preparation adapts
  without per-call `cfg` checks.

### omeinsum-rs

Simpler matricization pipeline:

1. **Mode classification** — batch, left, right, contracted.
2. **Ensure contiguous** — always copies if strided (no fusability check).
3. **Permute to canonical order** — `A[left, contracted, batch]`,
   `B[contracted, right, batch]`. Batch-last layout ensures each batch slice
   is contiguous in column-major memory.
4. **GEMM dispatch** — non-batched or batched (loop of regular GEMMs).
5. **Output permutation** — result permuted from `[left, right, batch]` to
   requested output order.

**Key features:**
- Always-materialize strategy: inputs always made contiguous before GEMM.
  Simpler code, but more memory copies.
- No fusability checking — every contraction goes through full
  permute → GEMM → permute.
- `TypeId`-based runtime dispatch to specialized kernels (tropical-gemm).

### Comparison

| Aspect | strided-rs | omeinsum-rs | Recommendation (→ crate) |
|--------|-----------|-------------|--------------------------|
| Copy avoidance | Fusability check (`try_fuse_group`) | Always copy | **Adopt** fusability check (→ `PrimDescriptor::Contract` CPU impl) |
| Element-wise bypass | `zip_map2_into` for Hadamard | Goes through GEMM | **Adopt** as `PrimDescriptor::ElementwiseMul` |
| Trace pre-reduction | Before GEMM, with conj materialization | After classification, as sum-over | **Adopt** as `TensorPrims::trace` (core universal set) |
| Owned-input optimization | Transfers ownership → zero-copy | Always allocates new buffer | **Adopt** ownership transfer (→ tenferro-einsum) |
| Backend requirements | Compile-time `BackendConfig` trait | Hardcoded | **Adopt** trait-based config (→ `TensorPrims::batched_gemm` CPU impl) |
| Tropical dispatch | Not supported | `TypeId` runtime dispatch | **Adopt** omeinsum approach (→ `TensorPrims::batched_gemm` dispatch) |
| Batch placement | ~~Batch-first `[batch, lo, sum]`~~ Fixed to batch-last | Batch-last `[left, contracted, batch]` | Both now batch-last |

## 2. N-ary Einsum (Contraction Tree)

### strided-rs (strided-opteinsum)

- **Recursive `eval_node()`** with borrowed-view passthrough:
  - Leaf nodes return operands as borrowed views (no copy).
  - Contract nodes with 1 child: identity passthrough or permutation-only
    (metadata-only, zero-copy).
  - Contract nodes with 2 children: call `eval_pair()`.
  - Contract nodes with 3+ children: build `omeco::EinCode`, run greedy
    optimizer, execute nested tree.

- **Buffer pool** (`BufferPool`): HashMap-indexed free lists keyed by buffer
  size. Freed buffers are returned to the pool after each pairwise contraction;
  subsequent intermediates reuse them.

- **Final contraction writes directly into user's output** via
  `execute_nested_into()` — no extra allocation for the root node.

### omeinsum-rs

- **Recursive `execute_tree()`**:
  - Leaf nodes return `tensor.clone()` (Arc clone, cheap).
  - Node (binary): recurse left and right, then `contract_binary()`.
  - No distinction between 1-child and 2-child nodes.

- **No buffer pool** — each binary contraction allocates a new result tensor.
  Arc-wrapped storage allows cheap tensor cloning but no reuse of freed
  intermediates.

- **Fallback path** (`execute_pairwise`): contracts left-to-right when no
  optimization is performed.

### Comparison

| Aspect | strided-rs | omeinsum-rs | Recommendation (→ crate) |
|--------|-----------|-------------|--------------------------|
| Borrowed-view passthrough | Yes (Leaf → borrow) | No (Leaf → Arc clone) | **Adopt** borrowed views (→ tenferro-einsum) |
| Permutation-only detection | Yes (metadata-only) | No | **Adopt** permutation detection (→ tenferro-einsum, uses zero-copy `Tensor::permute`) |
| Buffer pool | HashMap by size | None | **Adopt** as opt-in option (→ tenferro-einsum) |
| Root writes into user output | Yes (`execute_nested_into`) | No | **Adopt** direct root write (→ tenferro-einsum) |
| Contraction optimizer | omeco greedy | omeco greedy + TreeSA | **Adopt** both optimizers (→ tenferro-einsum) |
| Unoptimized fallback | 3+ child → inline optimize | Left-to-right pairwise | Either acceptable |
| Operation decomposition | Explicit diag→trace→permute→GEMM | Monolithic contract + general unary loop | **Adopt** strided-rs explicit decomposition via core `TensorPrims` |

## 3. Single-Tensor Operations

### strided-rs (strided-opteinsum `single_tensor.rs`)

Five-step pipeline with two zero-allocation fast paths:

1. **Full trace** (`"ii->"` or `"ijk,ijk->"` with all same index):
   Single loop over diagonal using `diag_stride = sum of all strides`.
   No allocation, single pass.

2. **Partial trace** (`"iij->j"`): Detect one repeated pair, loop directly
   over those two axes' diagonal with other axes iterated normally.
   No allocation, single pass.

3. **General case**: `reduce_axis()` calls for each summed axis, then permute.

4. **Repeat (broadcast)**: stride-0 view for new dimensions, then materialize.

5. **Duplicate** (`"i->ii"`): write to all matching diagonal positions.

### omeinsum-rs

Single function `execute_unary_naive()`:

1. **Index classification**: outer (output) vs inner (summed-over).
2. **Trace handling**: repeated output indices extract diagonal via
   equality check in nested loop.
3. **Summation**: direct nested loop over inner dimensions.
4. **No fast paths**: all single-tensor ops go through the general loop.

### Comparison

| Aspect | strided-rs | omeinsum-rs | Recommendation (→ crate) |
|--------|-----------|-------------|--------------------------|
| Full trace fast path | Yes (single loop, no alloc) | No | **Adopt** as `TensorPrims::trace` (core universal set) |
| Partial trace fast path | Yes (single loop, no alloc) | No | **Adopt** via `diag` + `TensorPrims::reduce` decomposition |
| General reduce | `reduce_axis()` per axis | Nested loop | **Adopt** as `TensorPrims::reduce` (core universal set) |
| Broadcast/repeat | Stride-0 view + copy | Not needed (einsum semantics) | **Adopt** as zero-copy `Tensor::broadcast` + `repeat` adjoint pair with `reduce` |
| Anti-diagonal (i→ii) | `single_tensor.rs` duplicate | Not implemented | **Adopt** as `TensorPrims::anti_diag` (AD adjoint of diag) |

## 4. Permutation Strategy

### strided-rs

- **Zero-copy views**: `StridedArrayView::permute()` reorders dims/strides
  in metadata only. The underlying data pointer is unchanged.
- **Fusability-aware materialization**: only copies when GEMM backend requires
  contiguous data *and* the dimension group is non-fusable.
- **`StridedView::clone()` is metadata-only** (cheap) — used for
  permute-only paths to avoid double copy.

### omeinsum-rs

- **Zero-copy permute via `Arc<Storage>`**: `Tensor::permute()` creates a new
  `Tensor` with modified shape/strides but shared `Arc<Storage>`. No data copy
  until GEMM needs contiguous input.
- **`ensure_contiguous()`**: copies when `!is_contiguous()`.
- **`permute_data()`**: always allocates new memory for physical reordering.

### Comparison

Both use zero-copy views for permutation. strided-rs's advantage is the
fusability check that can avoid materializing even when the view is
technically non-contiguous (but strides happen to be dense within dimension
groups).

## 5. GPU Strategy

### strided-rs

No GPU support currently. The design document (tenferro unified backend) plans:
- cuTENSOR/hipTensor for einsum (via `TensorPrims` trait, runtime dlopen).

### omeinsum-rs

- **cuTENSOR integration**: bypasses reshape-to-GEMM entirely, passes strides
  and modes directly to the cuTENSOR library.
- **Plan caching**: `PlanCache` with `HashMap<CacheKey, Plan>` avoids
  re-creating cuTENSOR plans for repeated contractions.
- **cudarc** for device memory management.

### Recommendation

Adopt omeinsum-rs's cuTENSOR integration pattern into **tenferro-device** (vtable,
plan caching) and **tenferro-prims** (`TensorPrims` GPU impl):
- `TensorPrims` trait wraps cuTENSOR's direct-contraction API via tenferro-device vtable.
- Plan caching for repeated contractions (in tenferro-device).
- Dispatch priority: GPU TensorPrims > CPU TensorPrims (GEMM) > naive loop.

## 6. Algebra Extensibility

### strided-rs

- `BgemmBackend<T>` trait: external crates can implement for custom scalar
  types (e.g., tropical semiring).
- Compile-time backend selection via `ActiveBackend` type alias and Cargo
  features.
- No built-in tropical support.

### omeinsum-rs

- `Algebra` trait with `zero()`, `add()`, `mul()`, `to_scalar()` methods.
- `TypeId`-based runtime dispatch to specialized SIMD kernels (tropical-gemm).
- Built-in tropical types: `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`.
- Argmax tracking for tropical backward pass.

### Recommendation

- Adopt the `Algebra` trait design for semiring extensibility (→ tenferro-algebra).
- Keep `BgemmBackend<T>` for pluggable GEMM backends (→ tenferro-prims).
- Support `TypeId`-based dispatch for performance-critical hot paths (→ tenferro-einsum).
- Argmax tracking as an opt-in feature (→ tenferro-algebra).

**Converged design**: The algebra trait is split across two crates:
- `tenferro-algebra` (POC): `HasAlgebra` trait (T → A mapping), `Semiring` trait,
  `Standard` type. Minimal foundation.
- `tenferro-tropical` (separate crate): `MaxPlus<T>`, `MinPlus<T>`, `MaxMul<T>`,
  `impl TensorPrims<MaxPlus> for CpuBackend`. Being external proves the
  extensibility pattern (orphan rule: MaxPlus is local to tenferro-tropical).

## 7. Summary: Best-of-Both for tenferro-prims + tenferro-einsum

> strided-einsum2 (binary contraction pipeline) → **tenferro-prims**
> strided-opteinsum + omeinsum-rs (N-ary engine) → **tenferro-einsum**

### TensorPrims<A> Architecture

GiggleLiu proposed a **universal set** of primitive operations for
`tenferro-prims` that synthesizes the best of both codebases.
The converged design uses a single `TensorPrims<A>` trait parameterized
by algebra `A`, with a cuTENSOR-compatible plan-based execution model
(`PrimDescriptor → plan → execute`) and dynamic extension queries:

**Core ops (universal set, required for all backends)**:

| # | Operation | Origin | strided-rs | omeinsum-rs | CPU impl | GPU impl |
|---|-----------|--------|-----------|-------------|----------|----------|
| 1 | `batched_gemm` | strided-rs | `bgemm_contiguous_into` | `generic_gemm` / `faer_gemm` | faer/cblas | cuTENSOR contract |
| 2 | `reduce` | Both | `reduce_axis` | `sum_axis` | strided-kernel | cuTENSOR reduce |
| 3 | `trace` | strided-rs | `reduce_trace_axes` | `execute_unary_naive` (general) | strided-kernel loop | cuTENSOR reduce on diagonal |
| 4 | `permute` | Both | `StridedView::permute` + copy | `permute_data` | strided-kernel copy | cuTENSOR permute |
| 5 | `anti_trace` | **New** (AD) | — | — | scatter-add loop | custom kernel |
| 6 | `anti_diag` | **New** (AD) | — | — | write-to-diagonal loop | custom kernel |

Note: `diag` and `repeat` are zero-copy stride tricks on `Tensor<T>`,
not in `TensorPrims` (no computation needed).

**Extended ops (optional, dynamically queried via `has_extension_for`)**:

| # | Operation | Origin | Description |
|---|-----------|--------|-------------|
| 1 | `contract` | omeinsum-rs + strided-rs | Fused permute + batched_gemm + unpermute (cuTENSOR's `cutensorContract`) |
| 2 | `elementwise_mul` | strided-rs | Hadamard product bypass (strided-rs's `zip_map2_into`) |

**Dispatch rule**: If `has_extension_for::<T>(Extension::Contract)` returns
`true` → use the fused operation via `PrimDescriptor::Contract`. Otherwise →
`tenferro-einsum` decomposes into core ops:
`diag → trace/reduce → permute → batched_gemm → permute`.

**Adjoint pairs** for AD: `trace ↔ anti_trace`, `diag ↔ anti_diag`,
`reduce ↔ repeat`, `permute ↔ inverse permute`, `batched_gemm` uses
Leibniz rule.

### From strided-rs (adopt into core ops)

1. **`batched_gemm`** — raw batched GEMM, the core computation.
   (→ `TensorPrims::batched_gemm`)
2. **`trace`** — `reduce_trace_axes()` for summing paired dimensions.
   (→ `TensorPrims::trace`)
3. **`reduce`** — `reduce_axis()` for summing unpaired dimensions.
   (→ `TensorPrims::reduce`)
4. **`permute`** — `StridedView::permute()` + physical copy.
   (→ `TensorPrims::permute`)
5. **Cache-optimized kernels** — `reduce_axis()` uses blocked, dimension-fused
   iteration from strided-kernel. (→ tenferro-prims via strided-kernel)

### From strided-rs (adopt into extended ops)

1. **Fusability checking** (`try_fuse_group`) — avoid unnecessary copies when
   dimension groups are already contiguous. (→ Contract extended op on CPU)
2. **Element-wise bypass** — skip GEMM for Hadamard products.
   (→ ElementwiseMul extended op)

### From strided-rs (adopt into tenferro-einsum)

1. **Owned-input optimization** — transfer ownership to avoid allocation.
2. **Buffer pool** (opt-in) — reuse intermediate buffers across pairwise
    contractions. Opt-in because it increases peak memory usage.
3. **Borrowed-view passthrough** — Leaf nodes return borrows, not clones.
4. **Permutation-only detection** — metadata-only transformation in
   contraction tree.
5. **Single-tensor fast paths** — full trace and partial trace zero-allocation
   loops (now decomposed into `TensorPrims::trace`).
6. **Direct root write** — final contraction writes into user's output buffer.

### From omeinsum-rs (adopt)

1. **Algebra trait** — semiring-generic `zero()`, `add()`, `mul()` interface.
   (→ tenferro-algebra: HasAlgebra, Semiring, Standard)
2. **Tropical-gemm dispatch** — `TypeId`-based runtime specialization for SIMD
   tropical kernels. (→ tenferro-tropical: `TensorPrims<MaxPlus> for CpuBackend`)
3. **Argmax tracking** — tropical backward pass support. (→ tenferro-tropical)
4. **cuTENSOR integration** — direct contraction without reshape-to-GEMM,
   plan caching. (→ Contract extended op on GPU + tenferro-device)
5. **TreeSA optimizer** — simulated annealing for better contraction orders
   (in addition to greedy). (→ tenferro-einsum)
6. **Column-major + batch-last layout** — both codebases now use batch-last
   (strided-rs bug fixed in PR #87).

### New (neither codebase)

1. **`TensorPrims<A>` algebra-parameterized trait** — core universal set +
   dynamically-queried extended ops, plan-based execution. (→ tenferro-prims)
2. **`anti_trace` / `anti_diag`** — AD adjoint operations for trace and
   diagonal. (→ core ops in `TensorPrims<A>`)
3. **`tenferro-tropical` as separate crate** — proves extensibility of
   algebra-parameterized design. (→ tenferro-tropical)
4. **Adjoint pair documentation** — clean VJP/JVP rules for each primitive.
   (→ tenferro-autograd)
5. **Global + per-call override** — `einsum()` uses global default,
   `einsum_with()` accepts explicit config. (→ tenferro-einsum)
6. **Custom scalar extensibility tests** — `ModInt<P>` test type to verify
   all dispatch tiers.
