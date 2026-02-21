# Einsum CPU Backend: Implementation Notes

This document details how the einsum CPU implementation is structured in
`tenferro-prims` and `tenferro-einsum`.

**Purpose**: Detect potential problems before implementation begins.

## Reference Implementations

The algorithms in `strided-einsum2`, `strided-opteinsum`, and `omeinsum-rs`
serve as **reference material** for the tenferro implementation. All three
are scheduled for deprecation — tenferro-rs becomes the sole einsum
implementation.

| Reference | Maps to | Status |
|---|---|---|
| `strided-einsum2` | tenferro-prims (binary contraction pipeline) | Deprecated |
| `strided-opteinsum` | tenferro-einsum (N-ary tree, parsing) | Deprecated |
| `omeinsum-rs` | tenferro-prims + tenferro-einsum (unified reference) | Deprecated |

**Surviving strided-rs dependencies**: `strided-traits`, `strided-view`,
`strided-kernel` remain as foundation dependencies. Only the einsum
layers are absorbed.

---

## Implementation Principles

1. **tenferro public API is preserved** — the existing 9 functions + AD API in
   `tenferro-einsum` remain unchanged (except where noted: P8, P9, P11).
2. **Explicit context passing** — all einsum functions receive
   `&mut B::Context` where `B: TensorPrims<A>`.  No global/thread-local
   state.  Follows Rust idiom of explicit ownership and mutability.
3. **strided-view, strided-kernel, strided-traits remain as dependencies** —
   tenferro-prims uses them internally.
4. **Self-contained implementation** — algorithms are re-implemented inside
   tenferro, referencing strided-einsum2 and omeinsum-rs for design but
   not depending on them.
5. **Backend\<T\> is absorbed into TensorPrims\<A\>** — faer/blas/naive selection
   happens inside CpuBackend, not via a separate trait.
6. **No type erasure (EinsumOperand)** — tenferro uses generic `Tensor<T>`
   throughout.
7. **All computation goes through TensorPrims** — einsum is a composite
   operation that orchestrates calls to `TensorPrims::plan()`/`execute()`.
   No direct use of strided-kernel or GEMM from the einsum layer.

---

## Layer 1: tenferro-prims — Binary Contraction Engine

### Function Mapping

Reference sources: `strided-einsum2` and `omeinsum-rs::backend`.

| Reference source | tenferro-prims target | API change | Notes |
|---|---|---|---|
| `Einsum2Plan::new(ia, ib, ic)` / omeinsum `contract()` mode classification | Internal type in tenferro-prims | `AxisId` → `u32` fixed | Classify batch/free/contracted modes |
| `einsum2_into(...)` / omeinsum `contract()` | `TensorPrims::execute()` Contract path | `StridedView` → converted from `Tensor<T>` | Main execution path |
| `einsum2_naive_into(...)` | `CpuBackend` fallback path | `map_a`/`map_b` closures → removed (Conjugate trait) | For custom scalar types |
| `Backend<T>` trait / omeinsum `Backend` trait | Removed → absorbed into `TensorPrims<A>` | — | faer/blas selected via feature flags |
| `FaerBackend` / `BlasBackend` / `NaiveBackend` | Inside `CpuBackend` | Feature-gated | |
| `validate_dimensions(...)` | Internal function in tenferro-prims | As-is | |
| `einsum2_dispatch(...)` / omeinsum GEMM pipeline | `CpuBackend::execute()` internals | permute → contiguous → GEMM | Core execution pipeline |
| trace.rs: `find_trace_indices` / `reduce_trace_axes` | `PrimDescriptor::Trace` / `PrimDescriptor::Reduce` | Via TensorPrims API | strided-kernel dependency |
| contiguous.rs: `ContiguousOperand` / `prepare_input_view` | `PrimDescriptor::MakeContiguous` | New PrimDescriptor variant | |
| bgemm_naive.rs: `bgemm_strided_into` | `CpuBackend` internal | unsafe pointer arithmetic | See P2 |
| bgemm_faer.rs | `CpuBackend` internal (feature: `faer`) | | |
| bgemm_blas.rs | `CpuBackend` internal (feature: `blas`) | | |

### AntiTrace / AntiDiag CPU Implementation

These are core prims (every backend must implement). On CPU, they are
implemented as **simple loops** with no external dependency:

```rust
// anti_trace: ∂C[j,k] → ∂A[i,j,i,k] = δ(i,i') × ∂C[j,k]
// Output zero-initialized, then scatter-add to diagonal
for i in 0..I {
    for j in 0..J {
        for k in 0..K {
            output[i, j, i, k] += grad[j, k];
        }
    }
}

// anti_diag: ∂C[i,j] → ∂A[i,i,j] = δ(i,i') × ∂C[i,j]
// Output zero-initialized, then write to diagonal
for i in 0..I {
    for j in 0..J {
        output[i, i, j] = grad[i, j];
    }
}
```

No dependency on strided-einsum2, Contract, or any einsum-level logic.
GPU backends may compose these via `Contract(eye, ∂C)` using
cuTENSOR/hipTENSOR — see `gpu-backend-design.md`.

### Type Adaptation

**Tensor\<T\> → StridedView conversion** happens inside CpuBackend:

```rust
// Conceptual code in CpuBackend::execute()
fn tensor_to_strided_view<T: Scalar>(t: &Tensor<T>) -> StridedView<T> {
    StridedView::from_raw_parts(
        t.buffer().as_slice(),
        t.dims(),
        t.strides(),
        t.offset(),
    )
}
```

**Conjugation**: strided-einsum2 uses type-level dispatch via `ElementOp<T>`
(Identity/Conj/Adjoint/Transpose marker types) to avoid materializing
conjugation.  tenferro uses the `Conjugate` trait (method-level dispatch).
In CpuBackend, convert to strided-view's `Conj` op when building
`StridedView`, preserving lazy conjugation internally.

On GPU, cuTENSOR supports `CUTENSOR_OP_CONJ` in tensor descriptors for
lazy conjugation. Standalone `Tensor::conj()` requires CPU transfer for
GPU tensors. See `gpu-backend-design.md` G11.

**Algebra parameterization**: strided-einsum2 has no algebra concept.
`TensorPrims<A>` is parameterized by algebra `A`.  For `Standard<S>` algebra,
use faer/blas.  For non-`Standard<S>` algebras (tropical, etc.), fall back to
the naive path (see P5).

---

## Layer 2: tenferro-einsum — N-ary Frontend

### Function Mapping

Reference sources: `strided-opteinsum` and `omeinsum-rs::einsum`.

| Reference source | tenferro-einsum target | API change | Notes |
|---|---|---|---|
| `parse_einsum(s)` / omeinsum `Einsum::new()` | `Subscripts::parse(notation)` + `ContractionTree` | `EinsumCode` split into two types | See P8 |
| `EinsumNode` / omeinsum `NestedEinsum` | `ContractionTree` internal representation | Not public | |
| `einsum(notation, operands, size_dict)` / omeinsum `Einsum::execute()` | `einsum(ctx, subscripts, operands)` | Add `ctx: &mut B::Context` | See P9, P11 |
| `einsum_with_pool(...)` | `einsum(...)` internals | Pool removed; global allocator handles reuse | — |
| `einsum_into(...)` | `einsum_into(ctx, subscripts, operands, α, β, output)` | Add `ctx: &mut B::Context` | See P11 |
| `EinsumCode::evaluate(...)` / omeinsum tree evaluation | `einsum_with_plan(ctx, tree, operands)` | Add `ctx: &mut B::Context` | See P11 |
| `BufferPool` (expr.rs) | Removed | — | Global allocator (mimalloc/jemalloc) handles reuse |
| `EinsumOperand` (F64/C64 enum) | Removed | — | Generic `Tensor<T>` replaces this |
| `single_tensor_einsum(...)` / omeinsum `execute_unary_naive()` | Internal function in tenferro-einsum | Via TensorPrims | See P7 |
| parse.rs | `Subscripts::parse()` + tree extraction | Parenthesized notation | See P8 |
| omeinsum backward (`contract_unary_backward`, `contract_binary_backward`) | tenferro-einsum AD functions | Via TensorPrims | Unary backward = einsum with swapped indices |

### Type Adaptation

**EinsumOperand → Tensor\<T\>**: strided-opteinsum dispatches at runtime via
`EinsumOperand` (F64/C64 enum).  tenferro resolves at compile time via
generic `Tensor<T>`.

**EinsumCode → Subscripts + ContractionTree**: strided-opteinsum's
`EinsumCode { root: EinsumNode, output_ids }` stores tree structure and
labels in one type.  tenferro separates into `Subscripts` (label info) and
`ContractionTree` (tree structure).  The parser must produce both.  See P8.

**omeco dependency**: `ContractionTree::optimize()` uses the omeco crate for
greedy N-ary contraction tree optimization.  Must add `omeco` to
tenferro-einsum's `Cargo.toml`.  See P10.

---

## Cross-Cutting Concerns

### Allocation Strategy

No custom `BufferPool`. Intermediate buffer allocation (GEMM contiguous
buffers, N-ary einsum intermediates) uses standard `Vec<T>` with the
global allocator. A performant global allocator (mimalloc, jemalloc) provides
thread-local free lists and size-class caching, achieving similar reuse
without manual pool management.

```
CpuContext
  └── PlanCache           (keyed by PrimDescriptor + shapes)
```

GPU backends may need a device-memory pool (cudaMalloc is expensive),
but that is separate from CPU allocation.

### Unsafe Code Inventory

All unsafe code is confined to `CpuBackend` internals.  No unsafe in public API.

| Origin file | Content | Target | Risk |
|---|---|---|---|
| `bgemm_naive.rs` | Pointer offset read/write (innermost loop) | `CpuBackend` internal | Low: well-tested in strided-rs |
| `contiguous.rs` | `Vec::set_len()` (uninit buffer) | `CpuBackend` internal | Medium: caller must write all elements |
| `contiguous.rs` | `StridedArray::col_major_from_buffer_uninit()` | `CpuBackend` internal | Low: strided-view API |
| `bgemm_blas.rs` | FFI calls (`cblas_sys::dgemm` etc.) | `CpuBackend` (blas feature) | Low: standard FFI |
| `bgemm_blas.rs` | `extern "C" fn` fallback registration | `CpuBackend` (blas-inject feature) | Low |

### Feature Flags and Dependencies

**New dependencies to add**:

| Crate | Add to | Purpose | Feature gate |
|---|---|---|---|
| `omeco` | tenferro-einsum | N-ary contraction tree optimization | None (required) |
| `faer` + `faer-traits` | tenferro-prims | CPU GEMM | `faer` (default) |
| `cblas-sys` | tenferro-prims | BLAS FFI | `blas` |
| `cblas-inject` | tenferro-prims | BLAS fallback | `blas-inject` |
| `rayon` | tenferro-prims | Parallelization | `parallel` |
| `strided-kernel` | tenferro-prims | map/reduce/copy kernels | None (required) |
| `strided-perm` | tenferro-prims | Stride fusability check | None (required) |
| `libloading` | tenferro-prims | GPU library loading | None (always on) |

All feature-gated crates are defined in workspace `Cargo.toml` under
`[workspace.dependencies]`.  Feature propagation from tenferro-einsum to
tenferro-prims is not needed (einsum calls prims only via the `TensorPrims`
trait).

---

## Problem Catalog

| ID | Problem | Severity | Resolution |
|---|---|---|---|
| ~~P1~~ | ~~MakeContiguous placement~~ | — | **Resolved**: Add `PrimDescriptor::MakeContiguous` |
| P2 | Unsafe code (bgemm_naive, contiguous) | Medium | Confine to CpuBackend internals.  Review during copy |
| P3 | Conjugation dispatch change | Medium | CPU: use strided-view's `Conj` op internally when building StridedView. GPU: lazy conjugation via cuTENSOR `CUTENSOR_OP_CONJ`. Standalone `conj()` requires CPU transfer for GPU tensors |
| ~~P4~~ | ~~Thread-local buffer pool~~ | — | ~~Merged into P6~~ → Removed (global allocator) |
| P5 | Algebra-specific dispatch (tropical, etc.) | Medium | `Standard<S>` → faer/blas.  Non-`Standard<S>` → naive fallback.  Dispatch on `A: Semiring` bounds |
| ~~P6~~ | ~~Unified BufferPool~~ | — | **Removed**: Global allocator (mimalloc/jemalloc) handles intermediate buffer reuse. No custom pool needed for CPU. GPU device-memory pool is separate |
| **P7** | **single_tensor_einsum via prims** | **Medium** | Route all steps (diag, trace, permute, broadcast, anti-diag) through `TensorPrims`.  Cache plans in `CpuContext.PlanCache` keyed by `(PrimDescriptor, shapes)` to avoid repeated plan generation |
| **P8** | **Parenthesized tree notation** | **Medium** | Support parenthesized contraction order in parser.  Parser returns `Subscripts` + `Option<ContractionTree>`.  POC skeleton needs update |
| **P9** | **Generative output (size_dict)** | **Medium** | Support `size_dict` for output labels not present in inputs.  Add parameter to tenferro-einsum public API.  POC skeleton needs update |
| P10 | omeco dependency | Low | Add to workspace dependencies |
| **P11** | **Explicit context passing in einsum API** | **High** | All einsum functions gain `ctx: &mut B::Context` and `B: TensorPrims<A>` type parameter.  Follows Rust idiom (explicit `&mut`, no global state).  All 9 einsum functions + AD functions need signature changes.  See revised signatures below |
| **P12** | **strided-einsum2 / omeinsum-rs deprecation** | — | Both deprecated. Algorithms are reference material, not dependencies. tenferro-prims and tenferro-einsum contain self-contained implementations |
| P13 | AntiTrace/AntiDiag CPU implementation | Low | Simple loops (scatter-add / write-to-diagonal). No dependency on strided-einsum2 or Contract. GPU uses Contract(eye, ∂C) composition |

### Revised einsum Signatures (P11)

All einsum public functions change from:

```rust
pub fn einsum<T: Scalar + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>
```

to:

```rust
pub fn einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    A: Semiring,
    B: TensorPrims<A>,
```

This applies to all 9 variants (einsum, einsum_with_subscripts,
einsum_with_plan, einsum_into, einsum_owned, etc.) and all AD functions
(tracked_einsum, dual_einsum, einsum_rrule, einsum_frule, einsum_hvp).

`ctx` provides access to `PlanCache` (P7).
`B` enables compile-time backend selection without global state.

### POC Skeleton Changes Required

The following changes to the existing API skeleton are needed before
implementation:

1. **`PrimDescriptor` enum** — add `MakeContiguous` variant (P1).
2. **`CpuContext` struct** — add `PlanCache` field (P7).
3. **`Subscripts::parse()`** — return type or companion function for
   parenthesized tree extraction (P8).
4. **einsum public functions** — add optional `size_dict` parameter for
   generative output dimensions (P9).
5. **einsum public functions** — add `ctx: &mut B::Context` parameter and
   `B: TensorPrims<A>` type parameter to all 9 einsum functions + all AD
   functions (P11).
