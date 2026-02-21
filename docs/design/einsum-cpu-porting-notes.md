# Einsum CPU Backend: Porting Notes (strided-rs → tenferro-rs)

This document details how to port the einsum implementation from
`strided-einsum2` and `strided-opteinsum` (in the strided-rs workspace) into
`tenferro-prims` and `tenferro-einsum` (in the tenferro-rs workspace), using
the CPU backend.

**Purpose**: Detect potential problems before implementation begins.

## Porting Principles

1. **tenferro public API is preserved** — the existing 9 functions + AD API in
   `tenferro-einsum` remain unchanged (except where noted: P8, P9, P11).
2. **Explicit context passing** — all einsum functions receive
   `&mut B::Context` where `B: TensorPrims<A>`.  No global/thread-local
   state.  Follows Rust idiom of explicit ownership and mutability.
3. **strided-view, strided-kernel, strided-traits remain as dependencies** —
   tenferro-prims uses them internally.
4. **strided-einsum2 and strided-opteinsum logic is copied** — not used as
   dependencies; algorithms are re-implemented inside tenferro.
5. **Backend\<T\> is absorbed into TensorPrims\<A\>** — faer/blas/naive selection
   happens inside CpuBackend, not via a separate trait.
6. **No type erasure (EinsumOperand)** — tenferro uses generic `Tensor<T>`
   throughout.
7. **All computation goes through TensorPrims** — einsum is a composite
   operation that orchestrates calls to `TensorPrims::plan()`/`execute()`.
   No direct use of strided-kernel or GEMM from the einsum layer.

---

## Layer 1: tenferro-prims — Binary Contraction Engine

The bulk of `strided-einsum2` maps here.

### Function Mapping

| strided-einsum2 source | tenferro-prims target | API change | Notes |
|---|---|---|---|
| `Einsum2Plan::new(ia, ib, ic)` | Internal type in tenferro-prims | `AxisId` → `u32` fixed | Axis classification logic copied as-is |
| `einsum2_into(c, a, b, ic, ia, ib, α, β)` | `TensorPrims::execute()` BatchedGemm path | `StridedView` → converted from `Tensor<T>` | Main execution path |
| `einsum2_into_owned(...)` | `TensorPrims::execute()` owned optimization | `StridedArray` → `Tensor<T>` (owned) | Buffer reuse optimization |
| `einsum2_naive_into(...)` | `CpuBackend` fallback path | `map_a`/`map_b` closures → removed (Conjugate trait) | For custom scalar types |
| `einsum2_with_backend_into(...)` | Removed | — | `TensorPrims<A>` replaces this |
| `Backend<T>` trait | Removed → absorbed into `TensorPrims<A>` | — | faer/blas selected via feature flags |
| `FaerBackend` / `BlasBackend` / `NaiveBackend` | Inside `CpuBackend` | Feature-gated | |
| `validate_dimensions(...)` | Internal function in tenferro-prims | As-is | |
| `einsum2_dispatch(...)` | `CpuBackend::execute()` internals | permute → contiguous → GEMM | Core execution pipeline |
| trace.rs: `find_trace_indices` / `reduce_trace_axes` | `PrimDescriptor::Trace` / `PrimDescriptor::Reduce` | Via TensorPrims API | strided-kernel dependency |
| contiguous.rs: `ContiguousOperand` / `prepare_input_view` | `PrimDescriptor::MakeContiguous` | New PrimDescriptor variant | |
| contiguous.rs: `prepare_output_view` / `finalize_into` | `PrimDescriptor::MakeContiguous` | Same | |
| bgemm_naive.rs: `bgemm_strided_into` | `CpuBackend` internal | unsafe pointer arithmetic copied | See P2 |
| bgemm_faer.rs | `CpuBackend` internal (feature: `faer`) | | |
| bgemm_blas.rs | `CpuBackend` internal (feature: `blas`) | | |

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

**Algebra parameterization**: strided-einsum2 has no algebra concept.
`TensorPrims<A>` is parameterized by algebra `A`.  For `Standard` algebra,
use faer/blas.  For non-`Standard` algebras (tropical, etc.), fall back to
the naive path (see P5).

---

## Layer 2: tenferro-einsum — N-ary Frontend

The bulk of `strided-opteinsum` maps here.

### Function Mapping

| strided-opteinsum source | tenferro-einsum target | API change | Notes |
|---|---|---|---|
| `parse_einsum(s) → EinsumCode` | `Subscripts::parse(notation)` + `ContractionTree` | `EinsumCode` split into two types | See P8 |
| `EinsumNode` (Leaf/Contract) | `ContractionTree` internal representation | Not public | |
| `einsum(notation, operands, size_dict)` | `einsum(ctx, subscripts, operands)` | `EinsumOperand` → `&Tensor<T>`, size_dict → separate param, add `ctx: &mut B::Context` | See P9, P11 |
| `einsum_with_pool(...)` | `einsum(...)` internals | Pool accessed via `ctx` | See P6 |
| `einsum_into(notation, operands, output, α, β, ...)` | `einsum_into(ctx, subscripts, operands, α, β, output)` | Add `ctx: &mut B::Context` | See P11 |
| `einsum_into_with_pool(...)` | `einsum_into(...)` internals | Pool accessed via `ctx` | |
| `EinsumCode::evaluate(...)` | `einsum_with_plan(ctx, tree, operands)` | Add `ctx: &mut B::Context` | See P11 |
| `EinsumCode::evaluate_into(...)` | `einsum_with_plan_into(ctx, tree, operands, α, β, output)` | Add `ctx: &mut B::Context` | See P11 |
| `BufferPool` (expr.rs) | Unified pool in `CpuContext` | Type-erased, generic | See P6 |
| `EinsumOperand` (F64/C64 enum) | Removed | — | Generic `Tensor<T>` replaces this |
| `EinsumScalar` trait | Removed | — | `Scalar + HasAlgebra` replaces this |
| `StridedData` (Owned/View enum) | Removed or `Tensor<T>` vs `TensorView<T>` | — | |
| expr.rs: `eval_node(...)` | Internal function in tenferro-einsum | Recursive tree evaluation | Core logic |
| expr.rs: `eval_pair(...)` | Internal function in tenferro-einsum | Calls `TensorPrims::execute()` | |
| expr.rs: `eval_pair_into(...)` | Internal function in tenferro-einsum | Same | |
| expr.rs: `execute_nested(...)` | Internal function in tenferro-einsum | Executes omeco result | |
| expr.rs: `compute_contract_output_ids(...)` | Internal function in tenferro-einsum | char → u32 | |
| expr.rs: `compute_child_needed_ids(...)` | Internal function in tenferro-einsum | As-is | |
| single_tensor.rs: `single_tensor_einsum(...)` | Internal function in tenferro-einsum | 5-step pipeline via TensorPrims | See P7 |
| parse.rs | `Subscripts::parse()` + tree extraction | Parenthesized notation | See P8 |
| operand.rs | Removed | — | |
| typed_tensor.rs | Removed | — | |

### Type Adaptation

**EinsumOperand → Tensor\<T\>**: strided-opteinsum dispatches at runtime via
`EinsumOperand` (F64/C64 enum).  tenferro resolves at compile time via
generic `Tensor<T>`.  The `BufferPool` becomes generic too (type-erased
internally via `TypeId`).

**EinsumCode → Subscripts + ContractionTree**: strided-opteinsum's
`EinsumCode { root: EinsumNode, output_ids }` stores tree structure and
labels in one type.  tenferro separates into `Subscripts` (label info) and
`ContractionTree` (tree structure).  The parser must produce both.  See P8.

**omeco dependency**: `ContractionTree::optimize()` uses the omeco crate for
greedy N-ary contraction tree optimization.  Must add `omeco` to
tenferro-einsum's `Cargo.toml`.  See P10.

---

## Cross-Cutting Concerns

### Buffer Pool (Unified Design)

**Placement**: Unified `BufferPool` in `CpuContext` (alongside `PlanCache`).

```
CpuContext
  ├── BufferPool          (type-erased, runtime on/off)
  └── PlanCache           (keyed by PrimDescriptor + shapes)
```

**Design sketch**:

```rust
pub struct BufferPool {
    pools: HashMap<TypeId, Box<dyn Any>>,  // TypeId → Vec<Vec<T>>
    enabled: bool,
    max_per_type: usize,    // e.g., 16
    max_bytes: usize,       // e.g., 64 MB
}

impl BufferPool {
    pub fn acquire<T: 'static>(&mut self, len: usize) -> Vec<T> { ... }
    pub fn recycle<T: 'static>(&mut self, vec: Vec<T>) { ... }
    pub fn set_enabled(&mut self, enabled: bool) { ... }
}
```

**Usage sites**:

| Purpose | strided-rs origin | tenferro target |
|---|---|---|
| GEMM contiguous buffers | `contiguous.rs` thread-local pool | `CpuBackend::execute()` via `CpuContext.pool` |
| N-ary intermediate tensors | `BufferPool` (expr.rs) | `einsum_with_plan()` via `CpuContext.pool` |

### Unsafe Code Inventory

All unsafe code is confined to `CpuBackend` internals.  No unsafe in public API.

| Origin file | Content | Target | Risk |
|---|---|---|---|
| `bgemm_naive.rs` | Pointer offset read/write (innermost loop) | `CpuBackend` internal | Low: well-tested in strided-rs |
| `contiguous.rs` | `Vec::set_len()` (uninit buffer) | `BufferPool::acquire()` | Medium: caller must write all elements |
| `contiguous.rs` | `StridedArray::col_major_from_buffer_uninit()` | `CpuBackend` internal | Low: strided-view API |
| `bgemm_blas.rs` | FFI calls (`cblas_sys::dgemm` etc.) | `CpuBackend` (blas feature) | Low: standard FFI |
| `bgemm_blas.rs` | `extern "C" fn` fallback registration | `CpuBackend` (blas-inject feature) | Low |
| `expr.rs` | Scalar pointer offset read | `tenferro-einsum` internal | Low: single element read |

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
| P3 | Conjugation dispatch change | Medium | Use strided-view's `Conj` op internally when building StridedView from Tensor.  Preserves lazy conjugation inside CpuBackend |
| ~~P4~~ | ~~Thread-local buffer pool~~ | — | Merged into P6 |
| P5 | Algebra-specific dispatch (tropical, etc.) | Medium | `Standard` → faer/blas.  Non-Standard → naive fallback.  Dispatch on `A: Semiring` bounds |
| **P6** | **Unified BufferPool** | **High** | Type-erased unified pool in `CpuContext`.  Runtime on/off toggle.  Serves both contiguous buffers and N-ary intermediates |
| **P7** | **single_tensor_einsum via prims** | **Medium** | Route all steps (diag, trace, permute, broadcast, anti-diag) through `TensorPrims`.  Cache plans in `CpuContext.PlanCache` keyed by `(PrimDescriptor, shapes)` to avoid repeated plan generation |
| **P8** | **Parenthesized tree notation** | **Medium** | Support parenthesized contraction order in parser.  Parser returns `Subscripts` + `Option<ContractionTree>`.  POC skeleton needs update |
| **P9** | **Generative output (size_dict)** | **Medium** | Support `size_dict` for output labels not present in inputs.  Add parameter to tenferro-einsum public API.  POC skeleton needs update |
| P10 | omeco dependency | Low | Add to workspace dependencies |
| **P11** | **Explicit context passing in einsum API** | **High** | All einsum functions gain `ctx: &mut B::Context` and `B: TensorPrims<A>` type parameter.  Follows Rust idiom (explicit `&mut`, no global state).  All 9 einsum functions + AD functions (`tracked_einsum`, `einsum_rrule`, `einsum_frule`, `einsum_hvp`) need signature changes.  See revised signatures below |

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

`ctx` provides access to `BufferPool` and `PlanCache` (P6, P7).
`B` enables compile-time backend selection without global state.

### POC Skeleton Changes Required

The following changes to the existing API skeleton are needed before
implementation:

1. **`PrimDescriptor` enum** — add `MakeContiguous` variant (P1).
2. **`CpuContext` struct** — add `BufferPool` and `PlanCache` fields (P6, P7).
3. **`Subscripts::parse()`** — return type or companion function for
   parenthesized tree extraction (P8).
4. **einsum public functions** — add optional `size_dict` parameter for
   generative output dimensions (P9).
5. **einsum public functions** — add `ctx: &mut B::Context` parameter and
   `B: TensorPrims<A>` type parameter to all 9 einsum functions + all AD
   functions (P11).
