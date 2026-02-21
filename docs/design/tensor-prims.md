# Tensor Prims: TensorPrims\<A\> Protocol

The central protocol layer. Defines `TensorPrims<A>` parameterized by algebra
`A`, with a cuTENSOR-compatible describe → plan → execute pattern.

See [contraction-pipeline.md](./contraction-pipeline.md) for the binary
contraction pipeline details and copy elision strategy.

---

## Design Overview

The protocol uses a **universal set** of primitive operations plus an
**extended set** of optimized composites. The trait is parameterized by
algebra `A` so different scalar types can plug in their own implementations.

```
tenferro-einsum (engine)
    │
    │  T: HasAlgebra → infers A automatically
    │
    ├── [has_extension_for::<T>(Contract)?]
    │   YES → execute Contract plan (fused permute+GEMM)
    │
    └── [otherwise]
        decompose into core ops:
        diag → trace/reduce → permute_view → make_contiguous → batched_gemm
```

**Dispatch is dynamic**: `has_extension_for::<T>(ext)` queries at runtime
whether a specific extended operation is available for scalar type `T`.
This is important because:
- GPU backends are loaded at runtime (dlopen)
- cuTENSOR supports `f32`/`f64`/Complex but not tropical types
- CPU backends may support `contract` for `f64` (faer) but not for custom types

Note: `diag` (diagonal extraction) and `repeat` (broadcast) are **zero-copy
stride tricks** handled at the `Tensor<T>` level, not in `TensorPrims`.

---

## Operation Categories

| Tier | Operation | cuTENSOR | hipTensor | CPU (strided-rs) |
|------|-----------|----------|-----------|-------------------|
| **Core** | `batched_gemm` | `cutensorContract` (subset) | `hiptensorContract` (subset) | `BgemmBackend::bgemm_contiguous_into` |
| **Core** | `reduce` | `cutensorReduce` | `hiptensorReduce` | `reduce_axis` (strided-kernel) |
| **Core** | `trace` | `cutensorReduce` on diagonal | `hiptensorReduce` on diagonal | `reduce_trace_axes` (strided-einsum2) |
| **Core** | `permute` | `cutensorPermute` | `hiptensorPermute` | `StridedView::permute` + copy |
| **Core** | `make_contiguous` | n/a (strides accepted natively) | n/a | full-tensor contiguity check + copy |
| **Core** | `anti_trace` | custom kernel | custom kernel | scatter-add loop |
| **Core** | `anti_diag` | custom kernel | custom kernel | write-to-diagonal loop |
| **Extended** | `contract` | `cutensorContract` (full) | `hiptensorContract` (full) | strided-einsum2 pipeline |
| **Extended** | `elementwise_mul` | `cutensorElementwiseBinary` | `hiptensorElementwiseBinary` | `zip_map2_into` |

Extended operations are in the same `PrimDescriptor` enum as core ops.
Whether a backend supports them is queried at runtime via
`has_extension_for::<T>(Extension::Contract)`.

---

## Adjoint Pairs for AD

The core operations form adjoint pairs, enabling clean VJP/JVP rules:

| Forward | Backward (adjoint) | Description |
|---------|-------------------|-------------|
| `trace(A)` | `anti_trace(∂y)` | Scatter-add gradient to diagonal |
| `diag(A)` | `anti_diag(∂y)` | Write gradient to diagonal positions |
| `reduce(A, dim)` | `repeat(∂y, dim)` | Broadcast gradient |
| `permute(A, p)` | `permute(∂y, p⁻¹)` | Inverse permutation |
| `batched_gemm(A, B)` | Leibniz rule | `∂A = gemm(∂C, B^T)`, `∂B = gemm(A^T, ∂C)` |

---

## Key Types

```rust
/// Describes any TensorPrims operation (cuTENSOR pattern: describe → plan → execute).
pub enum PrimDescriptor {
    // Core
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    Reduce { modes_a: Vec<u32>, modes_c: Vec<u32>, op: ReduceOp },
    Trace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    Permute { modes_a: Vec<u32>, modes_b: Vec<u32> },
    MakeContiguous,
    AntiTrace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    AntiDiag { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    // Extended (dynamically queried)
    Contract { modes_a: Vec<u32>, modes_b: Vec<u32>, modes_c: Vec<u32> },
    ElementwiseMul,
    // Unary element-wise (for linalg AD)
    ElementwiseUnary { op: UnaryOp },
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

/// Unary element-wise operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp { Negate, Reciprocal, Abs, Sqrt }

/// Extended operation identifiers for dynamic capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension { Contract, ElementwiseMul }
```

---

## TensorPrims\<A\> Trait

```rust
/// Backend trait parameterized by algebra A.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, make_contiguous,
/// anti_trace, anti_diag) must be implemented. Extended ops (contract,
/// elementwise_mul) have default implementations that decompose into core ops.
///
/// The algebra parameter A enables extensibility: external crates can
/// implement TensorPrims<MyAlgebra> for CpuBackend (orphan rule compatible).
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Execution context (CPU: thread pool; GPU: CUDA stream).
    type Context;

    /// Create an execution plan (cuTENSOR: describe → plan).
    fn plan<T: ScalarBase>(
        ctx: &Self::Context,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan (cuTENSOR: plan → execute).
    fn execute<T: ScalarBase>(
        ctx: &Self::Context,
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for scalar type T.
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}
```

Key design decisions:

1. **Associated functions, not methods** — No `&self` receiver. Call as
   `CpuBackend::plan::<f64>(&ctx, ...)`. Execution resources (thread pool,
   CUDA stream) are passed via `type Context`.
2. **StridedView/StridedViewMut directly** — Not `Storage<T>` + `TensorMeta`.
3. **Modes are `u32`** — Matching cuTENSOR's unsigned mode labels.
4. **Single trait with dynamic extension query** — `has_extension_for::<T>(ext)`
   for runtime capability detection. Supports GPU backends loaded via dlopen.
5. **Plan-based execution for all ops** — cuTENSOR pattern: `PrimDescriptor`
   → `plan` → `execute`. Plans cache expensive analysis for reuse.
6. **Algebra parameterization** — Enables orphan-rule-compatible extension.
7. **diag/repeat on Tensor, not TensorPrims** — Zero-copy stride tricks that
   don't need backend dispatch.

---

## CpuBackend

```rust
pub struct CpuBackend;

/// CPU execution context — wraps a rayon thread pool.
pub struct CpuContext {
    pool: rayon::ThreadPool,
}

/// Standard arithmetic on CPU (faer GEMM for f64/f32, naive for others).
impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(ctx: &CpuContext, desc: &PrimDescriptor, shapes: &[&[usize]])
        -> Result<CpuPlan<T>> { ... }

    fn execute<T: ScalarBase>(ctx: &CpuContext, plan: &CpuPlan<T>, ...) -> Result<()> { ... }

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        // CPU supports Contract and ElementwiseMul for all standard types
        true
    }
}

/// CPU plan — concrete enum, no type erasure.
enum CpuPlan<T: ScalarBase> {
    BatchedGemm { m: usize, n: usize, k: usize, ... },
    Reduce { axis: usize, op: ReduceOp },
    Trace { paired: Vec<(u32, u32)> },
    Permute { perm: Vec<usize> },
    MakeContiguous { /* contiguity analysis result */ },
    Contract { /* strided-einsum2 cached analysis */ },
    ElementwiseMul,
    ...
}
```

**Core ops implementation**:

| Core Op | CPU Implementation |
|---------|-------------------|
| `batched_gemm` | faer/cblas GEMM via `BgemmBackend` trait |
| `reduce` | `reduce_axis` from strided-kernel |
| `trace` | `reduce_trace_axes` from strided-einsum2 |
| `permute` | `StridedView::permute` + `copy_into` from strided-kernel |
| `make_contiguous` | Full-tensor contiguity check + col-major copy if needed |
| `anti_trace` | Scatter-add loop (for AD backward) |
| `anti_diag` | Write-to-diagonal loop (for AD backward) |

**Extended ops implementation**:

| Extended Op | CPU Implementation |
|------------|-------------------|
| `contract` | strided-einsum2 pipeline (fusability + GEMM) |
| `elementwise_mul` | `zip_map2_into` from strided-kernel |

### GEMM Backend Selection (Compile-Time, Future)

```toml
# tenferro-prims/Cargo.toml (planned)
[features]
default = ["faer"]
faer = ["dep:faer"]
cblas = ["dep:cblas-sys"]
```

| Feature | GEMM source | Use case |
|---------|------------|----------|
| `faer` (default) | Pure Rust, zero external deps | Standalone apps, guaranteed build |
| `cblas` | Requires `cblas-src` or `cblas-inject` | HPC, Julia integration |

When `cblas` is selected, the actual CBLAS implementation is provided by the
downstream user (`cblas-src` for OpenBLAS/MKL, `cblas-inject` for Julia's
`libblastrampoline`).

---

## Backend Implementation Matrix

| Backend | Algebra | Extended ops | Notes |
|---------|---------|-------------|-------|
| CpuBackend | Standard | Contract, ElementwiseMul | faer/cblas GEMM |
| CpuBackend | MaxPlus | None (decompose to core) | tropical-gemm SIMD |
| CpuBackend | MyAlgebra | User choice | User-provided kernels |
| GpuBackend [future] | Standard | Contract, ElementwiseMul | cuTENSOR/hipTensor |
| GpuBackend [future] | MaxPlus | None | No cuTENSOR tropical support |

---

## Tropical and User-Defined Algebras

**Tropical backend** (in separate `tenferro-tropical` crate):

```rust
pub struct MaxPlus;

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus; }

impl TensorPrims<MaxPlus> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;
    type Context = CpuContext;

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        false  // tropical uses core ops decomposition, no fused contract
    }
    ...
}
```

**User-defined algebra** (in user crate):

```rust
struct MyScalar(f64);
struct MyAlgebra;

impl ScalarBase for MyScalar { ... }
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = MyPlan<T>;
    type Context = CpuContext;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred
```

---

## Custom Closures: Use strided-kernel Directly

The `TensorPrims` trait does not provide a closure-based API, because GPU
backends cannot execute arbitrary Rust closures. For custom element-wise
operations, users access strided-kernel directly via `Tensor::view()`:

```rust
// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out.view_mut(), &a_view, &b_view, |a, b| a * b + 1.0);
```

This keeps `tenferro-prims` purely cuTENSOR/hipTensor-compatible.

---

## Usage Examples

```rust
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims, PrimDescriptor, ReduceOp, Standard};

let ctx = CpuContext::new(4);  // 4-thread pool

// Plan + execute: GEMM
let desc = PrimDescriptor::BatchedGemm { batch_dims: vec![], m: 3, n: 5, k: 4 };
let plan = CpuBackend::plan::<f64>(&ctx, &desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
CpuBackend::execute(&ctx, &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();

// Plan + execute: Reduction
let desc = PrimDescriptor::Reduce { modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum };
let plan = CpuBackend::plan::<f64>(&ctx, &desc, &[&[3, 4], &[3]]).unwrap();
CpuBackend::execute(&ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

// Dynamic extension check
if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    let desc = PrimDescriptor::Contract { modes_a: vec![0,1], modes_b: vec![1,2], modes_c: vec![0,2] };
    let plan = CpuBackend::plan::<f64>(&ctx, &desc, &shapes).unwrap();
    CpuBackend::execute(&ctx, &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
}
```

---

## GPU Backend (Future)

GPU support via cuTENSOR/hipTensor is planned but not in the POC.

### GPU Vtable

cuTENSOR and hipTensor have nearly identical C APIs. A single function
pointer table (`TensorLibVtable`) abstracts over both:

```rust
struct TensorLibVtable {
    create_handle: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32>,
    destroy_handle: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,
    create_contraction: Symbol<unsafe extern "C" fn(/* ... */) -> i32>,
    create_plan: Symbol<unsafe extern "C" fn(/* ... */) -> i32>,
    contract: Symbol<unsafe extern "C" fn(/* ... */) -> i32>,
    // Permutation, reduction, elementwise (same pattern)
    ...
}

impl TensorLibVtable {
    fn load_cutensor(lib: &Library) -> Result<Self> { ... }
    fn load_hiptensor(lib: &Library) -> Result<Self> { ... }
}

pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,
}
```

### GPU Plan Caching

```rust
#[derive(Hash, Eq, PartialEq, Clone)]
pub struct PlanCacheKey {
    pub shapes: Vec<Vec<usize>>,
    pub strides: Vec<Vec<usize>>,
    pub modes: Vec<Vec<u32>>,
    pub dtype: u32,
}

pub struct PlanCache {
    cache: HashMap<PlanCacheKey, GpuPlan>,
    capacity: usize,
}
```

### GPU Backend Discovery

The caller (Julia, Python, or standalone Rust) provides the path to
the shared library. No auto-search.

```rust
pub struct BackendRegistry {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self { ... }  // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()> { ... }
}
```
