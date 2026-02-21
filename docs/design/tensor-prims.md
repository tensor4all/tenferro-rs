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
| **Core** | `anti_trace` | Contract(eye, ∂C) | Contract(eye, ∂C) | scatter-add loop |
| **Core** | `anti_diag` | Contract(eye, ∂C) | Contract(eye, ∂C) | write-to-diagonal loop |
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
    // Unary element-wise (for linalg AD and resolve_conj)
    ElementwiseUnary { op: UnaryOp },
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

/// Unary element-wise operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp { Negate, Reciprocal, Abs, Sqrt, Conj }

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

    /// Execution context (CPU: thread pool + plan cache; GPU: CUDA stream).
    type Context;

    /// Create an execution plan (cuTENSOR: describe → plan).
    ///
    /// Takes `&mut Self::Context` because plan creation may update the
    /// plan cache (PlanCache) stored in the context.
    fn plan<T: ScalarBase>(
        ctx: &mut Self::Context,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan (cuTENSOR: plan → execute).
    ///
    /// Takes `&mut Self::Context` for consistency and to allow future
    /// context state updates (e.g., workspace resizing on GPU).
    fn execute<T: ScalarBase>(
        ctx: &mut Self::Context,
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
   `CpuBackend::plan::<f64>(&mut ctx, ...)`. Execution resources (thread pool,
   CUDA stream, plan cache) are passed via `type Context`.
2. **`&mut Context`** — Both `plan()` and `execute()` take mutable context.
   `plan()` updates the PlanCache. `execute()` may resize GPU workspace.
3. **StridedView/StridedViewMut directly** — Not `Storage<T>` + `TensorMeta`.
4. **Modes are `u32`** — Matching cuTENSOR's unsigned mode labels.
5. **Single trait with dynamic extension query** — `has_extension_for::<T>(ext)`
   for runtime capability detection. Supports GPU backends loaded via dlopen.
6. **Plan-based execution for all ops** — cuTENSOR pattern: `PrimDescriptor`
   → `plan` → `execute`. Plans cache expensive analysis for reuse.
7. **Algebra parameterization** — Enables orphan-rule-compatible extension.
8. **diag/repeat on Tensor, not TensorPrims** — Zero-copy stride tricks that
   don't need backend dispatch.

---

## CpuBackend

```rust
pub struct CpuBackend;

/// CPU execution context — wraps a rayon thread pool and plan cache.
///
/// Intermediate buffer allocation relies on the global allocator
/// (e.g., mimalloc/jemalloc) rather than a custom buffer pool.
pub struct CpuContext {
    pool: rayon::ThreadPool,
    plan_cache: PlanCache,
}

impl CpuContext {
    pub fn new(num_threads: usize) -> Self;
    pub fn num_threads(&self) -> usize;
    pub fn thread_pool(&self) -> &rayon::ThreadPool;
    pub fn plan_cache_mut(&mut self) -> &mut PlanCache;
}

/// Standard arithmetic on CPU (faer GEMM for f64/f32, naive for others).
impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;
    type Context = CpuContext;

    fn plan<T: ScalarBase>(ctx: &mut CpuContext, desc: &PrimDescriptor, shapes: &[&[usize]])
        -> Result<CpuPlan<T>> { ... }

    fn execute<T: ScalarBase>(ctx: &mut CpuContext, plan: &CpuPlan<T>, ...) -> Result<()> { ... }

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
    ElementwiseUnary { op: UnaryOp },
    ...
}
```

### resolve_conj

Each backend provides `resolve_conj()` to materialize lazy conjugation:

```rust
impl CpuBackend {
    pub fn resolve_conj<T: Scalar>(ctx: &mut CpuContext, src: &Tensor<T>) -> Tensor<T>;
}
```

If `src.is_conjugated()` is false, returns a shallow clone. Otherwise,
applies `ElementwiseUnary(Conj)` and returns a new tensor with
`conjugated = false`. Equivalent to PyTorch's `torch.resolve_conj()`.

### Core ops implementation

| Core Op | CPU Implementation |
|---------|-------------------|
| `batched_gemm` | faer/cblas GEMM via `BgemmBackend` trait |
| `reduce` | `reduce_axis` from strided-kernel |
| `trace` | `reduce_trace_axes` from strided-einsum2 |
| `permute` | `StridedView::permute` + `copy_into` from strided-kernel |
| `make_contiguous` | Full-tensor contiguity check + col-major copy if needed |
| `anti_trace` | Scatter-add loop (for AD backward) |
| `anti_diag` | Write-to-diagonal loop (for AD backward) |

### Extended ops implementation

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

## GPU Backends

Two separate backends — `CudaBackend` and `RocmBackend` — as distinct
types in `tenferro-prims`. Subtle API differences and future custom
kernel needs justify separate implementations over a unified vtable.

See [gpu-backend-design.md](./gpu-backend-design.md) for full details.

```rust
pub struct CudaBackend {
    _handle: *mut c_void,
    _lib: libloading::Library,
}

pub struct CudaContext {
    _stream: *mut c_void,
    _workspace: Vec<u8>,
    _plan_cache: PlanCache,
}

impl TensorPrims<Standard> for CudaBackend {
    type Plan<T: ScalarBase> = CudaPlan<T>;
    type Context = CudaContext;
    // ...
}

// RocmBackend follows the same pattern
pub struct RocmBackend { ... }
pub struct RocmContext { ... }
impl TensorPrims<Standard> for RocmBackend { ... }
```

### Backend Registry

```rust
pub struct BackendRegistry {
    cpu: CpuBackend,
    cuda: Option<CudaBackend>,
    rocm: Option<RocmBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self;                            // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()>;
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()>;
    pub fn cpu(&self) -> &CpuBackend;
    pub fn cuda(&self) -> Option<&CudaBackend>;
    pub fn rocm(&self) -> Option<&RocmBackend>;
}
```

---

## PlanCache

```rust
pub struct PlanCache {
    _entries: HashMap<u64, Box<dyn Any>>,
}
```

Plan cache is stored in the Context (CpuContext, CudaContext, RocmContext).
This is why `plan()` and `execute()` take `&mut Context`.

**Cache key design** (to be finalized):
- CPU: keyed by `(PrimDescriptor, shapes)` — strides are not part of the
  key because CPU plans depend on shape, not layout
- GPU: keyed by `(PrimDescriptor, shapes, strides)` — cuTENSOR plans are
  stride-dependent (see gpu-backend-design.md G4)

---

## Backend Implementation Matrix

| Backend | Algebra | Extended ops | Notes |
|---------|---------|-------------|-------|
| CpuBackend | Standard | Contract, ElementwiseMul | faer/cblas GEMM |
| CpuBackend | MaxPlus | None (decompose to core) | tropical-gemm SIMD |
| CpuBackend | MyAlgebra | User choice | User-provided kernels |
| CudaBackend [future] | Standard | Contract, ElementwiseMul | cuTENSOR |
| RocmBackend [future] | Standard | Contract, ElementwiseMul | hipTENSOR |

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
operations, users access strided-kernel directly via `buffer().as_slice()`:

```rust
// Custom closures: use strided-kernel directly (CPU only)
// Build StridedView from buffer + dims + strides + offset
let a_slice = tensor_a.buffer().as_slice().unwrap();
// ... construct StridedView and use strided_kernel
```

This keeps `tenferro-prims` purely cuTENSOR/hipTensor-compatible.
