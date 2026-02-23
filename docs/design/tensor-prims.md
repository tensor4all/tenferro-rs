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
    │  T: HasAlgebra (UX sugar) → infers A = T::Algebra automatically
    │  A: Semiring → A::Scalar is the canonical scalar type
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
/// Backend trait parameterized by algebra Alg.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, make_contiguous,
/// anti_trace, anti_diag) must be implemented. Extended ops (contract,
/// elementwise_mul) have default implementations that decompose into core ops.
///
/// The algebra parameter Alg enables extensibility: external crates can
/// implement TensorPrims<MyAlgebra> for CpuBackend (orphan rule compatible).
pub trait TensorPrims<Alg: Algebra> {
    /// Backend-specific plan type.
    type Plan;

    /// Execution context (CPU: thread pool + plan cache; GPU: CUDA stream).
    type Context;

    /// Create an execution plan (cuTENSOR: describe → plan).
    ///
    /// Takes `&mut Self::Context` because plan creation may update the
    /// plan cache (PlanCache) stored in the context.
    fn plan(
        ctx: &mut Self::Context,
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a plan (cuTENSOR: plan → execute).
    ///
    /// Operations receive `Tensor<Alg::Scalar>` directly (PyTorch-aligned).
    /// CPU backends convert to strided views internally; GPU backends
    /// extract device pointers.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Alg::Scalar,
        inputs: &[&Tensor<Alg::Scalar>],
        beta: Alg::Scalar,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for this algebra.
    fn has_extension_for(ext: Extension) -> bool;
}
```

Key design decisions:

1. **Associated functions, not methods** — No `&self` receiver. Call as
   `CpuBackend::plan(&mut ctx, ...)`. Execution resources (thread pool,
   CUDA stream, plan cache) are passed via `type Context`.
2. **`&mut Context`** — Both `plan()` and `execute()` take mutable context.
   `plan()` updates the PlanCache. `execute()` may resize GPU workspace.
3. **`Tensor<Alg::Scalar>` in execute** — Operations take `Tensor` references
   directly (PyTorch-aligned). CPU backends convert to `StridedView` internally;
   GPU backends extract device pointers. The `Plan` type is not parameterized
   by scalar type (no GAT); the algebra's scalar type is fixed by `Alg`.
4. **Modes are `u32`** — Matching cuTENSOR's unsigned mode labels.
5. **Single trait with dynamic extension query** — `has_extension_for(ext)`
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
/// Implemented generically for all scalar types S that implement Scalar.
impl<S: Scalar> TensorPrims<Standard<S>> for CpuBackend {
    type Plan = CpuPlan;
    type Context = CpuContext;

    fn plan(ctx: &mut CpuContext, desc: &PrimDescriptor, shapes: &[&[usize]])
        -> Result<CpuPlan> { ... }

    fn execute(ctx: &mut CpuContext, plan: &CpuPlan,
        alpha: S, inputs: &[&Tensor<S>], beta: S, output: &mut Tensor<S>,
    ) -> Result<()> { ... }

    fn has_extension_for(ext: Extension) -> bool {
        // CPU supports Contract and ElementwiseMul for all standard types
        true
    }
}

/// CPU plan — concrete enum, no type erasure.
enum CpuPlan {
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

impl<S: Scalar> TensorPrims<Standard<S>> for CudaBackend {
    type Plan = CudaPlan;
    type Context = CudaContext;
    // ...
}

// RocmBackend follows the same pattern
pub struct RocmBackend { ... }
pub struct RocmContext { ... }
impl<S: Scalar> TensorPrims<Standard<S>> for RocmBackend { ... }
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

**Cache key policy** (decided):

- **CPU** (`CpuBackend`): Key = `(PrimDescriptor variant, input shapes,
  output shape)`. Strides are **not** included because `CpuPlan` records
  the shape-derived loop structure; strides are resolved at `execute()`
  time from the live `StridedView`. Concretely, the key must include:
  - The `PrimDescriptor` discriminant (enum variant tag)
  - All mode labels (`modes_a`, `modes_b`, `modes_c`) from the descriptor
  - The concrete dimension value for each mode (i.e., the shapes of all
    input and output tensors)
  - For `BatchedGemm`: `(batch_dims, m, n, k)`
  - For `Reduce`/`Trace`: the `ReduceOp` kind
  - For `ElementwiseUnary`: the `UnaryOp` kind

- **GPU** (`CudaBackend`, `RocmBackend`): Key = `(PrimDescriptor variant,
  input shapes, output shape, input strides, output strides, scalar data
  type)`. cuTENSOR/hipTENSOR plans encode memory layout into the plan
  handle — a plan created for row-major input cannot be reused for
  column-major input. Concretely, the key must include everything in the
  CPU key **plus**:
  - The stride vector for each input tensor
  - The stride vector for the output tensor
  - The scalar element type (e.g., `f32` vs `f64`), encoded as a
    `cutensorDataType_t` / `hiptensorDataType_t` discriminant

  See [gpu-backend-design.md](./gpu-backend-design.md) for the full GPU
  plan lifecycle and the G4 problem entry.

---

## Backend Implementation Matrix

| Backend | Algebra | Extended ops | Notes |
|---------|---------|-------------|-------|
| CpuBackend | Standard\<S\> (any S: Scalar) | Contract, ElementwiseMul | faer/cblas GEMM |
| CpuBackend | MaxPlus\<f64\> | None (decompose to core) | tropical-gemm SIMD |
| CpuBackend | MyAlgebra | User choice | User-provided kernels |
| CudaBackend [future] | Standard\<S\> (any S: Scalar) | Contract, ElementwiseMul | cuTENSOR |
| RocmBackend [future] | Standard\<S\> (any S: Scalar) | Contract, ElementwiseMul | hipTENSOR |

---

## Tropical and User-Defined Algebras

**Tropical backend** (in separate `tenferro-tropical` crate):

```rust
pub struct MaxPlus<T>(PhantomData<T>);

// HasAlgebra is UX sugar: wires MaxPlus<f64> scalar to MaxPlus<f64> algebra
impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus<f64>; }

impl TensorPrims<MaxPlus<f64>> for CpuBackend {
    type Plan = TropicalPlan;
    type Context = CpuContext;

    fn has_extension_for(ext: Extension) -> bool {
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
// HasAlgebra is UX sugar: wires MyScalar to MyAlgebra for automatic inference
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan = MyPlan;
    type Context = CpuContext;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred via HasAlgebra
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
