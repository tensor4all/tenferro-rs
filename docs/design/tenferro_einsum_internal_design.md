# tenferro-einsum Internal Design

> This document details the internal architecture of the tenferro einsum
> subsystem, covering the tensor operation protocol layer (`tenferro-prims`),
> the einsum engine (`tenferro-einsum`), and backend implementations for CPU
> and GPU.
>
> This is a companion to the [tenferro design document](./tenferro_design.md),
> the [ecosystem overview](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro_unified_tensor_backend.md)
> in tensor4all-meta, and the [algorithm comparison](./einsum_algorithm_comparison.md).
>
> **POC repository**: <https://github.com/tensor4all/tenferro-rs/>

## Layered Architecture

```
Layer 4: Einsum Engine (tenferro-einsum)
         High-level API on Tensor<T>
         Absorbs strided-opteinsum + omeinsum-rs
         N-ary contraction tree, algebra dispatch, backward
         Decomposes einsum into universal set primitives
         ↓
Layer 3: Tensor Type (tenferro-tensor)
         Tensor<T> = DataBuffer + shape + strides
         Zero-copy view ops: permute, broadcast, diagonal, reshape
         view()/view_mut() bridge to strided-rs
         ↓
Layer 2: Tensor Operation Protocol (tenferro-prims)
         "Tensor BLAS": TensorPrims<A> parameterized by algebra A
         cuTENSOR pattern: PrimDescriptor → plan → execute
         Core ops: batched_gemm, reduce, trace, permute,
           anti_trace, anti_diag
         Extended ops (dynamic query): contract, elementwise_mul
         HasAlgebra trait: T → A automatic inference
         Absorbs strided-einsum2 (binary contraction pipeline)
         CPU impl depends on strided-kernel (stays in strided-rs)
         GPU dispatch via tenferro-device
         ↓
Shared:  Device Layer (tenferro-device)
         LogicalMemorySpace + ComputeDevice, preferred device selection, Error, Result
         Used by: tenferro-prims, tenferro-tensor, tenferro-einsum

Foundation: strided-rs (independent workspace, not absorbed)
         strided-traits -> strided-view -> strided-kernel
         General-purpose strided array library, no BLAS dependency
```

### Design Rationale

GiggleLiu proposed that strided-rs should serve as a "tensor-level BLAS" --
the CPU counterpart to cuTENSOR -- with standardized interfaces applicable to
CPU, GPU, and tropical tensors. Specifically, he proposed a **universal set**
of primitive operations (`batched_gemm`, `trace`, `diag`, `permute`, `repeat`,
`anti_diag`, `anti_trace`) that any backend must implement, plus an **extended
set** of optimized composites (`contract`, `elementwise_mul`) that backends
may optionally provide for better performance.

We introduce `tenferro-prims` as a low-level "Tensor BLAS" protocol layer
with a **unified `TensorPrims<A>` trait** parameterized by algebra `A`:

1. **Core operations** (universal set): `batched_gemm`, `reduce`, `trace`,
   `permute`, `anti_trace`, `anti_diag` — every backend must implement these.
   `tenferro-einsum` decomposes any einsum into these primitives.
2. **Extended operations** (optional): `contract` (fused permute + GEMM) and
   `elementwise_mul` — dynamically queried via `has_extension_for::<T>(ext)`.
   Backends that support these get dispatched to directly for better performance.
3. **Plan-based execution** (cuTENSOR pattern): All operations follow
   `PrimDescriptor → plan → execute`. Plans cache expensive analysis (GPU kernel
   selection, CPU fusability checks) for reuse.
4. **Algebra parameterization**: `TensorPrims<A>` enables external crates to
   implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).
   The `HasAlgebra` trait on scalar types enables automatic inference:
   `Tensor<f64>` → `Standard`, `Tensor<MaxPlus<f64>>` → `MaxPlus`.

This is motivated by the observation that **cuTENSOR and hipTensor have
nearly identical APIs** (AMD intentionally mirrors NVIDIA's API). cuTENSOR's
`cutensorContract` corresponds to the extended `contract`; the core primitives
map to individual cuTENSOR operations or stride tricks.

The core operations form **adjoint pairs** for clean AD support:
`trace ↔ anti_trace`, `diag ↔ anti_diag`, `reduce ↔ repeat`,
`permute ↔ inverse permute`, `batched_gemm` uses the Leibniz rule.

`tenferro-tensor` defines the `Tensor<T>` type above it, and `tenferro-einsum`
provides a high-level einsum API on `Tensor<T>`, internally delegating
operations to `tenferro-prims`. Custom closure-based operations
(which cannot run on GPU) are not part of `TensorPrims`; users access
strided-kernel directly via `Tensor::view()`.

**POC status**: The [tenferro-rs POC](https://github.com/tensor4all/tenferro-rs/)
implements the five-crate structure (`tenferro-device`, `tenferro-algebra`,
`tenferro-prims`, `tenferro-tensor`, `tenferro-einsum`) with stub implementations. The `TensorPrims`
trait and public einsum API are defined. `CpuBackend` is the only backend; GPU
backends, `BackendRegistry`, and `TensorLibVtable` are future work.

---

## Layer 2: tenferro-prims

### Unified TensorPrims<A> Architecture

`tenferro-prims` uses a **single trait** `TensorPrims<A>` parameterized by
algebra `A`, with a cuTENSOR-compatible plan-based execution model:

```
tenferro-einsum
    │
    │  T: HasAlgebra → infers algebra A automatically
    │
    ├─ [has_extension_for::<T>(Contract)?]
    │   YES → plan + execute Contract (fused permute+GEMM)
    │
    └─ [otherwise]
        plan + execute core ops:
        diag → trace/reduce → permute → batched_gemm → permute
```

### Operation Categories

| Tier | Operation | cuTENSOR | hipTensor | CPU (strided-rs) |
|------|-----------|----------|-----------|-------------------|
| **Core** | `batched_gemm` | `cutensorContract` (subset) | `hiptensorContract` (subset) | `BgemmBackend::bgemm_contiguous_into` |
| **Core** | `reduce` | `cutensorReduce` | `hiptensorReduce` | `reduce_axis` (strided-kernel) |
| **Core** | `trace` | `cutensorReduce` on diagonal | `hiptensorReduce` on diagonal | `reduce_trace_axes` (strided-einsum2) |
| **Core** | `permute` | `cutensorPermute` | `hiptensorPermute` | `StridedView::permute` + copy |
| **Core** | `anti_trace` | custom kernel | custom kernel | scatter-add loop |
| **Core** | `anti_diag` | custom kernel | custom kernel | write-to-diagonal loop |
| **Extended (dynamic)** | `contract` | `cutensorContract` (full) | `hiptensorContract` (full) | strided-einsum2 pipeline |
| **Extended (dynamic)** | `elementwise_mul` | `cutensorElementwiseBinary` | `hiptensorElementwiseBinary` | `zip_map2_into` |

Extended operations are in the same `PrimDescriptor` enum as core ops.
Whether a backend supports them is queried at runtime via
`has_extension_for::<T>(Extension::Contract)`.

Note: `diag` (diagonal extraction) and `repeat` (broadcast) are **zero-copy
stride tricks** on `Tensor<T>`, not in `TensorPrims` — they don't need backend
dispatch.

### Adjoint Pairs for AD

The core operations form adjoint pairs, enabling clean VJP/JVP rules
for automatic differentiation:

| Forward | Backward (adjoint) | Description |
|---------|-------------------|-------------|
| `trace(A)` | `anti_trace(∂y)` | Scatter-add gradient to diagonal |
| `diag(A)` | `anti_diag(∂y)` | Write gradient to diagonal positions |
| `reduce(A, dim)` | `repeat(∂y, dim)` | Broadcast gradient |
| `permute(A, p)` | `permute(∂y, p⁻¹)` | Inverse permutation |
| `batched_gemm(A, B)` | Leibniz rule | `∂A = gemm(∂C, B^T)`, `∂B = gemm(A^T, ∂C)` |

### ReduceOp Enum

```rust
/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}
```

### Plan-Based Execution (All Operations)

All operations follow the cuTENSOR pattern of describe → plan → execute:

```rust
/// Describes any TensorPrims operation.
pub enum PrimDescriptor {
    // Core
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    Reduce { modes_a: Vec<u32>, modes_c: Vec<u32>, op: ReduceOp },
    Trace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    Permute { modes_a: Vec<u32>, modes_b: Vec<u32> },
    AntiTrace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    AntiDiag { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    // Extended (dynamically queried)
    Contract { modes_a: Vec<u32>, modes_b: Vec<u32>, modes_c: Vec<u32> },
    ElementwiseMul,
}

/// Extended operation identifiers for dynamic capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension { Contract, ElementwiseMul }
```

### TensorPrims<A> Trait

```rust
/// Backend trait parameterized by algebra A.
///
/// Provides a cuTENSOR-compatible plan-based execution model.
/// Core ops (batched_gemm, reduce, trace, permute, anti_trace, anti_diag)
/// must be implemented. Extended ops (contract, elementwise_mul) are
/// dynamically queried via has_extension_for.
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Create an execution plan (cuTENSOR: describe → plan).
    fn plan<T: ScalarBase>(
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan (cuTENSOR: plan → execute).
    fn execute<T: ScalarBase>(
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

### Extended Operations

Extended operations (`Contract`, `ElementwiseMul`) are part of the same
`PrimDescriptor` enum and use the same `plan` → `execute` flow. Whether a
backend supports them is queried dynamically:

```rust
// Check if backend supports fused contraction for this scalar type
if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    let desc = PrimDescriptor::Contract { modes_a, modes_b, modes_c };
    let plan = CpuBackend::plan::<f64>(&desc, &shapes)?;
    CpuBackend::execute(&plan, alpha, &inputs, beta, &mut output)?;
} else {
    // Decompose into core ops: diag → trace → permute → batched_gemm
}
```

Key design decisions:

1. **Associated functions, not methods** — No `&self` receiver. Call as
   `CpuBackend::plan::<f64>(...)` instead of `backend.plan(...)`.
2. **StridedView/StridedViewMut directly** — Not `Storage<T>` + `TensorMeta`.
3. **Modes are `u32`** — Matching cuTENSOR's unsigned mode labels.
4. **Single trait with dynamic extension query** — `TensorPrims<A>` with
   `has_extension_for::<T>(ext)` for runtime capability detection. This
   supports GPU backends loaded at runtime (dlopen) where capabilities
   are not known at compile time.
5. **Plan-based execution for all ops** — cuTENSOR pattern: `PrimDescriptor`
   → `plan` → `execute`. Plans cache expensive analysis for reuse.
6. **Algebra parameterization** — `TensorPrims<A>` enables orphan-rule-compatible
   extension: external crates implement `TensorPrims<MyAlgebra> for CpuBackend`.
7. **diag/repeat on Tensor, not TensorPrims** — These are zero-copy stride
   tricks that don't need backend dispatch (no computation involved).

### Custom Closures: Use strided-kernel Directly

The `TensorPrims` trait does not provide a closure-based API, because GPU
backends cannot execute arbitrary Rust closures (cuTENSOR only supports
predefined operator enums, and GPU shaders require special compilation).

For custom element-wise operations, users access strided-kernel directly
via `Tensor::view()`:

```rust
// TensorPrims<A>: works on CPU and GPU (plan-based)
let plan = CpuBackend::plan::<f64>(&desc, &shapes)?;
CpuBackend::execute(&plan, alpha, &inputs, beta, &mut output)?;

// Extended contraction (if backend supports it)
if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    let plan = CpuBackend::plan::<f64>(&contract_desc, &shapes)?;
    CpuBackend::execute(&plan, alpha, &[&a, &b], beta, &mut c)?;
}

// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out_view, &a_view, &b_view, |a, b| a * b + 1.0);
```

This keeps `tenferro-prims` purely cuTENSOR/hipTensor-compatible.
strided-kernel (in the independent strided-rs workspace) provides
cache-optimized iteration for arbitrary closures: dimension fusion,
L1 blocking, importance-weighted ordering, SIMD auto-vectorization.

---

## Layer 1: Backend Implementations

### Device Layer (tenferro-device)

The POC `tenferro-device` crate provides a memory/compute separation abstraction:

```rust
/// Logical memory space where tensor buffers reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    /// Always-available host memory.
    MainMemory,
    /// GPU-accessible memory space.
    GpuMemory { space_id: usize },
}

/// Compute device where kernels execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    /// CPU device with execution context ID.
    Cpu { device_id: usize },
    /// NVIDIA CUDA device with a specific device ID.
    Cuda { device_id: usize },
    /// AMD HIP device with a specific device ID.
    Hip { device_id: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind { Contract, BatchedGemm, Reduce, Trace, Permute, ElementwiseMul }

/// Returns executable compute devices in descending preference order.
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>;

/// Error type used across the tenferro workspace.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    #[error("device error: {0}")]
    DeviceError(String),

    #[error("no compatible compute device for {op:?} in memory space {space:?}")]
    NoCompatibleComputeDevice { space: LogicalMemorySpace, op: OpKind },

    #[error("operation across distinct logical memory spaces is not allowed by default")]
    CrossMemorySpaceOperation,

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Result type alias using Error.
pub type Result<T> = std::result::Result<T, Error>;
```

The POC does **not** include `BackendRegistry`, `GpuBackend`, or
`TensorLibVtable`. These are future work for GPU support. The current
`tenferro-device` crate is intentionally minimal: `LogicalMemorySpace`,
`ComputeDevice`, `preferred_compute_devices`, `Error`, and `Result`.

### CPU Backend

The CPU backend (`CpuBackend`) implements `TensorPrims<Standard>` using
strided-rs kernels. It supports both core and extended operations
(via `has_extension_for` returning `true`). In the POC, all operations
are stubs (`todo!()`).

**Core ops implementation**:

| Core Op | CPU Implementation |
|---------|-------------------|
| `batched_gemm` | faer/cblas GEMM via `BgemmBackend` trait |
| `reduce` | `reduce_axis` from strided-kernel |
| `trace` | `reduce_trace_axes` from strided-einsum2 |
| `permute` | `StridedView::permute` + `copy_into` from strided-kernel |
| `anti_trace` | Scatter-add loop (new, for AD backward) |
| `anti_diag` | Write-to-diagonal loop (new, for AD backward) |

**Extended ops implementation**:

| Extended Op | CPU Implementation |
|------------|-------------------|
| `contract` | strided-einsum2 pipeline (fusability + GEMM) |
| `elementwise_mul` | `zip_map2_into` from strided-kernel |

#### GEMM Backend Selection (Compile-Time, Future)

```toml
# tenferro-prims/Cargo.toml (planned)
[features]
default = ["faer"]
faer = ["dep:faer"]
cblas = ["dep:cblas-sys"]

[dependencies]
faer = { version = "...", optional = true }
cblas-sys = { version = "...", optional = true }
```

| Feature | GEMM source | Use case |
|---------|------------|----------|
| `faer` (default) | Pure Rust, zero external deps | Standalone apps, guaranteed build |
| `cblas` | Requires `cblas-src` or `cblas-inject` | HPC, Julia integration |

When `cblas` is selected, the actual CBLAS implementation is provided by the
downstream user:

- **`cblas-src`**: Links OpenBLAS or MKL (standalone Rust apps)
- **`cblas-inject`**: Julia injects `libblastrampoline` symbols at runtime

`tenferro-prims` depends only on `cblas-sys` function signatures and is
agnostic to the CBLAS provider.

#### CPU Contraction Plan (Future, for Contract extended op)

The CPU plan caches analysis results from strided-einsum2. When
`has_extension_for::<T>(Extension::Contract)` returns `true`, the
Contract variant of `PrimDescriptor` produces this plan:

```rust
pub struct CpuContractionPlan {
    /// Canonical mode ordering (left, contracted, batch for A; etc.)
    a_perm: Vec<usize>,
    b_perm: Vec<usize>,
    c_perm: Vec<usize>,

    /// Fusability results from try_fuse_group
    a_fusable: bool,
    b_fusable: bool,

    /// Dimension sizes after classification
    batch_size: usize,
    left_size: usize,
    right_size: usize,
    contract_size: usize,

    /// GEMM dispatch
    gemm: GemmDispatch,  // Faer | Cblas | Naive

    /// Pre-computed workspace size
    workspace_size: usize,

    /// Element-wise bypass flag (no contraction axes -> Hadamard product)
    elementwise_bypass: bool,

    /// Blocking plan from strided-kernel (for cache-optimized iteration)
    blocking: Option<BlockingPlan>,
}
```

#### CPU Contraction Pipeline (strided-einsum2, Future, for Contract extended op)

The six-step pipeline from strided-einsum2, used when the Contract
extended operation is available. This fused version internally calls
batched_gemm after preprocessing:

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
   If fusable -> zero-copy metadata extraction.
   If not -> allocate col-major buffer and copy.

5. GEMM dispatch
   Call selected backend: faer::bgemm or cblas::dgemm/zgemm.
   Naive loop fallback for non-Scalar types (integers, tropical).

6. Copy-back
   If output was non-contiguous, copy from internal buffer back to
   the original strided destination.
```

Steps 1-4 are analyzed during `plan_contraction`. Steps 5-6
are executed during `contract`.

### GPU Backend (Future)

GPU support via cuTENSOR/hipTensor is planned but not in the POC. The
design involves:

#### GPU Backend Discovery (Future, via tenferro-device)

The caller (Julia, Python, or standalone Rust) will provide the path to
the shared library. This avoids:
- Requiring separate builds for NVIDIA vs AMD
- Automatic search finding wrong library versions
- Forcing users to set environment variables

```rust
// Future tenferro-device additions
pub struct BackendRegistry {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
}

impl BackendRegistry {
    pub fn new() -> Self { ... }  // CPU only
    pub fn load_cutensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn load_hiptensor(&mut self, path: &str) -> Result<()> { ... }
    pub fn available_devices(&self) -> Vec<Device> { ... }
}
```

#### GPU Vtable (Future, in tenferro-device)

cuTENSOR and hipTensor have nearly identical C APIs. A single function
pointer table will abstract over both:

```rust
struct TensorLibVtable {
    // Handle lifecycle
    create_handle: Symbol<unsafe extern "C" fn(*mut *mut c_void) -> i32>,
    destroy_handle: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Tensor descriptor
    create_tensor_descriptor: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, desc: *mut *mut c_void,
        num_modes: u32, extent: *const i64, stride: *const i64,
        data_type: u32, alignment: u32,
    ) -> i32>,
    destroy_tensor_descriptor: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Contraction
    create_contraction: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, desc: *mut *mut c_void,
        desc_a: *mut c_void, modes_a: *const i32, op_a: u32,
        desc_b: *mut c_void, modes_b: *const i32, op_b: u32,
        desc_c: *mut c_void, modes_c: *const i32, op_c: u32,
        desc_d: *mut c_void, modes_d: *const i32,
        compute: *mut c_void,
    ) -> i32>,

    // Plan
    create_plan: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, plan: *mut *mut c_void,
        desc: *mut c_void, pref: *mut c_void, ws_size: u64,
    ) -> i32>,
    destroy_plan: Symbol<unsafe extern "C" fn(*mut c_void) -> i32>,

    // Execution
    contract: Symbol<unsafe extern "C" fn(
        handle: *mut c_void, plan: *mut c_void,
        alpha: *const c_void, a: *const c_void, b: *const c_void,
        beta: *const c_void, c: *const c_void, d: *mut c_void,
        workspace: *mut c_void, ws_size: u64, stream: *mut c_void,
    ) -> i32>,

    // Permutation, reduction, elementwise (same pattern)
    create_permutation: Symbol<...>,
    permute: Symbol<...>,
    create_reduction: Symbol<...>,
    reduce_exec: Symbol<...>,
    create_elementwise_binary: Symbol<...>,
    elementwise_binary_exec: Symbol<...>,

    // Workspace
    estimate_workspace_size: Symbol<...>,

    // Cleanup
    destroy_operation_descriptor: Symbol<...>,
}

impl TensorLibVtable {
    fn load_cutensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"cutensorCreate")?,
            destroy_handle: lib.get(b"cutensorDestroy")?,
            create_contraction: lib.get(b"cutensorCreateContraction")?,
            contract: lib.get(b"cutensorContract")?,
            // ...
        })
    }

    fn load_hiptensor(lib: &Library) -> Result<Self> {
        Ok(Self {
            create_handle: lib.get(b"hiptensorCreate")?,
            destroy_handle: lib.get(b"hiptensorDestroy")?,
            create_contraction: lib.get(b"hiptensorCreateContraction")?,
            contract: lib.get(b"hiptensorContract")?,
            // ...
        })
    }
}
```

The `GpuBackend` struct will wrap the vtable:

```rust
pub struct GpuBackend {
    vtable: TensorLibVtable,
    handle: *mut c_void,
    _lib: Library,  // prevent unloading
}
```

#### GPU Plan Caching (Future, in tenferro-device)

Plans are expensive to create on GPU. A cache will avoid re-creation for
repeated contraction patterns (common in tensor network algorithms):

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

---

## Layer 4: Einsum Engine (tenferro-einsum)

High-level einsum API on `Tensor<T>`. Internally extracts `view()` from
each tensor and delegates binary contractions to `tenferro-prims`.

### Public API

The POC defines three API levels, all same-type `T` (no mixed-type inputs):

```rust
/// Execute einsum using string notation.
///
/// Parses the subscript string, optimizes the contraction order, and
/// executes the contraction. The backend is selected automatically from
/// the tensors' device.
///
/// Parentheses in the subscript string specify contraction order
/// explicitly (e.g., "ij,(jk,kl)->il" contracts B and C first).
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with pre-built Subscripts.
///
/// Avoids re-parsing the subscript string on each call. Useful when
/// the same contraction pattern is applied to tensors of varying shapes.
pub fn einsum_with_subscripts<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with a pre-optimized ContractionTree.
///
/// Avoids both subscript parsing and contraction order optimization.
/// Ideal for hot loops where the same contraction is executed repeatedly
/// on tensors of the same shape.
pub fn einsum_with_plan<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

The three levels form a progressive optimization ladder:

| Level | Avoids | Use case |
|-------|--------|----------|
| `einsum` | Nothing | One-off contractions, prototyping |
| `einsum_with_subscripts` | String parsing | Same pattern, varying shapes |
| `einsum_with_plan` | Parsing + tree optimization | Hot loops, fixed shapes |

Each allocating function has an accumulating counterpart that writes into
a pre-allocated output buffer with BLAS-style scaling:

```rust
/// output = alpha * einsum(operands) + beta * output
pub fn einsum_into<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
) -> Result<()>;

pub fn einsum_with_subscripts_into<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
) -> Result<()>;

pub fn einsum_with_plan_into<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
) -> Result<()>;
```

| Level | Avoids | Use case |
|-------|--------|----------|
| `einsum_into` | Nothing | One-off contractions with accumulation |
| `einsum_with_subscripts_into` | String parsing | Same pattern, varying shapes, with accumulation |
| `einsum_with_plan_into` | Parsing + tree optimization | Hot loops with pre-allocated output buffer |

The `_into` variants are critical for hot-loop performance: they eliminate
output allocation per call and enable accumulation semantics (`beta != 0`)
that directly map to the underlying `TensorPrims::execute` alpha/beta pattern.

### Subscripts

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
#[derive(Debug, Clone)]
pub struct Subscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer label arrays.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self { ... }

    /// Parse subscripts from NumPy/PyTorch-style string notation.
    ///
    /// Each character (a-z, A-Z) represents a dimension label.
    /// Parentheses specify contraction order explicitly.
    ///
    /// Examples:
    ///   "ij,jk->ik"           -- matrix multiplication
    ///   "ii->i"               -- diagonal extraction
    ///   "ij,(jk,kl)->il"      -- contract B*C first, then A
    pub fn parse(notation: &str) -> Result<Self> { ... }
}
```

### ContractionTree

```rust
/// Contraction tree determining pairwise contraction order for N-ary einsum.
pub struct ContractionTree { ... }

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self>;

    /// Manually build a contraction tree from a pairwise contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}
```

### String Notation Examples

```rust
// Matrix multiplication
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();

// Trace
let tr = einsum("ii->", &[&a]).unwrap();

// Outer product
let outer = einsum("i,j->ij", &[&v, &v]).unwrap();

// Batch GEMM
let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();

// Explicit contraction order with parentheses
let d = einsum("ij,(jk,kl)->il", &[&a, &b, &c]).unwrap();

// Integer label notation
let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let c = einsum_with_subscripts(&subs, &[&a, &b]).unwrap();

// Pre-optimized plan for hot loops
let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
for _ in 0..n_steps {
    let c = einsum_with_plan(&tree, &[&a, &b]).unwrap();
}
```

### N-ary Contraction (Internal)

For N > 2 inputs, the einsum engine uses contraction tree optimization
to find the optimal pairwise contraction order, then dispatches each
pairwise contraction through `TensorPrims` primitives.

#### Dispatch Strategy

```rust
/// Internal: dispatches to TensorPrims<A> backend.
/// T: HasAlgebra infers algebra A automatically.
fn einsum_impl<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>
where
    CpuBackend: TensorPrims<T::Algebra>,
{
    match operands.len() {
        0 => Err(Error::InvalidArgument("no inputs".into())),
        1 => single_tensor_op::<T>(operands[0], &subscripts),
        2 => binary_contraction::<T>(operands[0], operands[1], &subscripts),
        _ => {
            // N-ary: optimize contraction order, then execute pairwise
            let tree = ContractionTree::optimize(&subscripts, ...)?;
            execute_tree(&tree, operands)
        }
    }
}
```

#### Binary Contraction Decomposition

For each binary contraction, the engine chooses between:

**Path A — Extended Contract available** (`has_extension_for::<T>(Contract)`):
```
  let desc = PrimDescriptor::Contract { modes_a, modes_b, modes_c };
  let plan = Backend::plan::<T>(&desc, &shapes)?;
  Backend::execute(&plan, alpha, &[&a, &b], beta, &mut c)?;
  → backend handles diag, trace, permute, GEMM internally
```

**Path B — Core ops only** (decompose into primitives):
```
  1. diag(a, paired_axes)        // zero-copy stride trick on Tensor<T>
  2. diag(b, paired_axes)        // zero-copy stride trick on Tensor<T>
  3. trace/reduce(a, trace_axes) // TensorPrims::trace or TensorPrims::reduce
  4. trace/reduce(b, trace_axes)
  5. permute_view(a, canonical)  // zero-copy metadata on Tensor<T>
  6. permute_view(b, canonical)
  7. batched_gemm(a, b, c)       // plan + execute BatchedGemm
  8. permute(c, output_order)    // plan + execute Permute (if needed)
```

**Example: `"iij,jkk->ik"`**:
```
  1. a' = a.diagonal([(0,1)])    → a'[i,j]    (zero-copy)
  2. b' = b.diagonal([(1,2)])    → b'[j,k]    (zero-copy)
  3. (no trace/reduce needed)
  4. a'' = a'.permute([i,j])     → canonical   (zero-copy)
  5. b'' = b'.permute([j,k])     → canonical   (zero-copy)
  6. batched_gemm(a'', b'', c)   → c[i,k]     (computation)
```

#### Single-Tensor Decomposition (Unary Operations)

Unary einsum operations are decomposed into the universal set primitives:

| Einsum | Decomposition |
|--------|--------------|
| `ii→` (full trace) | `trace(A, [(0,1)])` |
| `ii→i` (diagonal) | `diag(A, [(0,1)])` (zero-copy on Tensor) |
| `iij→j` (partial trace) | `diag(A, [(0,1)])` → `reduce(A', axis=0)` |
| `ij→ji` (transpose) | `permute(A, [1,0])` |
| `i→ij` (broadcast) | `repeat(A, j_dim)` (zero-copy on Tensor) |
| `i→ii` (embed diagonal) | `anti_diag(A, [(0,1)])` |
| `ij→i` (sum axis) | `reduce(A, axis=1, ReduceOp::Sum)` |

#### Backward Decomposition (VJP)

Each forward operation has a clean adjoint:

```
Forward:  C[i,k] = einsum("ij,jk->ik", A, B)
         = batched_gemm(A[i,j], B[j,k])

Backward: ∂A[i,j] = batched_gemm(∂C[i,k], B^T[k,j])
          ∂B[j,k] = batched_gemm(A^T[j,i], ∂C[i,k])

Forward:  y[j] = einsum("iij->j", A)
         = reduce(diag(A, [(0,1)]), axis=0)

Backward: ∂A = anti_diag(repeat(∂y, i_dim), [(0,1)])
```

#### Optimizations from strided-opteinsum (Future)

- **Borrowed-view passthrough**: Leaf nodes return borrows, not clones.
- **Permutation-only detection**: Metadata-only transformation for
  nodes that only permute axes (no contraction).
- **Buffer pool** (opt-in): Reuse intermediate buffers across pairwise
  contractions. Trades memory for speed.
- **Direct root write**: Final contraction writes into user's output
  buffer (no extra allocation).

### Algebra Dispatch

Backend selection is determined by the algebra parameter `A` in
`TensorPrims<A>`. The `HasAlgebra` trait on scalar types enables automatic
inference:

```rust
// T: HasAlgebra → infers algebra A
// Backend is selected based on A + device:
//
// impl TensorPrims<Standard> for CpuBackend → faer/cblas GEMM
// impl TensorPrims<MaxPlus> for CpuBackend  → tropical-gemm (tenferro-tropical)
// impl TensorPrims<Standard> for GpuBackend → cuTENSOR/hipTensor [future]
// impl TensorPrims<MyAlgebra> for CpuBackend → user-provided kernels
```

Tropical semiring types (`MaxPlus`, `MinPlus`, `MaxMul`) are in the separate
`tenferro-tropical` crate. They implement `TensorPrims<MaxPlus> for CpuBackend`
with SIMD-optimized tropical-gemm via `TypeId`-based runtime dispatch
(from omeinsum-rs).

### Backward Pass (VJP/JVP, Future)

The `TensorPrims<A>` design provides clean adjoint pairs for AD:

| Forward op | VJP rule (backward) |
|-----------|-------------------|
| `trace(A)` | `∂A = anti_trace(∂y)` |
| `diag(A)` | `∂A = anti_diag(∂y)` |
| `reduce(A, dim)` | `∂A = repeat(∂y, dim)` |
| `permute(A, p)` | `∂A = permute(∂y, p⁻¹)` |
| `batched_gemm(A, B)` | `∂A = gemm(∂C, B^T)`, `∂B = gemm(A^T, ∂C)` |

Contraction VJP for automatic differentiation:

```rust
/// VJP: grad_A = batched_gemm(grad_C, B^T), grad_B = batched_gemm(A^T, grad_C)
pub fn contract_vjp<T: ScalarBase>(
    a: &Tensor<T>, modes_a: &[u32],
    b: &Tensor<T>, modes_b: &[u32],
    grad_c: &Tensor<T>, modes_c: &[u32],
) -> Result<(Tensor<T>, Tensor<T>)>;

/// JVP: dC = batched_gemm(dA, B) + batched_gemm(A, dB)  (Leibniz rule)
pub fn contract_jvp<T: ScalarBase>(...) -> Result<Tensor<T>>;
```

Both VJP and JVP go through `TensorPrims` primitives, so they work on
CPU and GPU uniformly. The adjoint operations (`anti_trace`, `anti_diag`)
are in the core `TensorPrims` trait, ensuring GPU support for backward passes.

---

## Compile-Time vs Runtime Decision Summary

| Choice | Mechanism | Rationale |
|--------|-----------|-----------|
| GPU vendor (cuTENSOR/hipTensor) | **Runtime** dlopen (future) | Single binary for all platforms; Julia/Python inject .so path |
| CPU GEMM (faer/cblas) | **Compile-time** feature (future) | Fundamentally different linking (pure Rust vs C ABI) |
| Elementwise ops | **Enum-based** in TensorPrims; closures via strided-kernel | cuTENSOR-compatible for GPU; custom closures via strided-kernel (CPU only) |
| libloading | **Always ON** (future, in tenferro-device) | Lightweight, no overhead when GPU absent |
| .so path | **Caller-injected** (future, via tenferro-device) | No auto-search; Julia/Python manage library versions |

---

## Crate Dependency Graph

The POC has five crates with the following dependency structure:

```
strided-rs (independent workspace):
strided-traits -> strided-view -> strided-kernel

tenferro-rs (workspace, depends on strided-rs):

tenferro-device              tenferro-algebra
  (LogicalMemorySpace +        (HasAlgebra trait,
   ComputeDevice, Error,       Semiring trait,
   Result)                     Standard type)
  (depends on: strided-view
   for StridedError,
   thiserror)
                              (depends on: strided-traits,
                               num-complex)
        │                              │
        ├──────────────┐       ┌───────┤
        │              ↓       ↓       │
        │         tenferro-prims       │
        │           (TensorPrims<A>,   │
        │            PrimDescriptor,   │
        │            CpuBackend,       │
        │            Extension,        │
        │            ReduceOp)         │
        │           (depends on:       │
        │            tenferro-device,  │
        │            tenferro-algebra, │
        │            strided-view,     │
        │            strided-traits)   │
        │              │               │
        ↓              │               │
   tenferro-tensor     │               │
     (Tensor<T> =      │               │
      DataBuffer       │               │
      + dims/strides/  │               │
      offset, view ops)│               │
     (depends on:      │               │
      tenferro-device, │               │
      strided-view,    │               │
      strided-traits,  │               │
      num-traits)      │               │
        │              │               │
        └──────┬───────┘       ┌───────┘
               ↓               ↓
          tenferro-einsum
            (einsum, einsum_with_subscripts,
             einsum_with_plan,
             Subscripts, ContractionTree)
            (depends on: tenferro-device,
             tenferro-algebra,
             tenferro-prims,
             tenferro-tensor,
             strided-traits)
```

Future crates (not in POC):
- `tenferro-tropical` -- Tropical algebra types, TensorPrims<MaxPlus> for CpuBackend
- `tenferro-linalg` -- SVD, QR, eigen (CPU: faer, GPU: cuSOLVER)
- `tenferro-autograd` -- TrackedTensor, DualTensor, VJP/JVP
- `tenferro-capi` -- C FFI for Julia/Python integration

---

## References

- [cuTENSOR API Reference](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)
- [hipTensor API Reference](https://rocm.docs.amd.com/projects/hipTensor/en/latest/api-reference/api-reference.html)
- [cudarc cuTENSOR bindings](https://github.com/coreylowman/cudarc/tree/main/src/cutensor)
- [omeinsum-rs cuTENSOR wrapper](https://github.com/tensor4all/omeinsum-rs) (internal)
- [strided-rs einsum2 pipeline](https://github.com/tensor4all/strided-rs) (internal)
- [tenferro-rs POC](https://github.com/tensor4all/tenferro-rs/)
- [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md)
- [tenferro Design](./tenferro_design.md)
- [tenferro Unified Tensor Backend Design (ecosystem)](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro_unified_tensor_backend.md)
