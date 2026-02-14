# tenferro Design: Detailed Crate API Designs

> **Ecosystem overview**: See the [tenferro Unified Tensor Backend Design](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro_unified_tensor_backend.md) in tensor4all-meta for high-level architecture, crate structure, and future phase plans.
>
> **Companion documents** (in this repo):
> - [Einsum Internal Design](./tenferro_einsum_internal_design.md) — detailed internal design of tenferro-prims and tenferro-einsum
> - [Einsum Algorithm Comparison](./einsum_algorithm_comparison.md) — strided-rs vs omeinsum-rs optimization comparison
> - [Async/Ownership Integration Design](../plans/2026-02-14-tensor-async-ownership-integration-design.md) — validated design decisions for CompletionEvent + TensorView

---

## Phase 1: Dense Array Foundation (POC)

### tenferro-device

The device crate provides shared infrastructure used across all tenferro crates.

```rust
/// Logical memory space where tensor buffers reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalMemorySpace {
    /// Always-available host memory.
    MainMemory,
    /// GPU-accessible memory space. A space may be attached to one or many GPUs.
    GpuMemory { space_id: usize },
}

/// Compute device where kernels execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputeDevice {
    Cpu { device_id: usize },
    Cuda { device_id: usize },
    Hip { device_id: usize },
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeDevice::Cpu { device_id } => write!(f, "cpu:{device_id}"),
            ComputeDevice::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            ComputeDevice::Hip { device_id } => write!(f, "hip:{device_id}"),
        }
    }
}

/// Operation kind used for capability filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind { Contract, BatchedGemm, Reduce, Trace, Permute, ElementwiseMul }

/// Returns executable compute devices in descending preference order.
pub fn preferred_compute_devices(
    space: LogicalMemorySpace,
    op_kind: OpKind,
) -> Result<Vec<ComputeDevice>>;
```

**Error types** using `thiserror`:

```rust
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

pub type Result<T> = std::result::Result<T, Error>;
```

**Dependencies**: `strided-view` (for `StridedError`), `thiserror`.

**Note**: `BackendRegistry`, `GpuBackend`, and `TensorLibVtable` are **not** in the POC. They are planned for future GPU support (see [GPU Strategy](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/tenferro_unified_tensor_backend.md#gpu-strategy) in the ecosystem doc).

### tenferro-algebra

Minimal algebra foundation for `TensorPrims<A>`. Provides the `HasAlgebra`
trait for automatic algebra inference and the `Standard` type for standard
arithmetic.

```rust
/// Maps a scalar type T to its default algebra A.
/// Enables automatic inference: Tensor<f64> → Standard, Tensor<MaxPlus<f64>> → MaxPlus.
pub trait HasAlgebra {
    type Algebra;
}

/// Standard arithmetic algebra (add = +, mul = *).
pub struct Standard;

impl HasAlgebra for f64 { type Algebra = Standard; }
impl HasAlgebra for f32 { type Algebra = Standard; }
impl HasAlgebra for Complex64 { type Algebra = Standard; }
// etc.

/// Semiring trait for algebra-generic operations.
pub trait Semiring {
    type Scalar: ScalarBase;
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
```

**Note**: Tropical types (`MaxPlus`, `MinPlus`, `MaxMul`) are in the separate
`tenferro-tropical` crate, not here. This separation proves that the algebra
extension mechanism works for external crates.

### tenferro-prims

The central protocol layer. Defines `TensorPrims<A>` parameterized by algebra `A`,
with a cuTENSOR-compatible describe → plan → execute pattern.

> **Detailed design**: See [tenferro Einsum Internal Design](./tenferro_einsum_internal_design.md)
> for the full internal design including CPU contraction pipeline details.

#### Design Overview

GiggleLiu proposed a **universal set** of primitive operations plus an
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
        diag → trace/reduce → permute → batched_gemm → permute
```

**Dispatch is dynamic**: `has_extension_for::<T>(ext)` queries at runtime
whether a specific extended operation is available for scalar type `T`.
This is important because:
- GPU backends are loaded at runtime (dlopen)
- cuTENSOR supports `f32`/`f64`/Complex but not tropical types
- CPU backends may support `contract` for `f64` (faer) but not for custom types

Note: `diag` (diagonal extraction) and `repeat` (broadcast) are **zero-copy
stride tricks** handled at the `Tensor<T>` level (see below), not in `TensorPrims`.

#### Adjoint Pairs for AD

The core operations form adjoint pairs, enabling clean VJP/JVP rules:

| Forward | Backward (adjoint) |
|---------|-------------------|
| trace | anti_trace |
| diag (on Tensor) | anti_diag |
| reduce | repeat (on Tensor) |
| permute | inverse permute |
| batched_gemm | Leibniz rule |

#### Key Types

```rust
/// Describes any TensorPrims operation (cuTENSOR pattern: describe → plan → execute).
pub enum PrimDescriptor {
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    Reduce { modes_a: Vec<u32>, modes_c: Vec<u32>, op: ReduceOp },
    Trace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    Permute { modes_a: Vec<u32>, modes_b: Vec<u32> },
    AntiTrace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    AntiDiag { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    // Extended
    Contract { modes_a: Vec<u32>, modes_b: Vec<u32>, modes_c: Vec<u32> },
    ElementwiseMul,
}

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

/// Extended operation identifiers for dynamic capability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension { Contract, ElementwiseMul }
```

#### TensorPrims\<A\> Trait

```rust
/// Backend trait parameterized by algebra A.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, anti_trace,
/// anti_diag) must be implemented. Extended ops (contract, elementwise_mul)
/// have default implementations that decompose into core ops.
///
/// The algebra parameter A enables extensibility: external crates can
/// implement TensorPrims<MyAlgebra> for CpuBackend (orphan rule compatible).
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
    /// Enables dynamic dispatch: GPU may support Contract for f64 but not
    /// for tropical types.
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}
```

#### CpuBackend

```rust
pub struct CpuBackend;

/// Standard arithmetic on CPU (faer GEMM for f64/f32, naive for others).
impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;

    fn plan<T: ScalarBase>(desc: &PrimDescriptor, shapes: &[&[usize]])
        -> Result<CpuPlan<T>> { ... }

    fn execute<T: ScalarBase>(plan: &CpuPlan<T>, ...) -> Result<()> { ... }

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
    Contract { /* strided-einsum2 cached analysis */ },
    ElementwiseMul,
    ...
}
```

**Tropical backend** (in separate `tenferro-tropical` crate):

```rust
// tenferro-tropical crate — external, proves extensibility
pub struct MaxPlus;

impl HasAlgebra for MaxPlus<f64> { type Algebra = MaxPlus; }

/// Tropical GEMM on CPU (SIMD-optimized tropical-gemm kernel).
impl TensorPrims<MaxPlus> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool {
        false  // tropical uses core ops decomposition, no fused contract
    }
    ...
}
```

**User-defined algebra** (in user crate):

```rust
// User crate — same pattern as tenferro-tropical
struct MyScalar(f64);
struct MyAlgebra;

impl ScalarBase for MyScalar { ... }
impl HasAlgebra for MyScalar { type Algebra = MyAlgebra; }

impl TensorPrims<MyAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = MyPlan<T>;
    ...
}

// Just works:
let a = Tensor::<MyScalar>::zeros(&[3, 4], ...);
einsum("ij,jk->ik", &[&a, &b])?;  // MyAlgebra auto-inferred
```

**Backend implementation matrix**:

| Backend | Algebra | Extended ops | Notes |
|---------|---------|-------------|-------|
| CpuBackend | Standard | Contract, ElementwiseMul | faer/cblas GEMM |
| CpuBackend | MaxPlus | None (decompose to core) | tropical-gemm SIMD |
| CpuBackend | MyAlgebra | User choice | User-provided kernels |
| GpuBackend [future] | Standard | Contract, ElementwiseMul | cuTENSOR/hipTensor |
| GpuBackend [future] | MaxPlus | None | No cuTENSOR tropical support |

**Usage examples**:

```rust
use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp, Standard};
use strided_view::StridedArray;

// Plan + execute: GEMM
let desc = PrimDescriptor::BatchedGemm { batch_dims: vec![], m: 3, n: 5, k: 4 };
let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();

// Plan + execute: Reduction
let desc = PrimDescriptor::Reduce { modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum };
let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
CpuBackend::execute(&plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

// Dynamic extension check
if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    let desc = PrimDescriptor::Contract { modes_a: vec![0,1], modes_b: vec![1,2], modes_c: vec![0,2] };
    let plan = CpuBackend::plan::<f64>(&desc, &shapes).unwrap();
    CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
}
```

**No Metal (Apple GPU) support**: M-series CPUs are fast enough for our
workloads (tensor network algorithms). Metal lacks a cuTENSOR-equivalent
tensor contraction library, requiring reshape+matmul decomposition that
would be slow for high-rank tensors. Not worth the implementation cost.

### tenferro-tensor

`Tensor<T>` is the core data type. It wraps a `DataBuffer<T>` with
shape/stride metadata and provides zero-copy view operations.

```rust
/// Memory ordering for new allocations only.
/// Not stored on the tensor — strides fully describe the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    ColumnMajor,  // First dimension has stride 1 (Fortran/Julia)
    RowMajor,     // Last dimension has stride 1 (C/NumPy)
}

/// Owned data buffer, device-aware.
pub enum DataBuffer<T> {
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional dense tensor.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,  // None = ready, Some = pending accelerator work
}
```

**Constructors**:

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self>;
    pub fn from_strided_array(array: StridedArray<T>) -> Self;
}
```

**Metadata**:

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn dims(&self) -> &[usize];
    pub fn strides(&self) -> &[isize];
    pub fn ndim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn logical_memory_space(&self) -> LogicalMemorySpace;
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice>;
    pub fn set_preferred_compute_device(&mut self, d: Option<ComputeDevice>);
    pub fn effective_compute_devices(&self, op: OpKind) -> Result<Vec<ComputeDevice>>;
}
```

**Explicit memory movement**:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Asynchronous explicit move between logical memory spaces.
    ///
    /// - Same source/destination space: always Ok, zero-copy no-op.
    /// - Different spaces: explicit transfer (never implicit in ops).
    pub fn to_memory_space_async(&self, dst: LogicalMemorySpace) -> Result<Tensor<T>>;
}
```

**View operations** (zero-copy, modify only metadata):

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Immutable strided view for use with TensorPrims or strided-kernel.
    pub fn view(&self) -> StridedView<'_, T>;

    /// Mutable strided view.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T>;

    /// Permute (reorder) dimensions. Zero-copy.
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;

    /// Broadcast to a larger shape. Zero-copy via stride 0.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>>;

    /// Extract diagonal view by merging pairs of axes. Zero-copy stride trick.
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;

    /// Reshape. Requires contiguous data.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>>;
}
```

**Data operations**:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Return a contiguous copy in the given memory order.
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;

    /// Consume this tensor and return a contiguous version.
    /// If already contiguous in the requested order, returns self
    /// without copying or allocating.
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;

    /// Check if tensor data is contiguous in memory.
    pub fn is_contiguous(&self) -> bool;

    /// Return a tensor with complex-conjugated elements.
    ///
    /// For real types (f32, f64), returns a copy unchanged.
    /// For complex types (Complex32, Complex64), negates the imaginary part.
    pub fn conj(&self) -> Tensor<T>;

    /// Consume this tensor and return one with complex-conjugated elements.
    /// Like conj() but may reuse the buffer (no allocation).
    pub fn into_conj(self) -> Tensor<T>;

    /// Wait for any pending accelerator computation to complete.
    /// No-op for CPU tensors or already-completed operations.
    pub fn wait(&self);

    /// Check if data is ready without blocking.
    /// Always returns true for CPU tensors.
    pub fn is_ready(&self) -> bool;
}
```

**Key differences from original design**:
- `DataBuffer<T>` is an enum in `tenferro-tensor` (not a separate crate, no `Arc` wrapping)
- Fields: `dims` (`Vec<usize>`), `strides` (`Vec<isize>`), `offset` (`isize`) -- no `SmallVec`
- `MemoryOrder` is only used at allocation time, **not stored** on the tensor
- Bridge to strided-rs via `view()` / `view_mut()` (not `as_strided_view()`)
- No `TensorMeta` struct -- not needed since `TensorPrims` uses `StridedView` directly
- No type casting methods yet (future work)

**Creating and using tensors**:

```rust
use tenferro_tensor::{Tensor, MemoryOrder};
use tenferro_device::LogicalMemorySpace;

// Create tensors
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let m = Tensor::<f64>::from_slice(&data, &[2, 3], MemoryOrder::ColumnMajor).unwrap();

// Zero-copy transpose
let mt = m.permute(&[1, 0]).unwrap();
assert_eq!(mt.dims(), &[3, 2]);

// Broadcasting (zero-copy via stride 0)
let col = Tensor::<f64>::ones(&[3, 1], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let expanded = col.broadcast(&[3, 4]).unwrap();
assert_eq!(expanded.dims(), &[3, 4]);

// Get strided views for low-level operations
let view = a.view();
let mut b = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
let view_mut = b.view_mut();
```

For **custom element-wise closures** (arbitrary user functions not in the
`TensorPrims` enum), use strided-kernel directly via `view()`:

```rust
// Custom closures: use strided-kernel directly (CPU only)
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out.view_mut(), &a_view, &b_view, |a, b| a * b + 1.0);
```

### tenferro-einsum

High-level einsum API on `Tensor<T>`. Supports string notation with
parenthesized contraction order, integer label notation, and pre-optimized
contraction trees.

**Subscripts**:

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
#[derive(Debug, Clone)]
pub struct Subscripts {
    pub inputs: Vec<Vec<u32>>,
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create from integer label arrays.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self;

    /// Parse from string notation: "ij,jk->ik"
    /// Supports parenthesized order: "ij,(jk,kl)->il"
    pub fn parse(notation: &str) -> Result<Self>;
}
```

**ContractionTree**:

```rust
pub struct ContractionTree { /* internal */ }

impl ContractionTree {
    /// Automatically optimize contraction order (cost-based heuristic).
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self>;

    /// Manually specify pairwise contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}
```

**Three API levels**:

```rust
/// Level 1: String notation — parse + optimize + execute.
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Level 2: Pre-built subscripts — optimize + execute.
pub fn einsum_with_subscripts<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Level 3: Pre-optimized tree — execute only.
pub fn einsum_with_plan<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

| Level | Parsing | Optimization | Execution | Use case |
|-------|---------|-------------|-----------|----------|
| `einsum` | Yes | Yes | Yes | One-off, convenience |
| `einsum_with_subscripts` | Cached | Yes | Yes | Same pattern, varying shapes |
| `einsum_with_plan` | Cached | Cached | Yes | Hot loops, same shapes |

**Consuming variants** (operands moved, buffer reuse possible):

```rust
/// Level 1: Consuming — parse + optimize + execute, input buffers may be reused.
pub fn einsum_owned<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;

/// Level 2: Consuming — optimize + execute.
pub fn einsum_with_subscripts_owned<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;

/// Level 3: Consuming — execute only.
pub fn einsum_with_plan_owned<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;
```

Input tensors are moved into the function. The implementation may reuse
their buffers for intermediate results or the final output, avoiding
allocation in contraction trees. Buffer reuse is deterministic — the
compiler guarantees no other references to the moved tensors exist.

**User examples**:

```rust
use tenferro_einsum::einsum;
use tenferro_tensor::{Tensor, MemoryOrder};
use tenferro_device::LogicalMemorySpace;

let col = MemoryOrder::ColumnMajor;
let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();

// Matrix multiplication
let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();

// Trace
let tr = einsum("ii->", &[&a]).unwrap();

// Batch matrix multiplication
let ba = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
let bb = Tensor::<f64>::zeros(&[10, 4, 5], LogicalMemorySpace::MainMemory, col);
let bc = einsum("bij,bjk->bik", &[&ba, &bb]).unwrap();

// Explicit contraction order via parentheses
let d = einsum("ij,(jk,kl)->il", &[&a, &b, &c]).unwrap();

// Integer label notation (for programmatic use)
use tenferro_einsum::{einsum_with_subscripts, Subscripts};
let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let c = einsum_with_subscripts(&subs, &[&a, &b]).unwrap();

// Pre-optimized tree (hot loops)
use tenferro_einsum::ContractionTree;
let tree = ContractionTree::optimize(&subs, &[&[2, 2], &[2, 2]]).unwrap();
let c = einsum_with_plan(&tree, &[&a, &b]).unwrap();

// Consuming variant: operands are moved, buffers may be reused
use tenferro_einsum::einsum_owned;
let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
let y = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
let z = einsum_owned("ij,jk->ik", vec![x, y]).unwrap();
// x, y are moved — their buffers can be reused for intermediates
```

**Key differences from original design**:
- String-first API: `einsum("ij,jk->ik", &[&a, &b])` instead of integer labels as primary
- Parenthesized contraction order in string notation
- `Subscripts::parse()` handles string-to-integer conversion
- Nine API functions: three allocating (`einsum`, `einsum_with_subscripts`, `einsum_with_plan`), three accumulating (`einsum_into`, `einsum_with_subscripts_into`, `einsum_with_plan_into`) with BLAS-style `alpha`/`beta` scaling, and three consuming (`einsum_owned`, `einsum_with_subscripts_owned`, `einsum_with_plan_owned`) for buffer reuse
- `Tensor::into_contiguous(self)` and `Tensor::into_conj(self)` consuming variants avoid allocation when possible
- No mixed-type inputs: all inputs and output must be the same type `T`

---

## Future Considerations

### GPU/CPU overlap and asynchronous execution

The current `TensorPrims::execute` API is synchronous — it blocks until
the operation completes. For GPU backends, this leaves the CPU idle during
GPU computation:

```
GPU: [== einsum A ==][== einsum B ==]
CPU: [    idle       ][    idle       ]
```

With asynchronous execution, the CPU could prepare the next operands or
perform independent computation while the GPU is busy:

```
GPU: [== einsum A ==][== einsum B ==][== einsum C ==]
CPU:                  [prepare B     ][prepare C     ]
```

#### Chosen approach: Tensor embeds async state

Rather than introducing separate `einsum` / `einsum_async` functions, the
`Tensor<T>` struct itself carries an optional `CompletionEvent` that tracks
pending accelerator computation. This follows the PyTorch model where
accelerator tensors are always potentially async:

```rust
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,  // None = ready, Some = pending accelerator work
}
```

This enables transparent accelerator-to-accelerator operation chaining
without CPU synchronization:

```rust
// Accelerator operations return immediately with event attached
let c = einsum("ij,jk->ik", &[&a_gpu, &b_gpu])?;
//  → GPU submit, c.event = Some(event_1), returns immediately

let d = einsum("ij,jk->ik", &[&c, &e_gpu])?;
//  → detects c.event → sets up stream dependency → no CPU wait
//  → GPU submit, d.event = Some(event_2), returns immediately

// CPU data access triggers implicit synchronization
println!("{:?}", d.view());  // view() calls wait() internally
```

The `einsum` API does not change — the same function handles both sync
(CPU) and async (accelerator) execution transparently. For CPU tensors,
`event` is always `None` with zero overhead.

Key methods:

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Wait for any pending accelerator computation to complete.
    /// No-op for CPU tensors or already-completed operations.
    pub fn wait(&self) { ... }

    /// Check if data is ready without blocking.
    pub fn is_ready(&self) -> bool { ... }

    // Existing methods that access data auto-sync:
    // view(), view_mut(), conj(), contiguous() — all call wait() internally
}
```

#### Alternatives considered

1. **Separate `einsum_async` function**: Rejected — splits the API
   unnecessarily. A single `einsum` that works for both sync and async
   is simpler.

2. **Trait-based (`TensorArg` trait with `wait()`)**: More extensible
   (could also support lazy expressions), but adds API complexity
   (`&[&dyn TensorArg<T>]` or generics explosion). Can be introduced
   later if needed — implementing the trait for `Tensor<T>` is backward
   compatible.

3. **Contraction tree-level pipelining only**: Useful as an internal
   optimization within N-ary einsum, but does not help when the user
   chains independent einsum calls manually. The Tensor-embedded event
   approach handles both cases.

#### Applicability beyond GPU: multi-threaded CPU parallelism

The `CompletionEvent` mechanism is not limited to GPU/accelerator backends.
It applies equally to **multi-threaded CPU execution**:

- **Contraction tree parallelism**: Independent subtrees of an N-ary einsum
  can be dispatched to different CPU threads. Each subtree result is a
  `Tensor` with a `CompletionEvent` that completes when the thread finishes.
  The parent contraction waits on both child events before proceeding.

- **User-level parallelism**: Independent `einsum` calls can run on separate
  threads. Passing the resulting `Tensor` (with pending event) to a
  subsequent `einsum` automatically chains via event dependencies — no
  explicit synchronization needed.

- **Compute-device model extension**: `ComputeDevice::Cpu { device_id }`
  matches `Cuda/Hip` and supports multiple CPU execution contexts.
  `device_id: 0` represents the default global thread pool (all cores).
  Higher IDs can map to Rayon `ThreadPool` instances bound to specific
  core sets via `core_affinity`, enabling NUMA-aware and cache-local
  execution. Memory placement remains independent via
  `LogicalMemorySpace`. The `CompletionEvent` mechanism remains unchanged.

This means the same `einsum` API and `wait()` / `is_ready()` interface
covers GPU async, CPU multi-thread, and potentially other execution models
(FPGA, distributed) without modification.

**Current status**: The `event` field is present in the POC `Tensor<T>`
struct as `Option<CompletionEvent>` (placeholder type) to signal the design
intent. Actual async execution will be implemented with accelerator and
multi-threaded backends.

### Tensor / TensorView ownership split

#### Motivation

The POC defines `permute(&self) -> Tensor<T>` as "zero-copy," but `Tensor<T>`
owns its `DataBuffer<T>`. For true zero-copy view operations, the new tensor
must share the original's data buffer. Two main approaches exist:

- **Arc-based sharing** (PyTorch model): Single `Tensor` type, buffer wrapped
  in `Arc`. Simple API, but buffer uniqueness is a runtime check.
- **Tensor / TensorView split** (ndarray model, `String` / `&str` pattern):
  Owned `Tensor` and borrowed `TensorView<'a>`. Buffer uniqueness is a
  compile-time guarantee.

#### Chosen direction: Tensor + TensorView

The Tensor/TensorView split is more idiomatic Rust and provides stronger
compile-time guarantees for buffer reuse:

```rust
/// Owned tensor. Holds exclusive ownership of the data buffer.
/// Can be moved (consumed) for buffer reuse.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,
}

/// Borrowed tensor view. References a Tensor's data buffer.
/// Zero-copy, lifetime-tied to the source Tensor.
///
/// Public API always returns TensorView with event = None
/// (wait is performed before construction). The event field
/// is used only by crate-internal as_operand_view() for
/// accelerator pipeline chaining.
pub struct TensorView<'a, T: ScalarBase> {
    data: &'a DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<&'a CompletionEvent>,  // Public API: None; internal operand path: propagated
}
```

#### API design

**Tensor (owned) methods:**

```rust
impl<T: ScalarBase> Tensor<T> {
    // --- Public: Borrow → TensorView (zero-copy, waits if pending) ---
    fn view(&self) -> TensorView<'_, T>;
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'_, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'_, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'_, T>>;

    // --- Internal: Non-blocking operand view (event propagated) ---
    pub(crate) fn as_operand_view(&self) -> TensorView<'_, T>;

    // --- Consume self → Tensor (buffer reuse, guaranteed) ---
    fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    fn into_conj(self) -> Tensor<T>;

    // --- Borrow → new Tensor (new allocation, waits if pending) ---
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**TensorView methods:**

```rust
impl<'a, T: ScalarBase> TensorView<'a, T> {
    // Further view ops (data already ready — event is None in public API)
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'a, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'a, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'a, T>>;

    // Materialize: copy data into a new owned Tensor
    fn to_tensor(&self) -> Tensor<T>;
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**einsum takes &Tensor references (not TensorView):**

```rust
// einsum takes Tensor references. Internally uses as_operand_view()
// to propagate pending events without blocking (pipeline-safe).
// Einstein notation handles permute/broadcast — no need to create
// TensorView before calling einsum.
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

// Consuming variant: moves Tensors, enabling buffer reuse for intermediates.
// Buffer reuse is guaranteed (compile-time) — no runtime refcount check.
pub fn einsum_owned<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;
```

#### Ownership safety examples

```rust
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, ColumnMajor);
let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, ColumnMajor);

// einsum takes &Tensor — notation handles permutation/broadcast
let c = einsum("ij,jk->ik", &[&a, &b])?;       // → new Tensor
let c_t = einsum("ji,jk->ik", &[&a, &b])?;     // transposed a via notation

// View operations for CPU data inspection (waits if pending)
let at = a.permute(&[1, 0])?;           // TensorView borrowing a
assert_eq!(at.dims(), &[4, 3]);         // data is ready to read

// Compile-time safety: can't consume while borrowed
let at = a.permute(&[1, 0])?;
let d = einsum_owned("...", vec![a]);    // ERROR: at borrows a
drop(at);
let d = einsum_owned("...", vec![a])?;   // OK: borrow released

// Consuming einsum: buffer reuse is deterministic
let d = einsum_owned("ij,jk->ik", vec![a, b])?;
// a, b are moved — compiler guarantees no other references exist
// Their buffers can be reused for intermediates — zero allocation
```

#### Comparison with Arc-based approach

| Aspect | TensorView (chosen) | Arc-based |
|--------|---------------------|-----------|
| Buffer uniqueness | Compile-time guarantee | Runtime `Arc::strong_count` check |
| `into_` buffer reuse | Always succeeds | May fail if views exist (fallback to alloc) |
| API types | `Tensor` + `TensorView` | `Tensor` only |
| Lifetime complexity | Yes (`'a` propagates) | None |
| Runtime overhead | Zero | Atomic refcount on clone/drop |
| Rust idiom | `String`/`&str`, `Vec`/`&[T]` | `Arc<T>` |
| Misuse detection | Compile error | Silent fallback to allocation |

#### Coherence with async execution (CompletionEvent propagation)

Public view APIs (`view()`, `permute()`, etc.) **wait** before returning,
so users always receive ready-to-read data. Accelerator pipeline chaining
is handled internally by `as_operand_view()`, which propagates pending
events without blocking.

**Two-tier API contract:**

| Tier | Methods | Event handling | User visibility |
|------|---------|---------------|-----------------|
| **Public (CPU-read)** | `view()`, `permute()`, `broadcast()`, `diagonal()`, `view_mut()`, `to_tensor()`, `contiguous()`, `conj()` | **Wait** if pending, return ready data | Yes |
| **Internal (pipeline)** | `pub(crate) as_operand_view()` | **Propagate** event as `Option<&'a CompletionEvent>` | No (crate-internal) |
| **Accelerator ops** | `einsum` (takes `&[&Tensor]`) | Internally calls `as_operand_view()`, **detects** events → sets up stream dependency | Yes (but event handling is transparent) |

**GPU pipeline preserved via internal path:**

```rust
let a = einsum("ij,jk->ik", &[&x_gpu, &y_gpu])?;
//  → einsum internally: as_operand_view(&x_gpu), as_operand_view(&y_gpu)
//  → GPU submit, a.event = Some(event_1), returns immediately

let b = einsum("ij,jk->ik", &[&a, &z_gpu])?;
//  → einsum internally: as_operand_view(&a) → event_1 propagated
//  → detects event_1 → sets up stream dependency → no CPU wait
//  → GPU submit, b.event = Some(event_2), returns immediately

// Public API access triggers synchronization
let bv = b.view();  // wait(event_2), then return TensorView (event=None)
```

Note: Einstein notation subsumes `permute`/`broadcast` for einsum operands
(`"ji,jk->ik"` transposes the first operand), so users rarely need to
create TensorView explicitly when chaining accelerator operations.

**Implementation note — `wait(&self)` requires interior mutability:**

`Tensor::wait(&self)` must clear the `event` field through a shared
reference. This requires interior mutability (e.g., `Cell<Option<CompletionEvent>>`
or similar). This is an implementation detail that does not affect the
public API contract.

**Potential issues to monitor:**

1. **Non-einsum accelerator operations**: Future element-wise operations
   (add, mul, etc.) on accelerators will also need the `as_operand_view()`
   internal path to maintain pipelines. This is architecturally consistent
   but must be applied to every new accelerator-capable operation.

2. **TensorView cannot be passed to einsum**: Since `einsum` takes
   `&[&Tensor]`, a user cannot pass a `TensorView` directly. This is
   intentional — Einstein notation already handles permute/broadcast,
   and `einsum` uses `as_operand_view()` internally for pipeline safety.
   If a use case arises that genuinely requires passing views to einsum,
   a trait-based approach can be introduced later.

The Arc approach remains viable if lifetime ergonomics prove too burdensome
in practice. Switching from TensorView to Arc is backward-compatible for
callers (TensorView disappears, all methods return Tensor).

**Current status**: POC uses `Tensor<T>` only (no TensorView yet).
View operations (`permute`, `broadcast`, `diagonal`) return `Tensor<T>`
with `todo!()` bodies. The TensorView split will be implemented when
view operations are filled in.

### einsum variants (implemented in POC)

The POC includes three variant families alongside the allocating versions:

- **Accumulating** (`einsum_into`, `einsum_with_subscripts_into`, `einsum_with_plan_into`) --
  write into a caller-provided output buffer with BLAS-style accumulation
  (`output = alpha * einsum(...) + beta * output`). Avoids output allocation in hot loops.

- **Consuming** (`einsum_owned`, `einsum_with_subscripts_owned`, `einsum_with_plan_owned`) --
  take ownership of input tensors (`Vec<Tensor<T>>`). The implementation may reuse
  input buffers for intermediate results, avoiding allocation in contraction trees.
  Buffer reuse is guaranteed safe by Rust's ownership system (no runtime refcount check).

### Insights from ITensor Julia ecosystem

| Aspect | ITensor Julia | tenferro | Notes |
|---|---|---|---|
| Sparse storage | DOK-of-Arrays | Single DataBuffer + offset map | tenferro is GPU-friendly |
| Axis fusion | FusionStyle dispatch | Not yet designed | Critical for quantum number tensors |

### Relationship with mdarray / mdarray-linalg

| | mdarray / mdarray-linalg | tenferro-* |
|---|---|---|
| Role | **numpy equivalent** -- general-purpose multidimensional array | **PyTorch equivalent** -- high-performance tensor library |
| Memory | Owned `Array<T, D>` | `DataBuffer<T>` (CPU/GPU) |
| GPU | No | cuTENSOR, hipTensor (no Metal) |
| Autodiff | No | tenferro-autograd (VJP/JVP) [future] |
| Dispatch | Direct function calls | `TensorPrims` trait (backend selection) |

Both are needed. mdarray is a foundational array library; tenferro builds a
richer tensor ecosystem with GPU support and automatic differentiation.

tenferro-linalg and mdarray-linalg are **parallel** (both call faer directly),
not serial:

```
faer (SVD, QR, eigen)
    ^                ^
tenferro-linalg  mdarray-linalg-faer
(Tensor<T>       (Array<T, D>
 -> MatRef)       -> MatRef)
```
