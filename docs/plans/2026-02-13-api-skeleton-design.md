# API Skeleton Design for tenferro-rs

Date: 2026-02-13

## Goal

Create function signatures and docstrings (empty bodies) for all public APIs so that `cargo doc` generates reviewable documentation. No implementation logic; bodies use `todo!()`.

## Scope

- **Included**: tenferro-device, tenferro-tensorops, tenferro-tensor, tenferro-einsum
- **Excluded**: Automatic differentiation (VJP/JVP), C FFI, GPU-specific implementations

## Dependencies

- `strided-rs` (git dependency): `strided-traits`, `strided-view`, `strided-kernel`

## Workspace Structure

```
tenferro-rs/
├── Cargo.toml                 # workspace root
├── tenferro-device/
│   ├── Cargo.toml
│   └── src/lib.rs
├── tenferro-tensorops/
│   ├── Cargo.toml
│   └── src/lib.rs
├── tenferro-tensor/
│   ├── Cargo.toml
│   └── src/lib.rs
└── tenferro-einsum/
    ├── Cargo.toml
    └── src/lib.rs
```

## Crate Designs

### tenferro-device (Shared Layer)

Dependencies: `strided-view` (for error conversion only)

```rust
/// Device type enumeration.
pub enum Device {
    Cpu,
    Cuda { device_id: usize },
    Hip { device_id: usize },
}

/// Error type used across the tenferro workspace.
pub enum Error {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    RankMismatch { expected: usize, got: usize },
    DeviceError(String),
    InvalidArgument(String),
    Strided(strided_view::StridedError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

- `From<StridedError> for Error` implemented for seamless conversion.

### tenferro-tensorops (Layer 2)

Dependencies: `strided-view`, `strided-traits`, `tenferro-device`

This crate defines the cuTENSOR-compatible tensor operation protocol. `TensorOps` is an internal dispatch trait (not exposed as a type parameter in user-facing APIs).

```rust
/// Contraction descriptor specifying index modes (cuTENSOR-compatible).
pub struct ContractionDescriptor {
    pub modes_a: Vec<u32>,
    pub modes_b: Vec<u32>,
    pub modes_c: Vec<u32>,
}

/// Pre-computed execution plan for contractions.
pub struct ContractionPlan<T: ScalarBase> { .. }

/// Reduction operation kind.
pub enum ReduceOp {
    Sum,
    Max,
    Min,
}

/// Backend trait for tensor operations (cuTENSOR-compatible protocol).
///
/// Used internally for dispatch; not exposed as a type parameter
/// in user-facing APIs (PyTorch-style automatic backend selection).
pub trait TensorOps {
    fn plan_contraction<T: ScalarBase>(
        desc: &ContractionDescriptor,
        dims_a: &[usize],
        dims_b: &[usize],
        dims_c: &[usize],
    ) -> Result<ContractionPlan<T>>;

    fn contract<T: ScalarBase>(
        plan: &ContractionPlan<T>,
        alpha: T,
        a: &StridedView<T>,
        b: &StridedView<T>,
        beta: T,
        c: &mut StridedViewMut<T>,
    ) -> Result<()>;

    fn elementwise_binary<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
    ) -> Result<()>;

    fn reduce<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
        op: ReduceOp,
    ) -> Result<()>;

    fn permute<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        b: &mut StridedViewMut<T>,
        modes_b: &[u32],
    ) -> Result<()>;
}

/// CPU backend implementation (strided-kernel + GEMM).
pub struct CpuBackend;
impl TensorOps for CpuBackend { .. }
```

### tenferro-tensor (Layer 3)

Dependencies: `strided-view`, `strided-traits`, `tenferro-device`

```rust
/// Memory ordering for new allocations.
///
/// Used only as a parameter when allocating new memory.
/// Tensor itself stores only strides (no order field).
pub enum MemoryOrder {
    ColumnMajor,
    RowMajor,
}

/// Owned data buffer, device-aware.
pub enum DataBuffer<T> {
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional tensor type.
///
/// Composed of shape + strides + DataBuffer. Provides zero-copy view
/// operations (permute, broadcast, diagonal) and data-copying operations
/// (contiguous, to_device).
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    device: Device,
}

impl<T: ScalarBase> Tensor<T> {
    // --- Constructors (order specified as argument) ---
    pub fn zeros(dims: &[usize], device: Device, order: MemoryOrder) -> Self;
    pub fn ones(dims: &[usize], device: Device, order: MemoryOrder) -> Self;
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self>;
    pub fn from_strided_array(array: StridedArray<T>) -> Self;

    // --- Metadata ---
    pub fn dims(&self) -> &[usize];
    pub fn strides(&self) -> &[isize];
    pub fn ndim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn device(&self) -> &Device;

    // --- View operations (zero-copy) ---
    pub fn view(&self) -> StridedView<'_, T>;
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T>;
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>>;
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>>;

    // --- Data operations ---
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    pub fn is_contiguous(&self) -> bool;
}
```

### tenferro-einsum (Layer 4)

Dependencies: `tenferro-tensor`, `tenferro-tensorops`, `strided-traits`

Backend is automatically selected from Tensor's device (PyTorch-style).

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
///
/// Each dimension is represented by an integer label. Shared labels
/// across inputs are contracted.
pub struct Subscripts {
    pub inputs: Vec<Vec<u32>>,
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer labels.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self;

    /// Parse from string notation (NumPy/PyTorch compatible).
    ///
    /// Example: "ij,jk->ik"
    pub fn parse(notation: &str) -> Result<Self>;
}

/// Contraction tree determining pairwise contraction order for N-ary einsum.
pub struct ContractionTree { .. }

impl ContractionTree {
    /// Automatically compute optimal contraction order.
    pub fn optimize(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
    ) -> Result<Self>;

    /// Manually build contraction tree from pairwise contraction sequence.
    ///
    /// Example: For A[ij] B[jk] C[kl] -> D[il],
    /// contract B,C first, then A with the result:
    /// `pairs = &[(1, 2), (0, 3)]`
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}

/// Execute einsum with string notation.
/// Backend is auto-selected from tensor device.
pub fn einsum<T: ScalarBase>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with pre-built Subscripts.
pub fn einsum_with_subscripts<T: ScalarBase>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

/// Execute einsum with pre-optimized contraction tree (for repeated computation).
pub fn einsum_with_plan<T: ScalarBase>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

## Key Design Decisions

1. **strided-rs as real dependency**: Provides actual types (StridedView, ScalarBase, etc.) in public APIs for accurate `cargo doc` links.
2. **PyTorch-style backend dispatch**: No backend type parameter in user APIs. TensorOps is internal; backend is selected from Tensor's device field.
3. **Memory layout via strides only**: Tensor has no MemoryOrder field. MemoryOrder is only a parameter for memory-allocating functions.
4. **Integer label subscripts**: Primary notation is omeinsum-rs-compatible integer arrays. String parsing ("ij,jk->ik") also provided.
5. **Contraction order**: Both automatic optimization and manual specification (from_pairs) supported.
6. **Crate naming**: `tenferro-*` (not `t4a-*`).
