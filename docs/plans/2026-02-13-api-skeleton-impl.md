# API Skeleton Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create all 4 tenferro crates with function signatures, docstrings, and `todo!()` bodies so `cargo doc` generates reviewable documentation.

**Architecture:** Bottom-up creation: tenferro-device -> tenferro-tensorops -> tenferro-tensor -> tenferro-einsum. Each crate depends on the ones below it. strided-rs is a git dependency from `https://github.com/tensor4all/strided-rs`.

**Tech Stack:** Rust workspace, strided-rs (strided-traits, strided-view), thiserror for error derive

**Design doc:** `docs/plans/2026-02-13-api-skeleton-design.md`

---

### Task 1: Create workspace root Cargo.toml

**Files:**
- Create: `Cargo.toml`

**Step 1: Create workspace Cargo.toml**

```toml
[workspace]
members = [
    "tenferro-device",
    "tenferro-tensorops",
    "tenferro-tensor",
    "tenferro-einsum",
]
resolver = "2"
```

**Step 2: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add workspace root Cargo.toml"
```

---

### Task 2: Create tenferro-device crate

**Files:**
- Create: `tenferro-device/Cargo.toml`
- Create: `tenferro-device/src/lib.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "tenferro-device"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Device abstraction and shared error types for the tenferro workspace."
publish = false

[dependencies]
strided-view = { git = "https://github.com/tensor4all/strided-rs" }
thiserror = "1.0"
```

**Step 2: Create src/lib.rs**

```rust
//! Device abstraction and shared error types for the tenferro workspace.
//!
//! This crate provides:
//! - [`Device`] enum representing compute devices (CPU, CUDA, HIP)
//! - [`Error`] and [`Result`] types used across all tenferro crates
//!
//! # Examples
//!
//! ```
//! use tenferro_device::Device;
//!
//! let dev = Device::Cpu;
//! ```

use std::fmt;

/// Compute device on which tensor data resides.
///
/// Currently only [`Device::Cpu`] is functional.
/// [`Device::Cuda`] and [`Device::Hip`] are defined for future GPU support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU device.
    Cpu,
    /// NVIDIA CUDA device with a specific device ID.
    Cuda {
        /// Zero-based CUDA device index.
        device_id: usize,
    },
    /// AMD HIP device with a specific device ID.
    Hip {
        /// Zero-based HIP device index.
        device_id: usize,
    },
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda { device_id } => write!(f, "cuda:{device_id}"),
            Device::Hip { device_id } => write!(f, "hip:{device_id}"),
        }
    }
}

/// Error type used across the tenferro workspace.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tensor shapes are incompatible for the requested operation.
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Tensor ranks (number of dimensions) do not match.
    #[error("rank mismatch: expected {expected}, got {got}")]
    RankMismatch {
        /// Expected rank.
        expected: usize,
        /// Actual rank.
        got: usize,
    },

    /// An error occurred on the compute device.
    #[error("device error: {0}")]
    DeviceError(String),

    /// An invalid argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// An error propagated from strided-view operations.
    #[error(transparent)]
    Strided(#[from] strided_view::StridedError),
}

/// Result type alias using [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
```

**Step 3: Verify it compiles**

Run: `cargo build -p tenferro-device`
Expected: successful compilation

**Step 4: Commit**

```bash
git add tenferro-device/
git commit -m "feat: add tenferro-device crate with Device enum and Error types"
```

---

### Task 3: Create tenferro-tensorops crate

**Files:**
- Create: `tenferro-tensorops/Cargo.toml`
- Create: `tenferro-tensorops/src/lib.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "tenferro-tensorops"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "cuTENSOR-compatible tensor operation protocol (TensorOps trait) for the tenferro workspace."
publish = false

[dependencies]
tenferro-device = { path = "../tenferro-device" }
strided-view = { git = "https://github.com/tensor4all/strided-rs" }
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
```

**Step 2: Create src/lib.rs**

```rust
//! cuTENSOR-compatible tensor operation protocol for the tenferro workspace.
//!
//! This crate defines the [`TensorOps`] trait, a backend-agnostic interface
//! for tensor contractions, element-wise operations, reductions, and permutations.
//! The API follows the cuTENSOR plan-based execution pattern:
//!
//! 1. Create a [`ContractionDescriptor`] specifying index modes
//! 2. Build a [`ContractionPlan`] (pre-allocates workspace, selects kernels)
//! 3. Execute the contraction with zero additional allocation
//!
//! # Backend dispatch
//!
//! [`TensorOps`] is used **internally** for dispatch. User-facing APIs
//! (e.g., `tenferro-einsum`) select the backend automatically based on
//! the tensor's [`Device`](tenferro_device::Device) (PyTorch-style).
//!
//! # CPU backend
//!
//! [`CpuBackend`] implements [`TensorOps`] using strided-kernel and GEMM.

use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_device::Result;

/// Describes a tensor contraction in terms of index modes (cuTENSOR-compatible).
///
/// Each mode is an integer label. Modes shared between two tensors are contracted
/// (summed over). This follows the cuTENSOR `cutensorOperationDescriptor` pattern.
///
/// # Examples
///
/// ```
/// use tenferro_tensorops::ContractionDescriptor;
///
/// // Matrix multiplication: C_{m,n} = A_{m,k} * B_{k,n}
/// let desc = ContractionDescriptor {
///     modes_a: vec![0, 1],  // m, k
///     modes_b: vec![1, 2],  // k, n
///     modes_c: vec![0, 2],  // m, n
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ContractionDescriptor {
    /// Index mode labels for tensor A.
    pub modes_a: Vec<u32>,
    /// Index mode labels for tensor B.
    pub modes_b: Vec<u32>,
    /// Index mode labels for the output tensor C.
    pub modes_c: Vec<u32>,
}

/// Pre-computed execution plan for a tensor contraction.
///
/// Created by [`TensorOps::plan_contraction`]. Encapsulates kernel selection
/// and workspace allocation so that [`TensorOps::contract`] can execute
/// with zero additional allocation.
pub struct ContractionPlan<T: ScalarBase> {
    _marker: PhantomData<T>,
}

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

/// Backend trait for tensor operations (cuTENSOR-compatible protocol).
///
/// Provides plan-based contraction, element-wise binary operations,
/// reductions, and permutations. All operations follow the BLAS/cuTENSOR
/// `C = α * op(inputs) + β * C` pattern.
///
/// This trait is used **internally** for backend dispatch. User-facing APIs
/// do not expose the backend as a type parameter.
pub trait TensorOps {
    /// Create an execution plan for a tensor contraction.
    ///
    /// The plan pre-computes kernel selection and workspace sizes based on
    /// the contraction descriptor and tensor dimensions.
    fn plan_contraction<T: ScalarBase>(
        desc: &ContractionDescriptor,
        dims_a: &[usize],
        dims_b: &[usize],
        dims_c: &[usize],
    ) -> Result<ContractionPlan<T>>;

    /// Execute a tensor contraction: `C = α * contract(A, B) + β * C`.
    ///
    /// Uses a pre-computed [`ContractionPlan`] for zero-allocation execution.
    fn contract<T: ScalarBase>(
        plan: &ContractionPlan<T>,
        alpha: T,
        a: &StridedView<T>,
        b: &StridedView<T>,
        beta: T,
        c: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Element-wise binary operation with mode alignment:
    /// `C_{modes_c} = α * A_{modes_a} + β * C_{modes_c}`.
    fn elementwise_binary<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
    ) -> Result<()>;

    /// Reduction over modes not present in the output:
    /// `C_{modes_c} = α * reduce_op(A_{modes_a}) + β * C_{modes_c}`.
    fn reduce<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        beta: T,
        c: &mut StridedViewMut<T>,
        modes_c: &[u32],
        op: ReduceOp,
    ) -> Result<()>;

    /// Permute tensor modes: `B_{modes_b} = α * A_{modes_a}`.
    fn permute<T: ScalarBase>(
        alpha: T,
        a: &StridedView<T>,
        modes_a: &[u32],
        b: &mut StridedViewMut<T>,
        modes_b: &[u32],
    ) -> Result<()>;
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on [`Device::Cpu`](tenferro_device::Device::Cpu).
pub struct CpuBackend;

impl TensorOps for CpuBackend {
    fn plan_contraction<T: ScalarBase>(
        _desc: &ContractionDescriptor,
        _dims_a: &[usize],
        _dims_b: &[usize],
        _dims_c: &[usize],
    ) -> Result<ContractionPlan<T>> {
        todo!()
    }

    fn contract<T: ScalarBase>(
        _plan: &ContractionPlan<T>,
        _alpha: T,
        _a: &StridedView<T>,
        _b: &StridedView<T>,
        _beta: T,
        _c: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn elementwise_binary<T: ScalarBase>(
        _alpha: T,
        _a: &StridedView<T>,
        _modes_a: &[u32],
        _beta: T,
        _c: &mut StridedViewMut<T>,
        _modes_c: &[u32],
    ) -> Result<()> {
        todo!()
    }

    fn reduce<T: ScalarBase>(
        _alpha: T,
        _a: &StridedView<T>,
        _modes_a: &[u32],
        _beta: T,
        _c: &mut StridedViewMut<T>,
        _modes_c: &[u32],
        _op: ReduceOp,
    ) -> Result<()> {
        todo!()
    }

    fn permute<T: ScalarBase>(
        _alpha: T,
        _a: &StridedView<T>,
        _modes_a: &[u32],
        _b: &mut StridedViewMut<T>,
        _modes_b: &[u32],
    ) -> Result<()> {
        todo!()
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p tenferro-tensorops`
Expected: successful compilation

**Step 4: Commit**

```bash
git add tenferro-tensorops/
git commit -m "feat: add tenferro-tensorops crate with TensorOps trait and CpuBackend"
```

---

### Task 4: Create tenferro-tensor crate

**Files:**
- Create: `tenferro-tensor/Cargo.toml`
- Create: `tenferro-tensor/src/lib.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "tenferro-tensor"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Dense tensor type with CPU/GPU support for the tenferro workspace."
publish = false

[dependencies]
tenferro-device = { path = "../tenferro-device" }
strided-view = { git = "https://github.com/tensor4all/strided-rs" }
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
num-traits = "0.2"
```

**Step 2: Create src/lib.rs**

```rust
//! Dense tensor type with CPU/GPU support.
//!
//! This crate provides [`Tensor<T>`], a multi-dimensional array type composed of
//! shape, strides, and a device-aware [`DataBuffer`]. It supports:
//!
//! - **Zero-copy view operations**: [`Tensor::permute`], [`Tensor::broadcast`],
//!   [`Tensor::diagonal`] modify only metadata (dims/strides)
//! - **Data operations**: [`Tensor::contiguous`] copies data into a contiguous layout
//! - **strided-rs interop**: [`Tensor::view`] / [`Tensor::view_mut`] produce
//!   [`StridedView`](strided_view::StridedView) /
//!   [`StridedViewMut`](strided_view::StridedViewMut) for use with
//!   [`TensorOps`](tenferro_tensorops::TensorOps) backends
//!
//! # Memory layout
//!
//! [`Tensor`] stores explicit strides and is not tied to any particular memory
//! order. [`MemoryOrder`] is only used as a parameter when allocating new memory
//! (e.g., [`Tensor::zeros`], [`Tensor::contiguous`]).
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::Device;
//!
//! let t = Tensor::<f64>::zeros(&[3, 4], Device::Cpu, MemoryOrder::ColumnMajor);
//! assert_eq!(t.dims(), &[3, 4]);
//! assert_eq!(t.ndim(), 2);
//! ```

use strided_traits::ScalarBase;
use strided_view::{StridedArray, StridedView, StridedViewMut};
use tenferro_device::{Device, Result};

/// Memory ordering for new allocations.
///
/// Specifies how elements are laid out in memory when creating new tensors
/// or copying data into a contiguous buffer. This is **not** stored on the
/// tensor itself — the tensor's [`strides`](Tensor::strides) fully describe
/// the memory layout.
///
/// - [`ColumnMajor`](MemoryOrder::ColumnMajor): First dimension is contiguous
///   (Fortran/Julia convention)
/// - [`RowMajor`](MemoryOrder::RowMajor): Last dimension is contiguous
///   (C/NumPy convention)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// Column-major (Fortran/Julia order). First dimension has stride 1.
    ColumnMajor,
    /// Row-major (C/NumPy order). Last dimension has stride 1.
    RowMajor,
}

/// Owned data buffer, device-aware.
///
/// Wraps the underlying storage for a tensor's data. Currently only CPU
/// storage is supported via [`StridedArray`].
pub enum DataBuffer<T> {
    /// CPU-resident data backed by a [`StridedArray`].
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional dense tensor.
///
/// `Tensor<T>` is the primary data type in tenferro. It owns its data via
/// [`DataBuffer`] and carries shape, strides, and device information.
///
/// ## Zero-copy views
///
/// Operations like [`permute`](Tensor::permute), [`broadcast`](Tensor::broadcast),
/// and [`diagonal`](Tensor::diagonal) return new `Tensor` values that share the
/// same underlying data buffer, modifying only the dims/strides/offset metadata.
///
/// ## strided-rs interop
///
/// Use [`view`](Tensor::view) and [`view_mut`](Tensor::view_mut) to obtain
/// [`StridedView`] / [`StridedViewMut`] references for passing to
/// low-level operations.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    device: Device,
}

impl<T: ScalarBase> Tensor<T> {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor (e.g., `&[3, 4]` for a 3×4 matrix)
    /// * `device` — Device on which to allocate the tensor
    /// * `order` — Memory layout for the new allocation
    pub fn zeros(dims: &[usize], device: Device, order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `dims` — Shape of the tensor
    /// * `device` — Device on which to allocate the tensor
    /// * `order` — Memory layout for the new allocation
    pub fn ones(dims: &[usize], device: Device, order: MemoryOrder) -> Self {
        todo!()
    }

    /// Create a tensor from a data slice.
    ///
    /// The slice length must equal the product of `dims`.
    /// Data is copied into owned storage with the specified memory order.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len()` does not match the product of `dims`.
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self> {
        todo!()
    }

    /// Create a tensor from an existing [`StridedArray`].
    ///
    /// Takes ownership of the array. The tensor inherits the array's
    /// dims, strides, and offset. Device is set to [`Device::Cpu`].
    pub fn from_strided_array(array: StridedArray<T>) -> Self {
        todo!()
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    /// Returns the shape (size of each dimension).
    pub fn dims(&self) -> &[usize] {
        todo!()
    }

    /// Returns the strides (in units of `T`).
    pub fn strides(&self) -> &[isize] {
        todo!()
    }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        todo!()
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        todo!()
    }

    /// Returns `true` if the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        todo!()
    }

    /// Returns the device on which this tensor resides.
    pub fn device(&self) -> &Device {
        todo!()
    }

    // ========================================================================
    // View operations (zero-copy)
    // ========================================================================

    /// Returns an immutable strided view of the tensor data.
    pub fn view(&self) -> StridedView<'_, T> {
        todo!()
    }

    /// Returns a mutable strided view of the tensor data.
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T> {
        todo!()
    }

    /// Permute (reorder) the dimensions of the tensor.
    ///
    /// This is a zero-copy operation that only modifies dims and strides.
    ///
    /// # Arguments
    ///
    /// * `perm` — Permutation of dimension indices (e.g., `&[1, 0]` to transpose)
    ///
    /// # Errors
    ///
    /// Returns an error if `perm` is not a valid permutation of `0..ndim()`.
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Broadcast the tensor to a larger shape.
    ///
    /// Dimensions of size 1 are expanded to the target size (zero-copy via
    /// stride 0). This is a zero-copy metadata operation.
    ///
    /// # Errors
    ///
    /// Returns an error if `target_dims` is incompatible with the current shape.
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Extract a diagonal view by merging pairs of axes.
    ///
    /// For each `(axis_i, axis_j)` pair, the two dimensions are replaced
    /// by a single diagonal dimension. This is a zero-copy stride trick.
    ///
    /// # Errors
    ///
    /// Returns an error if any axis is out of range or the paired
    /// dimensions have different sizes.
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>> {
        todo!()
    }

    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    /// Requires contiguous data; returns an error if the tensor is not contiguous.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not contiguous or the new shape
    /// has a different total element count.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>> {
        todo!()
    }

    // ========================================================================
    // Data operations
    // ========================================================================

    /// Return a contiguous copy of this tensor in the given memory order.
    ///
    /// If the tensor is already contiguous in the requested order,
    /// this may avoid copying (implementation-defined).
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T> {
        todo!()
    }

    /// Returns `true` if the tensor data is contiguous in memory.
    ///
    /// A tensor is contiguous if its elements occupy a dense block of
    /// memory with no gaps, in either column-major or row-major order.
    pub fn is_contiguous(&self) -> bool {
        todo!()
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p tenferro-tensor`
Expected: successful compilation

**Step 4: Commit**

```bash
git add tenferro-tensor/
git commit -m "feat: add tenferro-tensor crate with Tensor<T>, DataBuffer, MemoryOrder"
```

---

### Task 5: Create tenferro-einsum crate

**Files:**
- Create: `tenferro-einsum/Cargo.toml`
- Create: `tenferro-einsum/src/lib.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "tenferro-einsum"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "High-level einsum with N-ary contraction tree optimization for the tenferro workspace."
publish = false

[dependencies]
tenferro-device = { path = "../tenferro-device" }
tenferro-tensorops = { path = "../tenferro-tensorops" }
tenferro-tensor = { path = "../tenferro-tensor" }
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
```

**Step 2: Create src/lib.rs**

```rust
//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides Einstein summation notation for [`Tensor`](tenferro_tensor::Tensor)
//! values. It supports:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Integer label notation**: omeinsum-rs compatible, using `u32` labels
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//!
//! # Backend dispatch
//!
//! The backend is selected automatically from the tensor's
//! [`Device`](tenferro_device::Device) (PyTorch-style). There is no backend
//! type parameter in the public API.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_einsum::{einsum, Subscripts};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::Device;
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], Device::Cpu, MemoryOrder::ColumnMajor);
//! let b = Tensor::<f64>::zeros(&[4, 5], Device::Cpu, MemoryOrder::ColumnMajor);
//!
//! // String notation
//! let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
//!
//! // Integer label notation
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let c = einsum_with_subscripts(&subs, &[&a, &b]).unwrap();
//! ```

use strided_traits::ScalarBase;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

/// Einsum subscripts using integer labels (omeinsum-rs compatible).
///
/// Each dimension is represented by a `u32` label. Labels shared across
/// multiple input tensors are contracted (summed over). Labels present
/// only in the output are free indices.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// // Matrix multiplication: C_{ik} = Σ_j A_{ij} * B_{jk}
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// assert_eq!(subs.inputs.len(), 2);
/// assert_eq!(subs.output, vec![0, 2]);
/// ```
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// // Parse from string notation
/// let subs = Subscripts::parse("ij,jk->ik").unwrap();
/// assert_eq!(subs.inputs.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct Subscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer label arrays.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Index labels for each input tensor
    /// * `output` — Index labels for the output tensor
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self {
        Self {
            inputs: inputs.iter().map(|s| s.to_vec()).collect(),
            output: output.to_vec(),
        }
    }

    /// Parse subscripts from NumPy/PyTorch-style string notation.
    ///
    /// Each character (`a`–`z`, `A`–`Z`) represents a dimension label.
    /// Input tensors are separated by commas, and `->` separates inputs
    /// from the output.
    ///
    /// # Examples
    ///
    /// - `"ij,jk->ik"` — matrix multiplication
    /// - `"ii->i"` — diagonal extraction
    /// - `"ijk->"` — full contraction (scalar result)
    ///
    /// # Errors
    ///
    /// Returns an error if the notation is malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        todo!()
    }
}

/// Contraction tree determining pairwise contraction order for N-ary einsum.
///
/// When contracting more than two tensors, the order in which pairwise
/// contractions are performed significantly affects performance.
/// `ContractionTree` encodes this order as a binary tree.
///
/// # Optimization
///
/// Use [`ContractionTree::optimize`] for automatic cost-based optimization
/// (e.g., greedy algorithm based on tensor sizes), or
/// [`ContractionTree::from_pairs`] for manual specification.
pub struct ContractionTree {
    // Internal representation is private.
    _private: (),
}

impl ContractionTree {
    /// Automatically compute an optimized contraction order.
    ///
    /// Uses a cost-based heuristic (e.g., greedy algorithm) to determine
    /// the pairwise contraction sequence that minimizes total operation count.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    ///
    /// # Errors
    ///
    /// Returns an error if subscripts and shapes are inconsistent.
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self> {
        todo!()
    }

    /// Manually build a contraction tree from a pairwise contraction sequence.
    ///
    /// Each pair `(i, j)` specifies which two tensors (or intermediate results)
    /// to contract next. Intermediate results are assigned indices starting
    /// from the number of input tensors.
    ///
    /// # Arguments
    ///
    /// * `subscripts` — Einsum subscripts for all tensors
    /// * `shapes` — Shape of each input tensor
    /// * `pairs` — Ordered list of pairwise contractions
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Three tensors: A[ij] B[jk] C[kl] -> D[il]
    /// // Contract B and C first, then A with the result:
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    /// let shapes = [&[3, 4][..], &[4, 5], &[5, 6]];
    /// let tree = ContractionTree::from_pairs(
    ///     &subs,
    ///     &shapes,
    ///     &[(1, 2), (0, 3)],  // B*C -> T(index=3), then A*T -> D
    /// ).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the pairs do not form a valid contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self> {
        todo!()
    }
}

/// Execute einsum using string notation.
///
/// Parses the subscript string, optimizes the contraction order, and
/// executes the contraction. The backend is selected automatically from
/// the tensors' device.
///
/// # Arguments
///
/// * `subscripts` — Einstein summation notation (e.g., `"ij,jk->ik"`)
/// * `operands` — Input tensors
///
/// # Examples
///
/// ```ignore
/// // Matrix multiplication
/// let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
///
/// // Trace
/// let tr = einsum("ii->", &[&a]).unwrap();
///
/// // Batch matrix multiplication
/// let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
/// ```
///
/// # Errors
///
/// Returns an error if the notation is invalid or tensor shapes are
/// incompatible with the subscripts.
pub fn einsum<T: ScalarBase>(subscripts: &str, operands: &[&Tensor<T>]) -> Result<Tensor<T>> {
    todo!()
}

/// Execute einsum with pre-built [`Subscripts`].
///
/// Avoids re-parsing the subscript string on each call. Useful when the
/// same contraction pattern is applied to tensors of varying shapes.
///
/// # Errors
///
/// Returns an error if tensor shapes are incompatible with the subscripts.
pub fn einsum_with_subscripts<T: ScalarBase>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>> {
    todo!()
}

/// Execute einsum with a pre-optimized [`ContractionTree`].
///
/// Avoids both subscript parsing and contraction order optimization.
/// Ideal for hot loops where the same contraction is executed repeatedly
/// on tensors of the same shape.
///
/// # Errors
///
/// Returns an error if the operand shapes do not match those used to
/// build the contraction tree.
pub fn einsum_with_plan<T: ScalarBase>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>> {
    todo!()
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p tenferro-einsum`
Expected: successful compilation

**Step 4: Commit**

```bash
git add tenferro-einsum/
git commit -m "feat: add tenferro-einsum crate with einsum, Subscripts, ContractionTree"
```

---

### Task 6: Verify full workspace build and cargo doc

**Step 1: Build entire workspace**

Run: `cargo build`
Expected: all 4 crates compile successfully

**Step 2: Check formatting**

Run: `cargo fmt --check`
Expected: no formatting issues (or run `cargo fmt` to fix)

**Step 3: Generate documentation**

Run: `cargo doc --no-deps --workspace`
Expected: documentation generated without errors

**Step 4: Verify doc output exists**

Run: `ls target/doc/tenferro_*/index.html`
Expected: one index.html per crate (tenferro_device, tenferro_tensorops, tenferro_tensor, tenferro_einsum)

**Step 5: Commit any formatting fixes**

```bash
cargo fmt
git add -A
git commit -m "style: apply cargo fmt"
```

---

### Task 7: Update AGENTS.md and README.md for new crate names

**Files:**
- Modify: `AGENTS.md` — update crate references from `t4a-*` to `tenferro-*`
- Modify: `README.md` — update crate names and descriptions

**Step 1: Update AGENTS.md**

Replace all `t4a-` prefixes with `tenferro-` in crate names and build commands.

**Step 2: Update README.md**

Replace all `t4a-` prefixes with `tenferro-` and update the project description.

**Step 3: Commit**

```bash
git add AGENTS.md README.md
git commit -m "docs: update crate names from t4a-* to tenferro-*"
```

---

### Task 8: Final verification and push

**Step 1: Run full checklist**

```bash
cargo fmt --check
cargo build --workspace
cargo doc --no-deps --workspace
```
Expected: all pass

**Step 2: Push to remote**

```bash
git push origin main
```
