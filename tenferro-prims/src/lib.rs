//! Tensor primitive operations for the tenferro workspace.
//!
//! This crate defines the [`TensorPrims<A>`] trait, a backend-agnostic interface
//! parameterized by algebra `A`. The API follows the cuTENSOR plan-based execution
//! pattern:
//!
//! 1. Create a [`PrimDescriptor`] specifying the operation and index modes
//! 2. Build a plan via [`TensorPrims::plan`] (pre-computes kernel selection)
//! 3. Execute the plan via [`TensorPrims::execute`]
//!
//! # Operation categories
//!
//! **Core operations** (every backend must implement):
//! - [`BatchedGemm`](PrimDescriptor::BatchedGemm): Batched matrix multiplication
//! - [`Reduce`](PrimDescriptor::Reduce): Sum/max/min reduction over modes
//! - [`Trace`](PrimDescriptor::Trace): Trace (contraction of paired diagonal modes)
//! - [`Permute`](PrimDescriptor::Permute): Mode reordering
//! - [`AntiTrace`](PrimDescriptor::AntiTrace): Scatter-add to diagonal (AD backward of trace)
//! - [`AntiDiag`](PrimDescriptor::AntiDiag): Write to diagonal positions (AD backward of diag)
//!
//! **Extended operations** (dynamically queried via [`TensorPrims::has_extension_for`]):
//! - [`Contract`](PrimDescriptor::Contract): Fused permute + GEMM contraction
//! - [`ElementwiseMul`](PrimDescriptor::ElementwiseMul): Element-wise multiplication
//!
//! # Algebra parameterization
//!
//! [`TensorPrims<A>`] is parameterized by algebra `A` (e.g.,
//! [`Standard`](tenferro_algebra::Standard), `MaxPlus`).
//! External crates implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule
//! compatible). The [`HasAlgebra`](tenferro_algebra::HasAlgebra) trait on scalar types
//! enables automatic inference: `Tensor<f64>` → `Standard`.
//!
//! # Examples
//!
//! ## Plan-based GEMM
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor};
//! use tenferro_algebra::Standard;
//! use strided_view::StridedArray;
//!
//! let a = StridedArray::<f64>::col_major(&[3, 4]);
//! let b = StridedArray::<f64>::col_major(&[4, 5]);
//! let mut c = StridedArray::<f64>::col_major(&[3, 5]);
//!
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![], m: 3, n: 5, k: 4,
//! };
//! let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//! CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
//! ```
//!
//! ## Reduction (sum over an axis)
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp};
//!
//! // Sum over columns: c_i = Σ_j A_{i,j}
//! let desc = PrimDescriptor::Reduce {
//!     modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum,
//! };
//! let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
//! CpuBackend::execute(&plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();
//! ```
//!
//! ## Dynamic extension check
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, Extension};
//!
//! if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
//!     let desc = PrimDescriptor::Contract {
//!         modes_a: vec![0, 1], modes_b: vec![1, 2], modes_c: vec![0, 2],
//!     };
//!     let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//!     CpuBackend::execute(
//!         &plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut(),
//!     ).unwrap();
//! }
//! ```

use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Standard;
use tenferro_device::Result;

/// Reduction operation kind.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ReduceOp;
///
/// let op = ReduceOp::Sum;
/// assert_eq!(op, ReduceOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

/// Extended operation identifiers for dynamic capability query.
///
/// Used with [`TensorPrims::has_extension_for`] to check at runtime whether
/// a backend supports an optimized extended operation for a given scalar type.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, TensorPrims, Extension};
///
/// // Check if fused contraction is available for f64
/// let available = CpuBackend::has_extension_for::<f64>(Extension::Contract);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Fused contraction (permute + GEMM). Maps to `cutensorContract` on GPU.
    Contract,
    /// Element-wise multiplication. Maps to `cutensorElementwiseBinary` on GPU.
    ElementwiseMul,
}

/// Describes a tensor primitive operation.
///
/// All operations follow the cuTENSOR pattern: describe → plan → execute.
/// Core operations must be supported by every backend. Extended operations
/// are dynamically queried via [`TensorPrims::has_extension_for`].
///
/// Modes are `u32` integer labels matching cuTENSOR conventions. Modes
/// shared between input and output tensors are batch/free dimensions;
/// modes present only in inputs are contracted.
///
/// # Examples
///
/// ```
/// use tenferro_prims::PrimDescriptor;
///
/// // Matrix multiplication: C_{m,n} = A_{m,k} * B_{k,n}
/// let desc = PrimDescriptor::Contract {
///     modes_a: vec![0, 1],  // m=0, k=1
///     modes_b: vec![1, 2],  // k=1, n=2
///     modes_c: vec![0, 2],  // m=0, n=2
/// };
/// ```
pub enum PrimDescriptor {
    // ====================================================================
    // Core operations (every backend must implement)
    // ====================================================================
    /// Batched matrix multiplication.
    ///
    /// `C[batch, m, n] = alpha * A[batch, m, k] * B[batch, k, n] + beta * C`
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Number of rows in A / C.
        m: usize,
        /// Number of columns in B / C.
        n: usize,
        /// Contraction dimension (columns of A / rows of B).
        k: usize,
    },

    /// Reduction over modes not present in the output.
    ///
    /// `C[modes_c] = alpha * reduce_op(A[modes_a]) + beta * C[modes_c]`
    Reduce {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C (subset of modes_a).
        modes_c: Vec<u32>,
        /// Reduction operation (Sum, Max, Min).
        op: ReduceOp,
    },

    /// Trace: contraction of paired diagonal modes.
    ///
    /// For each pair `(i, j)`, sums over the diagonal where mode i == mode j.
    Trace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes to trace over.
        paired: Vec<(u32, u32)>,
    },

    /// Permute (reorder) tensor modes.
    ///
    /// `B[modes_b] = alpha * A[modes_a]`
    Permute {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor B (same labels, different order).
        modes_b: Vec<u32>,
    },

    /// Anti-trace: scatter-add gradient to diagonal (AD backward of trace).
    AntiTrace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal scatter.
        paired: Vec<(u32, u32)>,
    },

    /// Anti-diag: write gradient to diagonal positions (AD backward of diag).
    AntiDiag {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal write.
        paired: Vec<(u32, u32)>,
    },

    // ====================================================================
    // Extended operations (dynamically queried)
    // ====================================================================
    /// Fused contraction: permute + GEMM in one operation.
    ///
    /// `C[modes_c] = alpha * contract(A[modes_a], B[modes_b]) + beta * C`
    ///
    /// Available when `has_extension_for::<T>(Extension::Contract)` returns true.
    Contract {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for input tensor B.
        modes_b: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
    },

    /// Element-wise multiplication of two tensors.
    ///
    /// Available when `has_extension_for::<T>(Extension::ElementwiseMul)` returns true.
    ElementwiseMul,
}

/// Backend trait for tensor primitive operations, parameterized by algebra `A`.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, anti_trace,
/// anti_diag) must be implemented. Extended ops (contract, elementwise_mul)
/// are dynamically queried via [`has_extension_for`](TensorPrims::has_extension_for).
///
/// # Algebra parameterization
///
/// The algebra parameter `A` enables extensibility: external crates can
/// implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).
///
/// # Associated functions (not methods)
///
/// All functions are associated functions (no `&self` receiver). Call as
/// `CpuBackend::plan::<f64>(...)` instead of `backend.plan(...)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor};
///
/// // Plan a batched GEMM
/// let desc = PrimDescriptor::BatchedGemm {
///     batch_dims: vec![], m: 3, n: 5, k: 4,
/// };
/// let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
///
/// // Execute: C = 1.0 * A*B + 0.0 * C
/// CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
/// ```
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Create an execution plan from an operation descriptor.
    ///
    /// The plan pre-computes kernel selection and workspace sizes.
    /// `shapes` contains the shape of each tensor involved in the operation
    /// (inputs first, then output).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp};
    ///
    /// let desc = PrimDescriptor::Reduce {
    ///     modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum,
    /// };
    /// let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
    /// ```
    fn plan<T: ScalarBase>(desc: &PrimDescriptor, shapes: &[&[usize]]) -> Result<Self::Plan<T>>;

    /// Execute a plan with the given scaling factors and tensor views.
    ///
    /// Follows the BLAS/cuTENSOR pattern:
    /// `output = alpha * op(inputs) + beta * output`
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Execute: output = 1.0 * gemm(a, b) + 0.0 * output  (overwrite)
    /// CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
    ///
    /// // Accumulate: output = 1.0 * gemm(a, b) + 1.0 * output  (add)
    /// CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 1.0, &mut c.view_mut()).unwrap();
    /// ```
    fn execute<T: ScalarBase>(
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for scalar type `T`.
    ///
    /// Returns `true` if the backend supports the given extended operation
    /// for the specified scalar type. This enables dynamic dispatch:
    /// GPU may support Contract for f64 but not for tropical types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_prims::{CpuBackend, TensorPrims, Extension};
    ///
    /// if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
    ///     // Use fused contraction for better performance
    /// } else {
    ///     // Decompose into core ops: permute → batched_gemm
    /// }
    /// ```
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}

/// CPU plan — concrete enum, no type erasure.
///
/// Created by [`CpuBackend::plan`](TensorPrims::plan) and consumed by
/// [`CpuBackend::execute`](TensorPrims::execute).
pub enum CpuPlan<T: ScalarBase> {
    /// Plan for batched GEMM.
    BatchedGemm {
        /// Number of rows.
        m: usize,
        /// Number of columns.
        n: usize,
        /// Contraction dimension.
        k: usize,
        _marker: PhantomData<T>,
    },
    /// Plan for reduction.
    Reduce {
        /// Axis to reduce over.
        axis: usize,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping.
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for fused contraction (extended op).
    Contract { _marker: PhantomData<T> },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul { _marker: PhantomData<T> },
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on
/// [`Device::Cpu`](tenferro_device::Device::Cpu).
/// Implements [`TensorPrims<Standard>`] for standard arithmetic.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor};
/// use strided_view::StridedArray;
///
/// // Transpose a matrix
/// let desc = PrimDescriptor::Permute {
///     modes_a: vec![0, 1],
///     modes_b: vec![1, 0],
/// };
/// let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 3]]).unwrap();
/// let a = StridedArray::<f64>::col_major(&[3, 4]);
/// let mut b = StridedArray::<f64>::col_major(&[4, 3]);
/// CpuBackend::execute(&plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();
/// ```
pub struct CpuBackend;

impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;

    fn plan<T: ScalarBase>(_desc: &PrimDescriptor, _shapes: &[&[usize]]) -> Result<CpuPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _plan: &CpuPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        todo!()
    }
}
