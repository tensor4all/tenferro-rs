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
