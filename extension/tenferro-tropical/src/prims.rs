//! [`TensorPrims`] implementations for tropical algebras on [`CpuBackend`].
//!
//! Each tropical algebra gets its own `impl TensorPrims<XxxAlgebra> for CpuBackend`.
//! The orphan rule is satisfied because `XxxAlgebra` is defined in this crate.
//!
//! Extended operations (Contract, ElementwiseMul) are not supported for
//! tropical algebras — `has_extension_for` always returns `false`.

use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_device::Result;
use tenferro_prims::{CpuBackend, Extension, PrimDescriptor, ReduceOp, TensorPrims};

use crate::algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};

/// Execution plan for tropical primitive operations on CPU.
///
/// Analogous to [`CpuPlan`](tenferro_prims::CpuPlan) but for tropical
/// algebras. The plan captures pre-computed kernel selection information.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp};
/// use tenferro_tropical::{MaxPlusAlgebra, TropicalPlan};
///
/// let desc = PrimDescriptor::Reduce {
///     modes_a: vec![0, 1],
///     modes_c: vec![0],
///     op: ReduceOp::Max,
/// };
/// let plan: TropicalPlan<f64> = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
/// ```
pub enum TropicalPlan<T: ScalarBase> {
    /// Plan for batched GEMM under tropical algebra.
    BatchedGemm {
        /// Number of rows.
        m: usize,
        /// Number of columns.
        n: usize,
        /// Contraction dimension.
        k: usize,
        _marker: PhantomData<T>,
    },
    /// Plan for reduction under tropical algebra.
    Reduce {
        /// Axis to reduce over.
        axis: usize,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace under tropical algebra.
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
}

// ---------------------------------------------------------------------------
// impl TensorPrims<MaxPlusAlgebra> for CpuBackend
// ---------------------------------------------------------------------------

impl TensorPrims<MaxPlusAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;

    fn plan<T: ScalarBase>(
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _plan: &TropicalPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// impl TensorPrims<MinPlusAlgebra> for CpuBackend
// ---------------------------------------------------------------------------

impl TensorPrims<MinPlusAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;

    fn plan<T: ScalarBase>(
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _plan: &TropicalPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// impl TensorPrims<MaxMulAlgebra> for CpuBackend
// ---------------------------------------------------------------------------

impl TensorPrims<MaxMulAlgebra> for CpuBackend {
    type Plan<T: ScalarBase> = TropicalPlan<T>;

    fn plan<T: ScalarBase>(
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<TropicalPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _plan: &TropicalPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    /// Tropical backends do not support extended operations.
    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        false
    }
}
