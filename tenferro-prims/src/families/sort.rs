use tenferro_algebra::{Algebra, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};
use crate::{RocmBackend, RocmContext};

/// Descriptor for sort-family planning.
///
/// # Examples
///
/// ```
/// use tenferro_prims::SortPrimsDescriptor;
///
/// let desc = SortPrimsDescriptor::Sort {
///     axis: 0,
///     descending: false,
///     stable: true,
/// };
/// assert!(matches!(desc, SortPrimsDescriptor::Sort { .. }));
///
/// let desc = SortPrimsDescriptor::Topk {
///     axis: 0,
///     k: 3,
///     largest: true,
///     sorted: true,
/// };
/// assert!(matches!(desc, SortPrimsDescriptor::Topk { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SortPrimsDescriptor {
    /// Sort elements along `axis`.
    ///
    /// Produces both sorted values and the corresponding permutation indices.
    Sort {
        /// Axis along which to sort.
        axis: usize,
        /// If true, sort in descending order.
        descending: bool,
        /// If true, use a stable sort (preserving relative order of equal elements).
        stable: bool,
    },
    /// Compute the permutation that would sort elements along `axis`.
    ///
    /// Returns only the index permutation (values output is left unmodified).
    Argsort {
        /// Axis along which to compute the sort permutation.
        axis: usize,
        /// If true, sort in descending order.
        descending: bool,
        /// If true, use a stable sort.
        stable: bool,
    },
    /// Select the top-k elements along `axis`.
    ///
    /// Returns the k largest (or smallest) values and their indices.
    Topk {
        /// Axis along which to select top-k.
        axis: usize,
        /// Number of elements to select.
        k: usize,
        /// If true, select the k largest; if false, select the k smallest.
        largest: bool,
        /// If true, the returned k elements are sorted.
        sorted: bool,
    },
}

/// Sort execution protocol family.
///
/// Provides sorting, argsort, and top-k operations on tensors. The `indices`
/// output tensor uses `i64` elements representing positions along the target
/// axis (same convention as [`crate::TensorIndexingPrims`]).
///
/// Unlike semiring and scalar families, sort does not use `alpha`/`beta`
/// scaling -- operations are pure data rearrangement.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext, SortPrimsDescriptor, TensorSortPrims};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = SortPrimsDescriptor::Sort {
///     axis: 0,
///     descending: false,
///     stable: true,
/// };
/// let _plan = <CpuBackend as TensorSortPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[5]],
/// )
/// .unwrap();
/// ```
pub trait TensorSortPrims<Alg: Algebra>: Sized
where
    Alg::Scalar: PartialOrd,
{
    /// Backend-specific execution plan.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a sort operation for the given input shape.
    ///
    /// `shapes` contains: `[input_shape]`.
    fn plan(
        ctx: &mut Self::Context,
        desc: &SortPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned sort operation.
    ///
    /// - For `Sort`: writes sorted values to `values_out` and permutation
    ///   indices to `indices_out`.
    /// - For `Argsort`: writes permutation indices to `indices_out`;
    ///   `values_out` is left unmodified.
    /// - For `Topk`: writes top-k values to `values_out` and their original
    ///   indices to `indices_out`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        input: &Tensor<Alg::Scalar>,
        values_out: &mut Tensor<Alg::Scalar>,
        indices_out: &mut Tensor<i64>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_sort_support(desc: &SortPrimsDescriptor) -> bool;
}

// ============================================================================
// CUDA stub (when `cuda` feature is NOT enabled)
// ============================================================================

#[cfg(not(feature = "cuda"))]
impl<S: Scalar + PartialOrd> TensorSortPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &SortPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "sort family descriptor {desc:?} is not implemented on CudaBackend"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _input: &Tensor<S>,
        _values_out: &mut Tensor<S>,
        _indices_out: &mut Tensor<i64>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "sort family execution is not implemented on CudaBackend".into(),
        ))
    }

    fn has_sort_support(_desc: &SortPrimsDescriptor) -> bool {
        false
    }
}

// ============================================================================
// ROCm stub
// ============================================================================

impl<S: Scalar + PartialOrd> TensorSortPrims<Standard<S>> for RocmBackend {
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &SortPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "sort family descriptor {desc:?} is not implemented on RocmBackend"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _input: &Tensor<S>,
        _values_out: &mut Tensor<S>,
        _indices_out: &mut Tensor<i64>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "sort family execution is not implemented on RocmBackend".into(),
        ))
    }

    fn has_sort_support(_desc: &SortPrimsDescriptor) -> bool {
        false
    }
}
