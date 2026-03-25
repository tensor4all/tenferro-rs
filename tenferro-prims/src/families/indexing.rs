use tenferro_algebra::{Algebra, Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};
use crate::{RocmBackend, RocmContext};

/// Reduction mode for scatter operations.
///
/// # Examples
///
/// ```
/// use tenferro_prims::ScatterReduction;
///
/// let r = ScatterReduction::None;
/// assert_eq!(r, ScatterReduction::None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScatterReduction {
    /// Overwrite the destination value.
    None,
    /// Add the source value into the destination.
    Add,
}

/// Descriptor for indexing-family planning.
///
/// # Examples
///
/// ```
/// use tenferro_prims::{IndexingPrimsDescriptor, ScatterReduction};
///
/// let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
/// assert!(matches!(desc, IndexingPrimsDescriptor::IndexSelect { .. }));
///
/// let desc = IndexingPrimsDescriptor::Scatter {
///     axis: 1,
///     reduction: ScatterReduction::Add,
/// };
/// assert!(matches!(desc, IndexingPrimsDescriptor::Scatter { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexingPrimsDescriptor {
    /// Select slices along `axis` using a 1-D index tensor.
    ///
    /// Given input of shape `[d0, ..., d_{axis}, ..., d_{n-1}]` and a 1-D
    /// index tensor of length `k`, produces output of shape
    /// `[d0, ..., k, ..., d_{n-1}]`.
    IndexSelect {
        /// Axis along which to select slices.
        axis: usize,
    },
    /// Gather elements along `axis` using an index tensor of the same rank.
    ///
    /// The index tensor shape must match the output shape.
    Gather {
        /// Axis along which to gather.
        axis: usize,
    },
    /// Scatter source values into output positions along `axis`.
    ///
    /// Inverse of gather: places values from `source` into `output` at
    /// positions determined by the index tensor.
    Scatter {
        /// Axis along which to scatter.
        axis: usize,
        /// Reduction mode: overwrite or accumulate.
        reduction: ScatterReduction,
    },
    /// Put source values at index positions (flat or per-axis).
    ///
    /// When `accumulate` is true, adds into the output instead of overwriting.
    IndexPut {
        /// Whether to accumulate (add) instead of overwriting.
        accumulate: bool,
    },
}

/// Indexing execution protocol family.
///
/// Provides index-based selection, gathering, and scattering of tensor
/// elements. Unlike the scalar and analytic families, indexing does not use
/// `alpha`/`beta` scaling — operations are pure data movement.
///
/// The `indices` tensor uses `i64` elements representing positions along the
/// target axis. Negative indices are not supported and will produce errors.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{CpuBackend, CpuContext, IndexingPrimsDescriptor, TensorIndexingPrims};
///
/// let mut ctx = CpuContext::new(1);
/// let desc = IndexingPrimsDescriptor::IndexSelect { axis: 0 };
/// let _plan = <CpuBackend as TensorIndexingPrims<Standard<f64>>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[3, 2], &[2], &[2, 2]],
/// )
/// .unwrap();
/// ```
pub trait TensorIndexingPrims<Alg: Algebra>: Sized {
    /// Backend-specific execution plan.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan an indexing operation for the given input/index/output shapes.
    ///
    /// `shapes` contains: `[input_shape, index_shape, output_shape]`.
    fn plan(
        ctx: &mut Self::Context,
        desc: &IndexingPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned indexing operation.
    ///
    /// - For `IndexSelect` and `Gather`: reads from `inputs[0]` using
    ///   `indices`, writes into `output`.
    /// - For `Scatter`: reads from `inputs[0]` (source values) using
    ///   `indices`, writes into `output`.
    /// - For `IndexPut`: reads from `inputs[0]` (values to put) using
    ///   `indices`, writes into `output`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        inputs: &[&Tensor<Alg::Scalar>],
        indices: &Tensor<i64>,
        output: &mut Tensor<Alg::Scalar>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_indexing_support(desc: IndexingPrimsDescriptor) -> bool;
}

// ============================================================================
// CUDA stub (when `cuda` feature is NOT enabled)
// ============================================================================

#[cfg(not(feature = "cuda"))]
impl<S: Scalar> TensorIndexingPrims<Standard<S>> for CudaBackend {
    type Plan = ();
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &IndexingPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "indexing family descriptor {desc:?} is not implemented on CudaBackend"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _inputs: &[&Tensor<S>],
        _indices: &Tensor<i64>,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "indexing family execution is not implemented on CudaBackend".into(),
        ))
    }

    fn has_indexing_support(_desc: IndexingPrimsDescriptor) -> bool {
        false
    }
}

// ============================================================================
// ROCm stub
// ============================================================================

impl<S: Scalar> TensorIndexingPrims<Standard<S>> for RocmBackend {
    type Plan = ();
    type Context = RocmContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &IndexingPrimsDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        Err(Error::DeviceError(format!(
            "indexing family descriptor {desc:?} is not implemented on RocmBackend"
        )))
    }

    fn execute(
        _ctx: &mut Self::Context,
        _plan: &Self::Plan,
        _inputs: &[&Tensor<S>],
        _indices: &Tensor<i64>,
        _output: &mut Tensor<S>,
    ) -> Result<()> {
        Err(Error::DeviceError(
            "indexing family execution is not implemented on RocmBackend".into(),
        ))
    }

    fn has_indexing_support(_desc: IndexingPrimsDescriptor) -> bool {
        false
    }
}
