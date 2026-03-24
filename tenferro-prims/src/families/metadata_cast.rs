use num_traits::{NumCast, ToPrimitive};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::shape_helpers::validate_shape_broadcastable;
use crate::{validate_shape_count, MetadataDType, MetadataTensorRef};
#[cfg(feature = "cuda")]
use crate::{ScalarPrimsDescriptor, ScalarTernaryOp};

/// Metadata-to-scalar bridge planning operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_prims::{MetadataCastPrimsDescriptor, MetadataDType};
///
/// let cast = MetadataCastPrimsDescriptor::PointwiseCast {
///     input_dtype: MetadataDType::Bool,
/// };
/// assert!(matches!(cast, MetadataCastPrimsDescriptor::PointwiseCast { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetadataCastPrimsDescriptor {
    /// Cast a metadata tensor into a scalar tensor of the backend's scalar dtype.
    PointwiseCast {
        /// Logical dtype of the metadata input.
        input_dtype: MetadataDType,
    },
    /// Select between scalar tensors using a bool metadata mask.
    Where {
        /// Logical dtype of the condition/mask metadata input.
        cond_dtype: MetadataDType,
    },
}

/// Erased inputs for metadata-to-scalar bridge execution.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_prims::{MetadataScalarTensorRef, MetadataTensorRef};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let mask = Tensor::<u8>::zeros(
///     &[2],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let input = MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum MetadataScalarTensorRef<'a, S: Scalar> {
    /// Metadata tensor input.
    Metadata(MetadataTensorRef<'a>),
    /// Scalar tensor input.
    Scalar(&'a Tensor<S>),
}

/// Metadata-to-scalar bridge protocol.
///
/// This family bridges integer/bool metadata tensors into scalar tensors so
/// higher-level crates can reuse scalar `where` and similar dense eager paths.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_prims::{
///     CpuBackend, CpuContext, MetadataCastPrimsDescriptor, MetadataDType,
///     TensorMetadataCastPrims,
/// };
///
/// let mut ctx = CpuContext::new(1);
/// let desc = MetadataCastPrimsDescriptor::PointwiseCast {
///     input_dtype: MetadataDType::I32,
/// };
/// let _plan = <CpuBackend as TensorMetadataCastPrims<f32>>::plan(
///     &mut ctx,
///     &desc,
///     &[&[2], &[2]],
/// )
/// .unwrap();
/// ```
pub trait TensorMetadataCastPrims<S: Scalar> {
    /// Backend plan type.
    type Plan;
    /// Backend execution context.
    type Context;

    /// Plan a metadata-to-scalar bridge operation.
    fn plan(
        ctx: &mut Self::Context,
        desc: &MetadataCastPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan>;

    /// Execute a previously planned metadata-to-scalar bridge operation.
    ///
    /// The execution contract matches the rest of tenferro prims:
    /// `output <- alpha * op(inputs) + beta * output`.
    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[MetadataScalarTensorRef<'_, S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()>;

    /// Report whether the backend advertises support for the given descriptor.
    fn has_metadata_cast_support(desc: MetadataCastPrimsDescriptor) -> bool;
}

/// Return whether a metadata-to-scalar descriptor is supported by phase 1.
pub(crate) fn supports_metadata_cast(desc: &MetadataCastPrimsDescriptor) -> bool {
    match desc {
        MetadataCastPrimsDescriptor::PointwiseCast { input_dtype } => {
            matches!(input_dtype, MetadataDType::Bool | MetadataDType::I32)
        }
        MetadataCastPrimsDescriptor::Where { cond_dtype } => {
            matches!(cond_dtype, MetadataDType::Bool)
        }
    }
}

/// Validate the shapes used by a metadata-to-scalar bridge descriptor.
pub(crate) fn validate_metadata_cast_shapes(
    desc: &MetadataCastPrimsDescriptor,
    shapes: &[&[usize]],
    op_name: &str,
) -> Result<()> {
    match desc {
        MetadataCastPrimsDescriptor::PointwiseCast { .. } => {
            validate_shape_count(shapes, 2, op_name)?;
            validate_shape_broadcastable(shapes[0], shapes[1], op_name)?;
            Ok(())
        }
        MetadataCastPrimsDescriptor::Where { .. } => {
            validate_shape_count(shapes, 4, op_name)?;
            validate_shape_broadcastable(shapes[0], shapes[3], op_name)?;
            validate_shape_broadcastable(shapes[1], shapes[3], op_name)?;
            validate_shape_broadcastable(shapes[2], shapes[3], op_name)?;
            Ok(())
        }
    }
}

pub(crate) fn cast_metadata_value<S, T>(value: T, label: &str) -> Result<S>
where
    S: Scalar + NumCast,
    T: ToPrimitive + Copy,
{
    NumCast::from(value).ok_or_else(|| {
        Error::InvalidArgument(format!(
            "{label} cannot be represented as {}",
            std::any::type_name::<S>()
        ))
    })
}

pub(crate) fn for_each_index_result(
    dims: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    let mut result = Ok(());
    crate::for_each_index(dims, |idx| {
        if result.is_ok() {
            result = f(idx);
        }
    });
    result
}

pub(crate) fn validate_where_bridge_inputs<'a, S: Scalar>(
    inputs: &'a [MetadataScalarTensorRef<'a, S>],
) -> Result<(MetadataTensorRef<'a>, &'a Tensor<S>, &'a Tensor<S>)> {
    if inputs.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "MetadataCastWhere expects 3 input(s) (got {})",
            inputs.len()
        )));
    }
    let MetadataScalarTensorRef::Metadata(cond) = inputs[0] else {
        return Err(Error::InvalidArgument(
            "MetadataCastWhere expects metadata condition input".into(),
        ));
    };
    let MetadataScalarTensorRef::Scalar(on_true) = inputs[1] else {
        return Err(Error::InvalidArgument(
            "MetadataCastWhere expects scalar on_true input".into(),
        ));
    };
    let MetadataScalarTensorRef::Scalar(on_false) = inputs[2] else {
        return Err(Error::InvalidArgument(
            "MetadataCastWhere expects scalar on_false input".into(),
        ));
    };
    Ok((cond, on_true, on_false))
}

pub(crate) fn validate_pointwise_cast_bridge_inputs<'a, S: Scalar>(
    inputs: &'a [MetadataScalarTensorRef<'a, S>],
) -> Result<MetadataTensorRef<'a>> {
    if inputs.len() != 1 {
        return Err(Error::InvalidArgument(format!(
            "MetadataCastPointwise expects 1 input(s) (got {})",
            inputs.len()
        )));
    }
    let MetadataScalarTensorRef::Metadata(input) = inputs[0] else {
        return Err(Error::InvalidArgument(
            "MetadataCastPointwise expects metadata input".into(),
        ));
    };
    Ok(input)
}

/// Scalar-family scalar ternary descriptor for metadata bridge reuse.
#[cfg(feature = "cuda")]
pub(crate) fn scalar_where_desc() -> ScalarPrimsDescriptor {
    ScalarPrimsDescriptor::PointwiseTernary {
        op: ScalarTernaryOp::Where,
    }
}
