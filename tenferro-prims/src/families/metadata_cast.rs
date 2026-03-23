use num_traits::{NumCast, ToPrimitive};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::{
    for_each_index, validate_shape_count, validate_shape_eq, MetadataDType, MetadataTensorRef,
    ScalarPrimsDescriptor, ScalarTernaryOp,
};

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
            validate_shape_eq(shapes[0], shapes[1], op_name)?;
            Ok(())
        }
        MetadataCastPrimsDescriptor::Where { .. } => {
            validate_shape_count(shapes, 4, op_name)?;
            validate_shape_eq(shapes[0], shapes[1], op_name)?;
            validate_shape_eq(shapes[0], shapes[2], op_name)?;
            validate_shape_eq(shapes[0], shapes[3], op_name)?;
            Ok(())
        }
    }
}

fn cast_metadata_value<S, T>(value: T, label: &str) -> Result<S>
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

fn for_each_index_result(dims: &[usize], mut f: impl FnMut(&[usize]) -> Result<()>) -> Result<()> {
    let mut result = Ok(());
    for_each_index(dims, |idx| {
        if result.is_ok() {
            result = f(idx);
        }
    });
    result
}

pub(crate) fn cast_metadata_tensor_to_host_scalar_tensor<S>(
    input: MetadataTensorRef<'_>,
    target_dims: &[usize],
    target_strides: &[isize],
    target_offset: isize,
) -> Result<Tensor<S>>
where
    S: Scalar + NumCast,
{
    match input {
        MetadataTensorRef::I32(tensor) => cast_metadata_tensor_from_host::<S, i32, _>(
            tensor,
            target_dims,
            target_strides,
            target_offset,
            |value| cast_metadata_value::<S, i32>(value, "metadata i32 value"),
        ),
        MetadataTensorRef::Bool(tensor) => cast_metadata_tensor_from_host::<S, u8, _>(
            tensor,
            target_dims,
            target_strides,
            target_offset,
            |value| {
                cast_metadata_value::<S, u8>(if value != 0 { 1 } else { 0 }, "metadata bool value")
            },
        ),
    }
}

fn cast_metadata_tensor_from_host<S, Src, F>(
    tensor: &Tensor<Src>,
    target_dims: &[usize],
    target_strides: &[isize],
    target_offset: isize,
    cast: F,
) -> Result<Tensor<S>>
where
    S: Scalar + NumCast,
    Src: Scalar + Copy,
    F: Fn(Src) -> Result<S> + Copy,
{
    let host_tensor = tensor.to_memory_space_async(LogicalMemorySpace::MainMemory)?;
    if host_tensor.buffer().as_slice().is_none() {
        return Err(Error::DeviceError(format!(
            "metadata tensor is not host-accessible after transfer: {:?}",
            host_tensor.logical_memory_space()
        )));
    }
    let host_view = tensor_to_view(&host_tensor)?;
    let mut output = Tensor::<S>::empty_strided(
        target_dims,
        target_strides,
        target_offset,
        LogicalMemorySpace::MainMemory,
    )?;
    let mut output_view = tensor_to_view_mut(&mut output)?;
    let dims = output_view.dims().to_vec();
    for_each_index_result(&dims, |idx| {
        output_view.set(idx, cast(host_view.get(idx))?);
        Ok(())
    })?;
    Ok(output)
}

pub(crate) fn blend_cast_into_host_output<S>(
    current_output: &mut Tensor<S>,
    casted: &Tensor<S>,
    alpha: S,
    beta: S,
) -> Result<()>
where
    S: Scalar + Copy,
{
    if current_output.buffer().as_slice().is_none() {
        return Err(Error::DeviceError(format!(
            "metadata cast output tensor is not host-accessible: {:?}",
            current_output.logical_memory_space()
        )));
    }
    if casted.buffer().as_slice().is_none() {
        return Err(Error::DeviceError(format!(
            "metadata cast source tensor is not host-accessible: {:?}",
            casted.logical_memory_space()
        )));
    }
    let casted_view = tensor_to_view(casted)?;
    let mut current_view = tensor_to_view_mut(current_output)?;
    let dims = current_view.dims().to_vec();
    for_each_index_result(&dims, |idx| {
        let blended = alpha * casted_view.get(idx) + beta * current_view.get(idx);
        current_view.set(idx, blended);
        Ok(())
    })
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
pub(crate) fn scalar_where_desc() -> ScalarPrimsDescriptor {
    ScalarPrimsDescriptor::PointwiseTernary {
        op: ScalarTernaryOp::Where,
    }
}
