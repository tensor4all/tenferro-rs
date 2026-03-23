use std::any::TypeId;

use num_traits::NumCast;
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::shape_helpers::broadcast_tensor_to_shape;
use crate::{
    scalar_where_desc, supports_metadata_cast, validate_metadata_cast_shapes,
    validate_pointwise_cast_bridge_inputs, validate_where_bridge_inputs, CudaBackend, CudaContext,
    MetadataCastPrimsDescriptor, MetadataTensorRef, TensorMetadataCastPrims, TensorScalarPrims,
};

fn ensure_cuda_tensor<T: Scalar>(tensor: &Tensor<T>, device_id: usize, label: &str) -> Result<()> {
    match tensor.logical_memory_space() {
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } if tensor_device == device_id => Ok(()),
        LogicalMemorySpace::GpuMemory {
            device_id: tensor_device,
        } => Err(Error::DeviceError(format!(
            "{label} is on CUDA device {tensor_device}, expected device {device_id}"
        ))),
        other => Err(Error::DeviceError(format!(
            "{label} is not resident on CUDA device {device_id}: {other:?}"
        ))),
    }
}

fn tensor_device_ptr_ref<T: Scalar>(tensor: &Tensor<T>, label: &str) -> Result<*const T> {
    tensor
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn tensor_device_ptr_mut<T: Scalar>(tensor: &mut Tensor<T>, label: &str) -> Result<*mut T> {
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut T)
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn supports_cuda_metadata_cast_output<S: Scalar + 'static>() -> bool {
    TypeId::of::<S>() == TypeId::of::<f32>() || TypeId::of::<S>() == TypeId::of::<f64>()
}

fn execute_pointwise_cast_cuda<S>(
    ctx: &CudaContext,
    input: MetadataTensorRef<'_>,
    alpha: S,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: Scalar + NumCast + 'static,
{
    if !supports_cuda_metadata_cast_output::<S>() {
        return Err(Error::InvalidArgument(format!(
            "metadata cast output type {} is not supported on CudaBackend",
            std::any::type_name::<S>()
        )));
    }

    let device_id = ctx.device_id();
    ensure_cuda_tensor(output, device_id, "MetadataCast output")?;
    let runtime = ctx.shared_runtime();
    let dst_len = output.buffer().len();
    let dst_strides = output.strides().to_vec();
    let dst_offset = output.offset();

    match input {
        MetadataTensorRef::Bool(input) => {
            let input = broadcast_tensor_to_shape(input, output.dims(), "MetadataCast input")?;
            ensure_cuda_tensor(&input, device_id, "MetadataCast input")?;
            let input_len = input.buffer().len();
            let input_strides = input.strides().to_vec();
            let input_offset = input.offset();
            let input_ptr = tensor_device_ptr_ref(&input, "MetadataCast input")?;
            let dst_ptr = tensor_device_ptr_mut(output, "MetadataCast output")?;
            if TypeId::of::<S>() == TypeId::of::<f32>() {
                unsafe {
                    runtime.metadata_cast_bool_f32(
                        input_ptr.cast::<u8>(),
                        input_len,
                        dst_ptr.cast::<f32>(),
                        dst_len,
                        output.dims(),
                        &input_strides,
                        input_offset,
                        &dst_strides,
                        dst_offset,
                        NumCast::from(alpha).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata alpha to f32".into())
                        })?,
                        NumCast::from(beta).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata beta to f32".into())
                        })?,
                    )
                }
            } else {
                unsafe {
                    runtime.metadata_cast_bool_f64(
                        input_ptr.cast::<u8>(),
                        input_len,
                        dst_ptr.cast::<f64>(),
                        dst_len,
                        output.dims(),
                        &input_strides,
                        input_offset,
                        &dst_strides,
                        dst_offset,
                        NumCast::from(alpha).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata alpha to f64".into())
                        })?,
                        NumCast::from(beta).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata beta to f64".into())
                        })?,
                    )
                }
            }
        }
        MetadataTensorRef::I32(input) => {
            let input = broadcast_tensor_to_shape(input, output.dims(), "MetadataCast input")?;
            ensure_cuda_tensor(&input, device_id, "MetadataCast input")?;
            let input_len = input.buffer().len();
            let input_strides = input.strides().to_vec();
            let input_offset = input.offset();
            let input_ptr = tensor_device_ptr_ref(&input, "MetadataCast input")?;
            let dst_ptr = tensor_device_ptr_mut(output, "MetadataCast output")?;
            if TypeId::of::<S>() == TypeId::of::<f32>() {
                unsafe {
                    runtime.metadata_cast_i32_f32(
                        input_ptr.cast::<i32>(),
                        input_len,
                        dst_ptr.cast::<f32>(),
                        dst_len,
                        output.dims(),
                        &input_strides,
                        input_offset,
                        &dst_strides,
                        dst_offset,
                        NumCast::from(alpha).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata alpha to f32".into())
                        })?,
                        NumCast::from(beta).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata beta to f32".into())
                        })?,
                    )
                }
            } else {
                unsafe {
                    runtime.metadata_cast_i32_f64(
                        input_ptr.cast::<i32>(),
                        input_len,
                        dst_ptr.cast::<f64>(),
                        dst_len,
                        output.dims(),
                        &input_strides,
                        input_offset,
                        &dst_strides,
                        dst_offset,
                        NumCast::from(alpha).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata alpha to f64".into())
                        })?,
                        NumCast::from(beta).ok_or_else(|| {
                            Error::InvalidArgument("cannot cast metadata beta to f64".into())
                        })?,
                    )
                }
            }
        }
    }
}

impl<S> TensorMetadataCastPrims<S> for CudaBackend
where
    S: Scalar + NumCast + 'static,
    CudaBackend: TensorScalarPrims<Standard<S>, Context = CudaContext>,
{
    type Plan = MetadataCastPrimsDescriptor;
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &MetadataCastPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        validate_metadata_cast_shapes(desc, shapes, "MetadataCast")?;
        if !supports_metadata_cast(desc) || !supports_cuda_metadata_cast_output::<S>() {
            return Err(Error::InvalidArgument(format!(
                "metadata cast descriptor {desc:?} is not supported on CudaBackend for {}",
                std::any::type_name::<S>()
            )));
        }
        Ok(desc.clone())
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[crate::MetadataScalarTensorRef<'_, S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        match plan {
            MetadataCastPrimsDescriptor::PointwiseCast { .. } => {
                let input = validate_pointwise_cast_bridge_inputs(inputs)?;
                execute_pointwise_cast_cuda(ctx, input, alpha, beta, output)
            }
            MetadataCastPrimsDescriptor::Where { .. } => {
                let (cond, on_true, on_false) = validate_where_bridge_inputs(inputs)?;
                let mut cond_scalar = Tensor::<S>::zeros(
                    output.dims(),
                    output.logical_memory_space(),
                    MemoryOrder::ColumnMajor,
                )?;
                execute_pointwise_cast_cuda(ctx, cond, S::one(), S::zero(), &mut cond_scalar)?;
                let on_true =
                    broadcast_tensor_to_shape(on_true, output.dims(), "MetadataCastWhere true")?;
                let on_false =
                    broadcast_tensor_to_shape(on_false, output.dims(), "MetadataCastWhere false")?;
                let scalar_desc = scalar_where_desc();
                let scalar_plan = <CudaBackend as TensorScalarPrims<Standard<S>>>::plan(
                    ctx,
                    &scalar_desc,
                    &[
                        cond_scalar.dims(),
                        on_true.dims(),
                        on_false.dims(),
                        output.dims(),
                    ],
                )?;
                <CudaBackend as TensorScalarPrims<Standard<S>>>::execute(
                    ctx,
                    &scalar_plan,
                    alpha,
                    &[&cond_scalar, &on_true, &on_false],
                    beta,
                    output,
                )
            }
        }
    }

    fn has_metadata_cast_support(desc: MetadataCastPrimsDescriptor) -> bool {
        supports_metadata_cast(&desc) && supports_cuda_metadata_cast_output::<S>()
    }
}
