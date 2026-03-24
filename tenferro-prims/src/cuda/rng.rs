use tenferro_algebra::Standard;
use tenferro_device::{Error, Generator, LogicalMemorySpace, Result};
use tenferro_tensor::Tensor;

use crate::{CudaBackend, CudaContext, RngPrimsDescriptor, TensorRngPrims};

/// CUDA execution plan for the RNG family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CudaRngPlan;
/// let _ = std::mem::size_of::<CudaRngPlan>();
/// ```
pub type CudaRngPlan = (RngPrimsDescriptor, Vec<usize>);

fn ensure_cuda_tensor<T>(tensor: &Tensor<T>, device_id: usize, label: &str) -> Result<()>
where
    T: tenferro_algebra::Scalar,
{
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

fn tensor_device_mut_ptr<T>(tensor: &mut Tensor<T>, label: &str) -> Result<*mut T>
where
    T: tenferro_algebra::Scalar,
{
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut T)
        .ok_or_else(|| Error::DeviceError(format!("{label} buffer is not on GPU")))
}

fn validate_rng_plan(desc: &RngPrimsDescriptor, shapes: &[&[usize]]) -> Result<Vec<usize>> {
    crate::validate_shape_count(shapes, 1, "CudaRng")?;
    match desc {
        RngPrimsDescriptor::Integer { low, high } if low >= high => Err(Error::InvalidArgument(
            format!("CudaRng requires low < high (got low={low}, high={high})"),
        )),
        _ => Ok(shapes[0].to_vec()),
    }
}

impl TensorRngPrims<Standard<f64>> for CudaBackend {
    type Plan = CudaRngPlan;
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        let output_shape = validate_rng_plan(desc, shapes)?;
        match desc {
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Ok((desc.clone(), output_shape))
            }
            RngPrimsDescriptor::Integer { .. } => Err(Error::InvalidArgument(
                "integer RNG planning is only supported for Tensor<i32>".into(),
            )),
        }
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        generator: &mut Generator,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        crate::validate_shape_eq(output.dims(), &plan.1, "CudaRng output")?;
        ensure_cuda_tensor(output, ctx.device_id(), "CudaRng output")?;
        let runtime = ctx.shared_runtime();
        let dst = tensor_device_mut_ptr(output, "CudaRng output")?;
        let dst_len = output.buffer().len();
        match &plan.0 {
            RngPrimsDescriptor::Uniform => unsafe {
                runtime.rng_fill_uniform_f64_raw(
                    generator,
                    dst,
                    dst_len,
                    output.dims(),
                    output.strides(),
                    output.offset(),
                )
            },
            RngPrimsDescriptor::Normal => unsafe {
                runtime.rng_fill_normal_f64_raw(
                    generator,
                    dst,
                    dst_len,
                    output.dims(),
                    output.strides(),
                    output.offset(),
                )
            },
            RngPrimsDescriptor::Integer { .. } => Err(Error::InvalidArgument(
                "integer RNG execution is only supported for Tensor<i32>".into(),
            )),
        }
    }

    fn has_rng_support(desc: RngPrimsDescriptor) -> bool {
        matches!(
            desc,
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal
        )
    }
}

impl TensorRngPrims<Standard<i32>> for CudaBackend {
    type Plan = CudaRngPlan;
    type Context = CudaContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        let output_shape = validate_rng_plan(desc, shapes)?;
        match desc {
            RngPrimsDescriptor::Integer { .. } => Ok((desc.clone(), output_shape)),
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Err(Error::InvalidArgument(
                    "floating-point RNG planning is only supported for Tensor<f64>".into(),
                ))
            }
        }
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        generator: &mut Generator,
        output: &mut Tensor<i32>,
    ) -> Result<()> {
        crate::validate_shape_eq(output.dims(), &plan.1, "CudaRng output")?;
        ensure_cuda_tensor(output, ctx.device_id(), "CudaRng output")?;
        let runtime = ctx.shared_runtime();
        let dst = tensor_device_mut_ptr(output, "CudaRng output")?;
        let dst_len = output.buffer().len();
        match &plan.0 {
            RngPrimsDescriptor::Integer { low, high } => unsafe {
                runtime.rng_fill_i32_raw(
                    generator,
                    *low,
                    *high,
                    dst,
                    dst_len,
                    output.dims(),
                    output.strides(),
                    output.offset(),
                )
            },
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Err(Error::InvalidArgument(
                    "floating-point RNG execution is only supported for Tensor<f64>".into(),
                ))
            }
        }
    }

    fn has_rng_support(desc: RngPrimsDescriptor) -> bool {
        matches!(desc, RngPrimsDescriptor::Integer { .. })
    }
}
