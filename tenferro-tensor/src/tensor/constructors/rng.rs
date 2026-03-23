#[cfg(feature = "cuda")]
use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{with_default_generator, Generator, LogicalMemorySpace, Result};

use super::super::Tensor;
use crate::MemoryOrder;

#[cfg(feature = "cuda")]
use crate::layout::compute_contiguous_strides;
#[cfg(feature = "cuda")]
use crate::DataBuffer;
#[cfg(feature = "cuda")]
use tenferro_device::cuda::runtime as device_cuda;

fn with_generator<'a, R, F>(
    memory_space: LogicalMemorySpace,
    generator: Option<&'a mut Generator>,
    f: F,
) -> Result<R>
where
    F: FnOnce(&mut Generator) -> Result<R>,
{
    match generator {
        Some(generator) => f(generator),
        None => with_default_generator(memory_space, f),
    }
}

fn finish_generated_allocation<T: Scalar>(
    data: Vec<T>,
    dims: &[usize],
    memory_space: LogicalMemorySpace,
    order: MemoryOrder,
) -> Result<Tensor<T>> {
    Tensor::finish_allocation(
        Tensor::main_memory_contiguous(data, dims, order),
        memory_space,
    )
}

#[cfg(feature = "cuda")]
fn gpu_generated_tensor<T: Scalar>(
    dims: &[usize],
    memory_space: LogicalMemorySpace,
    order: MemoryOrder,
) -> Result<Tensor<T>> {
    let LogicalMemorySpace::GpuMemory { device_id } = memory_space else {
        return Err(Error::DeviceError(format!(
            "expected CUDA memory space, got {memory_space:?}"
        )));
    };
    let runtime = device_cuda::get_or_init(device_id)?;
    let allocation = runtime.alloc::<T>(dims.iter().product())?;
    let tensor = Tensor::from_parts(
        unsafe {
            DataBuffer::from_gpu_parts(
                allocation.device_ptr(),
                allocation.len(),
                memory_space,
                move || drop(allocation),
            )
        },
        Arc::from(dims),
        Arc::from(compute_contiguous_strides(dims, order)),
        0,
        memory_space,
        None,
        None,
        false,
        None,
    );
    Ok(tensor)
}

impl Tensor<f64> {
    /// Create a tensor filled with uniform samples on `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::rand(
    ///     &[2, 2],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     None,
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[2, 2]);
    /// ```
    pub fn rand(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        with_generator(memory_space, generator, |generator| {
            #[cfg(feature = "cuda")]
            if matches!(memory_space, LogicalMemorySpace::GpuMemory { .. }) {
                let tensor = gpu_generated_tensor::<f64>(dims, memory_space, order)?;
                let LogicalMemorySpace::GpuMemory { device_id } = memory_space else {
                    unreachable!("gpu_generated_tensor only accepts GPU memory");
                };
                let runtime = device_cuda::get_or_init(device_id)?;
                let dst = tensor.buffer().as_device_ptr().ok_or_else(|| {
                    Error::DeviceError("CUDA RNG destination is not on GPU".into())
                })? as *mut f64;
                let dst_len = tensor.buffer().len();
                unsafe {
                    runtime.rng_fill_uniform_f64_raw(
                        generator,
                        dst,
                        dst_len,
                        tensor.dims(),
                        tensor.strides(),
                        tensor.offset(),
                    )?;
                }
                return Ok(tensor);
            }
            let n_elements: usize = dims.iter().product();
            let mut data = Vec::with_capacity(n_elements);
            for _ in 0..n_elements {
                data.push(generator.sample_uniform_f64());
            }
            finish_generated_allocation(data, dims, memory_space, order)
        })
    }

    /// Create a tensor filled with standard-normal samples.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::randn(
    ///     &[4],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     None,
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[4]);
    /// ```
    pub fn randn(
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        with_generator(memory_space, generator, |generator| {
            #[cfg(feature = "cuda")]
            if matches!(memory_space, LogicalMemorySpace::GpuMemory { .. }) {
                let tensor = gpu_generated_tensor::<f64>(dims, memory_space, order)?;
                let LogicalMemorySpace::GpuMemory { device_id } = memory_space else {
                    unreachable!("gpu_generated_tensor only accepts GPU memory");
                };
                let runtime = device_cuda::get_or_init(device_id)?;
                let dst = tensor.buffer().as_device_ptr().ok_or_else(|| {
                    Error::DeviceError("CUDA RNG destination is not on GPU".into())
                })? as *mut f64;
                let dst_len = tensor.buffer().len();
                unsafe {
                    runtime.rng_fill_normal_f64_raw(
                        generator,
                        dst,
                        dst_len,
                        tensor.dims(),
                        tensor.strides(),
                        tensor.offset(),
                    )?;
                }
                return Ok(tensor);
            }
            let n_elements: usize = dims.iter().product();
            let mut data = Vec::with_capacity(n_elements);
            for _ in 0..n_elements {
                data.push(generator.sample_standard_normal_f64());
            }
            finish_generated_allocation(data, dims, memory_space, order)
        })
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// uniform samples on `[0, 1)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<f64>::zeros(&[2, 3], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor).unwrap();
    /// let t = Tensor::<f64>::rand_like(&base, None).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn rand_like(reference: &Self, generator: Option<&mut Generator>) -> Result<Self> {
        Self::rand(
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// standard-normal samples.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<f64>::zeros(&[2, 3], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor).unwrap();
    /// let t = Tensor::<f64>::randn_like(&base, None).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn randn_like(reference: &Self, generator: Option<&mut Generator>) -> Result<Self> {
        Self::randn(
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }
}

impl Tensor<i32> {
    /// Create a tensor filled with integer samples in `[low, high)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<i32>::randint(
    ///     -2,
    ///     5,
    ///     &[2, 2],
    ///     tenferro_device::LogicalMemorySpace::MainMemory,
    ///     MemoryOrder::ColumnMajor,
    ///     None,
    /// ).unwrap();
    /// assert_eq!(t.dims(), &[2, 2]);
    /// ```
    pub fn randint(
        low: i32,
        high: i32,
        dims: &[usize],
        memory_space: LogicalMemorySpace,
        order: MemoryOrder,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        with_generator(memory_space, generator, |generator| {
            #[cfg(feature = "cuda")]
            if matches!(memory_space, LogicalMemorySpace::GpuMemory { .. }) {
                let tensor = gpu_generated_tensor::<i32>(dims, memory_space, order)?;
                let LogicalMemorySpace::GpuMemory { device_id } = memory_space else {
                    unreachable!("gpu_generated_tensor only accepts GPU memory");
                };
                let runtime = device_cuda::get_or_init(device_id)?;
                let dst = tensor.buffer().as_device_ptr().ok_or_else(|| {
                    Error::DeviceError("CUDA RNG destination is not on GPU".into())
                })? as *mut i32;
                let dst_len = tensor.buffer().len();
                unsafe {
                    runtime.rng_fill_i32_raw(
                        generator,
                        low,
                        high,
                        dst,
                        dst_len,
                        tensor.dims(),
                        tensor.strides(),
                        tensor.offset(),
                    )?;
                }
                return Ok(tensor);
            }
            let n_elements: usize = dims.iter().product();
            let mut data = Vec::with_capacity(n_elements);
            for _ in 0..n_elements {
                data.push(generator.sample_integer_i32(low, high)?);
            }
            finish_generated_allocation(data, dims, memory_space, order)
        })
    }

    /// Create a tensor with the same shape/layout convention as another tensor and fill it with
    /// integer samples in `[low, high)`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let base = Tensor::<i32>::zeros(&[2, 3], tenferro_device::LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let t = Tensor::<i32>::randint_like(&base, -2, 5, None).unwrap();
    /// assert_eq!(t.dims(), base.dims());
    /// ```
    pub fn randint_like(
        reference: &Self,
        low: i32,
        high: i32,
        generator: Option<&mut Generator>,
    ) -> Result<Self> {
        Self::randint(
            low,
            high,
            reference.dims(),
            reference.logical_memory_space(),
            Self::like_order(reference),
            generator,
        )
    }
}
