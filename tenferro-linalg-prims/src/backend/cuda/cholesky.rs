#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "cuda")]
use tenferro_device::Error;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

#[cfg(feature = "cuda")]
use super::runtime::{context_device_ptr, copy_device_to_host, load_runtime, DeviceAllocation};
#[cfg(feature = "cuda")]
use super::scalar_type::CudaDataType;
use super::scalar_type::CudaLinalgScalar;
#[cfg(feature = "cuda")]
use crate::backend::linalg_utils::clone_batched_column_major;
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_square};
use crate::CholeskyTensorExResult;

#[cfg(feature = "cuda")]
const CUBLAS_FILL_MODE_LOWER: i32 = 0;

#[cfg(feature = "cuda")]
pub(super) fn has_cholesky_support<T: CudaLinalgScalar>() -> bool {
    matches!(T::cuda_data_type(), CudaDataType::F32 | CudaDataType::F64)
}

#[cfg(not(feature = "cuda"))]
fn has_cholesky_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn cholesky_dtype<T: CudaLinalgScalar>() -> Result<CudaDataType> {
    if has_cholesky_support::<T>() {
        Ok(T::cuda_data_type())
    } else {
        Err(Error::DeviceError(format!(
            "CUDA cholesky currently supports only f32/f64, got {:?}",
            T::cuda_data_type()
        )))
    }
}

#[cfg(feature = "cuda")]
pub(super) fn cholesky<T>(ctx: &mut tenferro_prims::CudaContext, a: &Tensor<T>) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    let result = cholesky_ex(ctx, a)?;
    if let Some(info) = result.info.iter().copied().find(|info| *info > 0) {
        return Err(Error::InvalidArgument(format!(
            "cholesky: matrix is not positive definite (minor {info})"
        )));
    }
    Ok(result.l)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn cholesky<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("cholesky")
}

#[cfg(feature = "cuda")]
pub(super) fn cholesky_ex<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<CholeskyTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    let _dtype = cholesky_dtype::<T>()?;
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);

    let mut l = Tensor::zeros(
        a.dims(),
        a.logical_memory_space(),
        tenferro_tensor::MemoryOrder::ColumnMajor,
    );
    if n == 0 || bc == 0 {
        return Ok(CholeskyTensorExResult {
            l,
            info: vec![0; bc],
        });
    }

    let a_work = clone_batched_column_major(ctx, a)?;
    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "cholesky a")?.cast::<T>();
    let l_base = context_device_ptr(ctx, &l, "cholesky l")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let l_offset = l.offset() as usize;
    let mat_size = n
        .checked_mul(n)
        .ok_or_else(|| Error::InvalidArgument("cholesky matrix size overflow".into()))?;
    let lda = i32::try_from(n)
        .map_err(|_| Error::InvalidArgument("cholesky lda exceeds i32 range".into()))?;
    let n_i32 = lda;

    let mut info = vec![0i32; bc];
    let info_alloc =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(cholesky info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * mat_size) }.cast::<c_void>();
        let l_ptr = unsafe { l_base.add(l_offset + batch * mat_size) };

        let lwork = runtime.cusolver_api().potrf_buffer_size(
            T::cuda_data_type(),
            runtime.cusolver_handle.raw,
            CUBLAS_FILL_MODE_LOWER,
            n_i32,
            a_ptr,
            lda,
        )?;
        let workspace = DeviceAllocation::alloc(
            ctx,
            usize::try_from(lwork).map_err(|_| {
                Error::InvalidArgument(format!("cholesky workspace size was negative: {lwork}"))
            })? * std::mem::size_of::<T>(),
            "cudaMalloc(cholesky workspace)",
        )?;

        runtime.cusolver_api().potrf(
            T::cuda_data_type(),
            runtime.cusolver_handle.raw,
            CUBLAS_FILL_MODE_LOWER,
            n_i32,
            a_ptr,
            lda,
            workspace.as_mut_ptr(),
            lwork,
            info_alloc.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info_alloc.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(cholesky info)",
        )?;
        let batch_info = host_info[0];
        info[batch] = batch_info;
        if batch_info < 0 {
            return Err(Error::DeviceError(format!(
                "cholesky reported an invalid parameter at position {}",
                -batch_info
            )));
        }
        if batch_info == 0 {
            unsafe {
                ctx.shared_runtime()
                    .copy_dtod_raw(a_ptr.cast::<T>(), l_ptr, mat_size)?;
            }
        }
    }

    l = l.tril(0);
    Ok(CholeskyTensorExResult { l, info })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn cholesky_ex<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<CholeskyTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("cholesky_ex")
}
