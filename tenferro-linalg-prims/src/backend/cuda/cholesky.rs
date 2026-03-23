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
use crate::backend::linalg_utils::{prepare_matrix_operand, MatrixOperandTransposeType};
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_square};
use crate::CholeskyTensorExResult;

#[cfg(feature = "cuda")]
const CUBLAS_FILL_MODE_LOWER: i32 = 0;

#[cfg(feature = "cuda")]
pub(super) fn has_cholesky_support<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_cholesky_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn cholesky_dtype<T: CudaLinalgScalar>() -> Result<CudaDataType> {
    if has_cholesky_support::<T>() {
        Ok(T::cuda_data_type())
    } else {
        Err(Error::DeviceError(format!(
            "CUDA cholesky currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )))
    }
}

#[cfg(feature = "cuda")]
fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidArgument(format!("{label} overflow: {lhs} * {rhs}")))
}

#[cfg(feature = "cuda")]
pub(super) fn cholesky<T>(ctx: &mut tenferro_prims::CudaContext, a: &Tensor<T>) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    let dtype = cholesky_dtype::<T>()?;
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);
    let l = prepare_matrix_operand(ctx, a, MatrixOperandTransposeType::None)?;
    if n == 0 || bc == 0 {
        return Ok(l.tril(0));
    }

    let runtime = load_runtime(ctx)?;
    let l_base = context_device_ptr(ctx, &l, "cholesky l")?.cast::<T>();
    let l_offset = l.offset() as usize;
    let mat_size = checked_mul(n, n, "cholesky matrix size")?;
    let lda = i32::try_from(n)
        .map_err(|_| Error::InvalidArgument("cholesky lda exceeds i32 range".into()))?;
    let n_i32 = lda;
    let first_l_ptr = unsafe { l_base.add(l_offset) }.cast::<c_void>();
    let lwork = runtime.cusolver_api().potrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        CUBLAS_FILL_MODE_LOWER,
        n_i32,
        first_l_ptr,
        lda,
    )?;
    let workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork).map_err(|_| {
                Error::InvalidArgument(format!("cholesky workspace size was negative: {lwork}"))
            })?,
            std::mem::size_of::<T>(),
            "cholesky workspace bytes",
        )?,
        "cudaMalloc(cholesky workspace)",
    )?;
    let info_alloc =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(cholesky info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let l_ptr = unsafe { l_base.add(l_offset + batch * mat_size) }.cast::<c_void>();
        runtime.cusolver_api().potrf(
            dtype,
            runtime.cusolver_handle.raw,
            CUBLAS_FILL_MODE_LOWER,
            n_i32,
            l_ptr,
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
        if batch_info < 0 {
            return Err(Error::DeviceError(format!(
                "cholesky reported an invalid parameter at position {}",
                -batch_info
            )));
        }
        if batch_info > 0 {
            return Err(Error::InvalidArgument(format!(
                "cholesky: matrix is not positive definite (minor {batch_info})"
            )));
        }
    }

    Ok(l.tril(0))
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
    let dtype = cholesky_dtype::<T>()?;
    let (n, batch_dims) = validate_square(a)?;
    let bc = batch_count(batch_dims);

    let mut l = Tensor::zeros(
        a.dims(),
        a.logical_memory_space(),
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )?;
    if n == 0 || bc == 0 {
        return Ok(CholeskyTensorExResult {
            l,
            info: vec![0; bc],
        });
    }

    let a_work = prepare_matrix_operand(ctx, a, MatrixOperandTransposeType::None)?;
    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "cholesky a")?.cast::<T>();
    let l_base = context_device_ptr(ctx, &l, "cholesky l")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let l_offset = l.offset() as usize;
    let mat_size = checked_mul(n, n, "cholesky matrix size")?;
    let lda = i32::try_from(n)
        .map_err(|_| Error::InvalidArgument("cholesky lda exceeds i32 range".into()))?;
    let n_i32 = lda;
    let first_a_ptr = unsafe { a_base.add(a_offset) }.cast::<c_void>();
    let lwork = runtime.cusolver_api().potrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        CUBLAS_FILL_MODE_LOWER,
        n_i32,
        first_a_ptr,
        lda,
    )?;
    let workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork).map_err(|_| {
                Error::InvalidArgument(format!("cholesky workspace size was negative: {lwork}"))
            })?,
            std::mem::size_of::<T>(),
            "cholesky workspace bytes",
        )?,
        "cudaMalloc(cholesky workspace)",
    )?;

    let mut info = vec![0i32; bc];
    let info_alloc =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(cholesky info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * mat_size) }.cast::<c_void>();
        let l_ptr = unsafe { l_base.add(l_offset + batch * mat_size) };

        runtime.cusolver_api().potrf(
            dtype,
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
