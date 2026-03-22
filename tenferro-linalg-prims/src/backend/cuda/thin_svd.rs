#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::os::raw::c_char;

#[cfg(not(feature = "cuda"))]
use tenferro_device::Result;
#[cfg(feature = "cuda")]
use tenferro_device::{Error, Result};
#[cfg(not(feature = "cuda"))]
use tenferro_tensor::Tensor;
#[cfg(feature = "cuda")]
use tenferro_tensor::{MemoryOrder, Tensor};

#[cfg(feature = "cuda")]
use super::runtime::{context_device_ptr, copy_device_to_host, load_runtime, DeviceAllocation};
#[cfg(feature = "cuda")]
use super::scalar_type::CudaDataType;
use super::scalar_type::CudaLinalgScalar;
#[cfg(feature = "cuda")]
use crate::backend::linalg_utils::{
    copy_batched_column_major, prepare_matrix_operand, to_matrix_operand_transpose_type,
};
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_matrix_shape};
use crate::SvdTensorResult;

#[cfg(feature = "cuda")]
fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidArgument(format!("{label} overflow: {lhs} * {rhs}")))
}

#[cfg(feature = "cuda")]
fn as_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{label} does not fit in i32: {value}")))
}

#[cfg(feature = "cuda")]
fn thin_svd_supported<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(feature = "cuda")]
pub(super) fn has_thin_svd_support<T: CudaLinalgScalar>() -> bool {
    thin_svd_supported::<T>()
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_thin_svd_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn thin_svd_dtype<T: CudaLinalgScalar>() -> Result<CudaDataType> {
    if thin_svd_supported::<T>() {
        Ok(T::cuda_data_type())
    } else {
        Err(Error::DeviceError(format!(
            "CUDA thin_svd currently supports only f32/f64/complex, got {:?}",
            T::cuda_data_type()
        )))
    }
}

#[cfg(feature = "cuda")]
fn check_info(op: &str, info: i32) -> Result<()> {
    if info == 0 {
        return Ok(());
    }
    if info < 0 {
        return Err(Error::DeviceError(format!(
            "{op} reported an invalid parameter at position {}",
            -info
        )));
    }
    Err(Error::DeviceError(format!(
        "{op} did not converge and returned info {info}"
    )))
}

#[cfg(feature = "cuda")]
fn tensor_device_ptr_any<T: tenferro_algebra::Scalar>(
    tensor: &Tensor<T>,
    label: &str,
) -> Result<*mut c_void> {
    tensor
        .buffer()
        .as_device_ptr()
        .map(|ptr| ptr as *mut c_void)
        .ok_or_else(|| Error::DeviceError(format!("{label} tensor has no GPU device pointer")))
}

#[cfg(feature = "cuda")]
pub(super) fn thin_svd<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<SvdTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    let dtype = thin_svd_dtype::<T>()?;
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let wide = m < n;

    let mut u_dims = vec![m, k];
    u_dims.extend_from_slice(batch_dims);
    let mut vt_dims = vec![k, n];
    vt_dims.extend_from_slice(batch_dims);
    let mut s_dims = vec![k];
    s_dims.extend_from_slice(batch_dims);

    let mut u = Tensor::zeros(&u_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor);
    let mut vt = Tensor::zeros(&vt_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor);
    let s = Tensor::zeros(&s_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor);

    if m == 0 || n == 0 || bc == 0 {
        return Ok(SvdTensorResult { u, s, vt });
    }

    let a_work = prepare_matrix_operand(ctx, a, to_matrix_operand_transpose_type(wide, false))?;
    let (m_work, n_work) = if wide { (n, m) } else { (m, n) };
    let runtime = load_runtime(ctx)?;

    let a_base = context_device_ptr(ctx, &a_work, "thin_svd a")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let a_stride = checked_mul(m_work, n_work, "thin_svd a_stride")?;

    let s_base = tensor_device_ptr_any(&s, "thin_svd s")?.cast::<T::Real>();
    let s_offset = s.offset() as usize;
    let s_stride = k;

    let m_i32 = as_i32(m_work, "thin_svd m")?;
    let n_i32 = as_i32(n_work, "thin_svd n")?;
    let lda = as_i32(m_work, "thin_svd lda")?;
    let ldu = as_i32(m_work, "thin_svd ldu")?;
    let ldvt = as_i32(k.max(1), "thin_svd ldvt")?;
    let lwork = runtime.cusolver_api().gesvd_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        m_i32,
        n_i32,
    )?;
    let workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork).map_err(|_| {
                Error::InvalidArgument(format!("thin_svd workspace size was negative: {lwork}"))
            })?,
            std::mem::size_of::<T>(),
            "thin_svd workspace bytes",
        )?,
        "cudaMalloc(thin_svd workspace)",
    )?;
    let rwork = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            5 * k,
            std::mem::size_of::<T::Real>(),
            "thin_svd rwork bytes",
        )?,
        "cudaMalloc(thin_svd rwork)",
    )?;
    let info =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(thin_svd info)")?;
    let mut host_info = [0_i32; 1];
    let job_thin = b'S' as c_char;

    let u_work = if wide {
        let mut dims = vec![m_work, k];
        dims.extend_from_slice(batch_dims);
        Tensor::zeros(&dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)
    } else {
        let mut dims = vec![m_work, k];
        dims.extend_from_slice(batch_dims);
        Tensor::zeros(&dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)
    };
    let vt_work = if wide {
        let mut dims = vec![k, n_work];
        dims.extend_from_slice(batch_dims);
        Tensor::zeros(&dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)
    } else {
        let mut dims = vec![k, n_work];
        dims.extend_from_slice(batch_dims);
        Tensor::zeros(&dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)
    };

    let u_base = context_device_ptr(ctx, &u_work, "thin_svd u")?.cast::<T>();
    let vt_base = context_device_ptr(ctx, &vt_work, "thin_svd vt")?.cast::<T>();
    let u_offset = u_work.offset() as usize;
    let vt_offset = vt_work.offset() as usize;
    let u_stride = checked_mul(m_work, k, "thin_svd u_stride")?;
    let vt_stride = checked_mul(k, n_work, "thin_svd vt_stride")?;

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let s_ptr = unsafe { s_base.add(s_offset + batch * s_stride) }.cast::<c_void>();
        let u_ptr = unsafe { u_base.add(u_offset + batch * u_stride) }.cast::<c_void>();
        let vt_ptr = unsafe { vt_base.add(vt_offset + batch * vt_stride) }.cast::<c_void>();
        runtime.cusolver_api().gesvd(
            dtype,
            runtime.cusolver_handle.raw,
            job_thin,
            job_thin,
            m_i32,
            n_i32,
            a_ptr,
            lda,
            s_ptr,
            u_ptr,
            ldu,
            vt_ptr,
            ldvt,
            workspace.as_mut_ptr(),
            lwork,
            rwork.as_mut_ptr(),
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(thin_svd info)",
        )?;
        check_info("thin_svd/gesvd", host_info[0])?;
    }

    if wide {
        let u_src =
            prepare_matrix_operand(ctx, &vt_work, to_matrix_operand_transpose_type(true, false))?;
        copy_batched_column_major(ctx, &u_src, &mut u)?;
        let vt_src =
            prepare_matrix_operand(ctx, &u_work, to_matrix_operand_transpose_type(true, false))?;
        copy_batched_column_major(ctx, &vt_src, &mut vt)?;
    } else {
        copy_batched_column_major(ctx, &u_work, &mut u)?;
        copy_batched_column_major(ctx, &vt_work, &mut vt)?;
    }

    Ok(SvdTensorResult { u, s, vt })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn thin_svd<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<SvdTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("thin_svd")
}
