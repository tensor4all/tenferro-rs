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
use crate::backend::linalg_utils::clone_batched_column_major;
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_matrix_shape};

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
pub(super) fn svdvals_supported<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(feature = "cuda")]
fn svdvals_dtype<T: CudaLinalgScalar>() -> Result<CudaDataType> {
    if svdvals_supported::<T>() {
        Ok(T::cuda_data_type())
    } else {
        Err(Error::DeviceError(format!(
            "CUDA svdvals currently supports only f32/f64/complex, got {:?}",
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
pub(super) fn svdvals<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<Tensor<T::Real>>
where
    T: CudaLinalgScalar,
{
    let dtype = svdvals_dtype::<T>()?;
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let mut s_dims = vec![k];
    s_dims.extend_from_slice(batch_dims);
    let s = Tensor::zeros(&s_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor);

    if m == 0 || n == 0 || bc == 0 {
        return Ok(s);
    }

    let a_input = if m < n {
        let mut perm: Vec<usize> = (0..a.ndim()).collect();
        perm.swap(0, 1);
        a.permute(&perm)?
    } else {
        a.clone()
    };
    let a_work = clone_batched_column_major(ctx, &a_input)?;
    let (m_work, n_work) = if m < n { (n, m) } else { (m, n) };

    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "svdvals a")?.cast::<T>();
    let s_base = tensor_device_ptr_any(&s, "svdvals s")?.cast::<T::Real>();
    let a_offset = a_work.offset() as usize;
    let s_offset = s.offset() as usize;
    let a_stride = checked_mul(m_work, n_work, "svdvals a_stride")?;
    let s_stride = k;

    let m_i32 = as_i32(m_work, "svdvals m")?;
    let n_i32 = as_i32(n_work, "svdvals n")?;
    let lda = as_i32(m_work, "svdvals lda")?;
    let ldu = lda;
    let ldvt = as_i32(n_work.max(1), "svdvals ldvt")?;
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
                Error::InvalidArgument(format!("svdvals workspace size was negative: {lwork}"))
            })?,
            std::mem::size_of::<T>(),
            "svdvals workspace bytes",
        )?,
        "cudaMalloc(svdvals workspace)",
    )?;
    let rwork = DeviceAllocation::alloc(
        ctx,
        checked_mul(5 * k, std::mem::size_of::<T::Real>(), "svdvals rwork bytes")?,
        "cudaMalloc(svdvals rwork)",
    )?;
    let info =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(svdvals info)")?;
    let mut host_info = [0_i32; 1];
    let job_none = b'N' as c_char;

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let s_ptr = unsafe { s_base.add(s_offset + batch * s_stride) }.cast::<c_void>();
        runtime.cusolver_api().gesvd(
            dtype,
            runtime.cusolver_handle.raw,
            job_none,
            job_none,
            m_i32,
            n_i32,
            a_ptr,
            lda,
            s_ptr,
            std::ptr::null_mut(),
            ldu,
            std::ptr::null_mut(),
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
            "cudaMemcpyDtoH(svdvals info)",
        )?;
        check_info("svdvals/gesvd", host_info[0])?;
    }

    Ok(s)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn svdvals<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<Tensor<T::Real>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("svdvals")
}
