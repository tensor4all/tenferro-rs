#[cfg(feature = "cuda")]
use std::ffi::c_void;

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
    copy_batched_column_major, prepare_matrix_operand, MatrixOperandTransposeType,
};
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_matrix_shape};
use crate::QrTensorResult;

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
fn qr_supported<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(feature = "cuda")]
pub(super) fn has_qr_support<T: CudaLinalgScalar>() -> bool {
    qr_supported::<T>()
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_qr_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn qr_dtype<T: CudaLinalgScalar>() -> Result<CudaDataType> {
    if has_qr_support::<T>() {
        Ok(T::cuda_data_type())
    } else {
        Err(Error::DeviceError(format!(
            "CUDA QR currently supports only f32/f64/complex, got {:?}",
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
        "{op} returned unexpected positive info {info}"
    )))
}

#[cfg(feature = "cuda")]
pub(super) fn qr<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<QrTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    let dtype = qr_dtype::<T>()?;
    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);

    let mut q_dims = vec![m, k];
    q_dims.extend_from_slice(batch_dims);
    let mut r_dims = vec![k, n];
    r_dims.extend_from_slice(batch_dims);

    let mut q = Tensor::zeros(&q_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)?;
    let mut r = Tensor::zeros(&r_dims, a.logical_memory_space(), MemoryOrder::ColumnMajor)?;

    if m == 0 || n == 0 || bc == 0 {
        return Ok(QrTensorResult { q, r });
    }

    let a_work = prepare_matrix_operand(ctx, a, MatrixOperandTransposeType::None)?;
    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "qr a")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let q_offset = q.offset() as usize;
    let a_stride = checked_mul(m, n, "qr a_stride")?;
    let q_stride = checked_mul(m, k, "qr q_stride")?;
    let lda = as_i32(m, "qr lda")?;
    let m_i32 = as_i32(m, "qr m")?;
    let n_i32 = as_i32(n, "qr n")?;
    let k_i32 = as_i32(k, "qr k")?;

    let tau = DeviceAllocation::alloc(
        ctx,
        checked_mul(k, bc, "qr tau count")?
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArgument("qr tau bytes overflow".into()))?,
        "cudaMalloc(qr tau)",
    )?;
    let tau_base = tau.as_mut_ptr().cast::<T>();

    let first_a_ptr = unsafe { a_base.add(a_offset) }.cast::<c_void>();
    let lwork_geqrf = runtime.cusolver_api().geqrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        m_i32,
        n_i32,
        first_a_ptr,
        lda,
    )?;
    let geqrf_workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork_geqrf).map_err(|_| {
                Error::InvalidArgument(format!(
                    "qr geqrf workspace size was negative: {lwork_geqrf}"
                ))
            })?,
            std::mem::size_of::<T>(),
            "qr geqrf workspace bytes",
        )?,
        "cudaMalloc(qr geqrf workspace)",
    )?;
    let info = DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(qr info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let tau_ptr = unsafe { tau_base.add(batch * k) }.cast::<c_void>();

        runtime.cusolver_api().geqrf(
            dtype,
            runtime.cusolver_handle.raw,
            m_i32,
            n_i32,
            a_ptr,
            lda,
            tau_ptr,
            geqrf_workspace.as_mut_ptr(),
            lwork_geqrf,
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(qr geqrf info)",
        )?;
        check_info("qr/geqrf", host_info[0])?;
    }

    let q_src = a_work.narrow(1, 0, k)?;
    copy_batched_column_major(ctx, &q_src, &mut q)?;

    let q_base = context_device_ptr(ctx, &q, "qr q")?.cast::<T>();
    let first_q_ptr = unsafe { q_base.add(q_offset) }.cast::<c_void>();
    let first_tau_ptr = unsafe { tau_base.add(0) }.cast::<c_void>();
    let lwork_orgqr = runtime.cusolver_api().orgqr_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        m_i32,
        k_i32,
        k_i32,
        first_q_ptr,
        lda,
        first_tau_ptr,
    )?;
    let orgqr_workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork_orgqr).map_err(|_| {
                Error::InvalidArgument(format!(
                    "qr orgqr workspace size was negative: {lwork_orgqr}"
                ))
            })?,
            std::mem::size_of::<T>(),
            "qr orgqr workspace bytes",
        )?,
        "cudaMalloc(qr orgqr workspace)",
    )?;

    for batch in 0..bc {
        let q_ptr = unsafe { q_base.add(q_offset + batch * q_stride) }.cast::<c_void>();
        let tau_ptr = unsafe { tau_base.add(batch * k) }.cast::<c_void>();

        runtime.cusolver_api().orgqr(
            dtype,
            runtime.cusolver_handle.raw,
            m_i32,
            k_i32,
            k_i32,
            q_ptr,
            lda,
            tau_ptr,
            orgqr_workspace.as_mut_ptr(),
            lwork_orgqr,
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(qr orgqr info)",
        )?;
        check_info("qr/orgqr", host_info[0])?;
    }

    let r_src = a_work.narrow(0, 0, k)?.triu(0);
    copy_batched_column_major(ctx, &r_src, &mut r)?;

    Ok(QrTensorResult { q, r })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn qr<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<QrTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("qr")
}
