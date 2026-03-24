#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(not(feature = "cuda"))]
use tenferro_device::Result;
#[cfg(feature = "cuda")]
use tenferro_device::{Error, Result};
#[cfg(feature = "cuda")]
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

#[cfg(feature = "cuda")]
use super::runtime::{context_device_ptr, copy_device_to_host, load_runtime, DeviceAllocation};
#[cfg(feature = "cuda")]
use super::scalar_type::CudaDataType;
use super::scalar_type::CudaLinalgScalar;
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{
    batch_count, materialize_broadcasted_batches_resolving_conj,
    materialize_broadcasted_pivot_batches, tensor_from_data_on_space, validate_lu_pivot_shape,
    validate_solve_rhs_shape, validate_square, BroadcastBatchIndexer,
};

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
fn solve_supported<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(feature = "cuda")]
pub(super) fn has_solve_support<T: CudaLinalgScalar>() -> bool {
    solve_supported::<T>()
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_solve_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
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
    Err(Error::InvalidArgument(format!(
        "{op} failed because factorization became singular at pivot {info}"
    )))
}

#[cfg(feature = "cuda")]
pub(super) fn solve_ex<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<crate::SolveTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve_ex")?;
    let bc = batch_count(&rhs.output_batch_dims);
    let x_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        b,
        rhs.structural_rank,
        &rhs.rhs_batch_indexer,
        "solve_ex",
        "b",
    )?;

    if n == 0 || bc == 0 {
        let info_shape = if rhs.output_batch_dims.is_empty() {
            vec![]
        } else {
            rhs.output_batch_dims.clone()
        };
        return Ok(crate::SolveTensorExResult {
            solution: x_work,
            info: tensor_from_data_on_space(vec![0; bc], &info_shape, a.logical_memory_space())?,
        });
    }

    if !solve_supported::<T>() {
        return Err(Error::DeviceError(format!(
            "CUDA solve_ex currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )));
    }

    let dtype = T::cuda_data_type();
    let a_stride = checked_mul(n, n, "solve_ex a_stride")?;
    let x_stride = checked_mul(n, rhs.nrhs, "solve_ex x_stride")?;
    let lda = as_i32(n, "solve_ex lda")?;
    let n_i32 = as_i32(n, "solve_ex n")?;
    let nrhs_i32 = as_i32(rhs.nrhs, "solve_ex nrhs")?;

    let a_batch_indexer =
        BroadcastBatchIndexer::new(batch_dims, &rhs.output_batch_dims, "solve_ex", "a")?;
    let a_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        a,
        2,
        &a_batch_indexer,
        "solve_ex",
        "a",
    )?;
    let x_out = Tensor::zeros(
        &rhs.output_dims,
        a.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let runtime = load_runtime(ctx)?;

    let a_base = context_device_ptr(ctx, &a_work, "solve_ex a")?.cast::<T>();
    let x_base = context_device_ptr(ctx, &x_work, "solve_ex x")?.cast::<T>();
    let x_out_base = context_device_ptr(ctx, &x_out, "solve_ex solution")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let x_offset = x_work.offset() as usize;
    let x_out_offset = x_out.offset() as usize;

    let lwork = runtime.cusolver_api().getrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        n_i32,
        n_i32,
        a_base.cast::<c_void>(),
        lda,
    )?;
    let workspace_bytes = checked_mul(
        usize::try_from(lwork).map_err(|_| {
            Error::DeviceError(format!(
                "solve_ex getrf workspace size was negative: {lwork}"
            ))
        })?,
        std::mem::size_of::<T>(),
        "solve_ex workspace bytes",
    )?;

    let workspace =
        DeviceAllocation::alloc(ctx, workspace_bytes, "cudaMalloc(solve_ex workspace)")?;
    let pivots = DeviceAllocation::alloc(
        ctx,
        checked_mul(n, std::mem::size_of::<i32>(), "solve_ex pivots bytes")?,
        "cudaMalloc(solve_ex pivots)",
    )?;
    let info =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(solve_ex info)")?;
    let mut host_info = [0_i32; 1];
    let mut info_out = vec![0_i32; bc];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let x_ptr = unsafe { x_base.add(x_offset + batch * x_stride) }.cast::<c_void>();
        let x_out_ptr = unsafe { x_out_base.add(x_out_offset + batch * x_stride) };

        runtime.cusolver_api().getrf(
            dtype,
            runtime.cusolver_handle.raw,
            n_i32,
            n_i32,
            a_ptr,
            lda,
            workspace.as_mut_ptr(),
            pivots.as_mut_ptr().cast::<i32>(),
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(solve_ex getrf info)",
        )?;
        let batch_info = host_info[0];
        info_out[batch] = batch_info;
        if batch_info < 0 {
            return Err(Error::DeviceError(format!(
                "solve_ex reported an invalid parameter at position {}",
                -batch_info
            )));
        }
        if batch_info > 0 {
            continue;
        }

        runtime.cusolver_api().getrs(
            dtype,
            runtime.cusolver_handle.raw,
            n_i32,
            nrhs_i32,
            a_ptr.cast::<c_void>(),
            lda,
            pivots.as_mut_ptr().cast::<i32>(),
            x_ptr,
            lda,
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(solve_ex getrs info)",
        )?;
        let batch_info = host_info[0];
        if batch_info < 0 {
            return Err(Error::DeviceError(format!(
                "solve_ex reported an invalid parameter at position {}",
                -batch_info
            )));
        }
        if batch_info > 0 {
            info_out[batch] = batch_info;
            continue;
        }

        unsafe {
            ctx.shared_runtime()
                .copy_dtod_raw(x_ptr.cast::<T>(), x_out_ptr, x_stride)?;
        }
    }

    let info_shape = if rhs.output_batch_dims.is_empty() {
        vec![]
    } else {
        rhs.output_batch_dims.clone()
    };
    Ok(crate::SolveTensorExResult {
        solution: x_out,
        info: tensor_from_data_on_space(info_out, &info_shape, a.logical_memory_space())?,
    })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn solve_ex<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
) -> Result<crate::SolveTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("solve_ex")
}

#[cfg(feature = "cuda")]
pub(super) fn solve<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve")?;
    let bc = batch_count(&rhs.output_batch_dims);
    let x_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        b,
        rhs.structural_rank,
        &rhs.rhs_batch_indexer,
        "solve",
        "b",
    )?;

    if n == 0 || bc == 0 {
        return Ok(x_work);
    }

    if !solve_supported::<T>() {
        return Err(Error::DeviceError(format!(
            "CUDA solve currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )));
    }

    let dtype = T::cuda_data_type();
    let a_stride = checked_mul(n, n, "solve a_stride")?;
    let x_stride = checked_mul(n, rhs.nrhs, "solve x_stride")?;
    let lda = as_i32(n, "solve lda")?;
    let n_i32 = as_i32(n, "solve n")?;
    let nrhs_i32 = as_i32(rhs.nrhs, "solve nrhs")?;

    let a_batch_indexer =
        BroadcastBatchIndexer::new(batch_dims, &rhs.output_batch_dims, "solve", "a")?;
    let a_work =
        materialize_broadcasted_batches_resolving_conj(ctx, a, 2, &a_batch_indexer, "solve", "a")?;
    let runtime = load_runtime(ctx)?;

    let a_base = context_device_ptr(ctx, &a_work, "solve a")?.cast::<T>();
    let x_base = context_device_ptr(ctx, &x_work, "solve x")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let x_offset = x_work.offset() as usize;

    let lwork = runtime.cusolver_api().getrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        n_i32,
        n_i32,
        a_base.cast::<c_void>(),
        lda,
    )?;
    let workspace_bytes = checked_mul(
        usize::try_from(lwork).map_err(|_| {
            Error::DeviceError(format!("solve getrf workspace size was negative: {lwork}"))
        })?,
        std::mem::size_of::<T>(),
        "solve workspace bytes",
    )?;

    let workspace = DeviceAllocation::alloc(ctx, workspace_bytes, "cudaMalloc(solve workspace)")?;
    let pivots = DeviceAllocation::alloc(
        ctx,
        checked_mul(n, std::mem::size_of::<i32>(), "solve pivots bytes")?,
        "cudaMalloc(solve pivots)",
    )?;
    let info = DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(solve info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let x_ptr = unsafe { x_base.add(x_offset + batch * x_stride) }.cast::<c_void>();

        runtime.cusolver_api().getrf(
            dtype,
            runtime.cusolver_handle.raw,
            n_i32,
            n_i32,
            a_ptr,
            lda,
            workspace.as_mut_ptr(),
            pivots.as_mut_ptr().cast::<i32>(),
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(solve getrf info)",
        )?;
        check_info("solve/getrf", host_info[0])?;

        runtime.cusolver_api().getrs(
            dtype,
            runtime.cusolver_handle.raw,
            n_i32,
            nrhs_i32,
            a_ptr.cast::<c_void>(),
            lda,
            pivots.as_mut_ptr().cast::<i32>(),
            x_ptr,
            lda,
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(solve getrs info)",
        )?;
        check_info("solve/getrs", host_info[0])?;
    }

    Ok(x_work)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn solve<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("solve")
}

#[cfg(feature = "cuda")]
pub(super) fn lu_solve<T>(
    ctx: &mut tenferro_prims::CudaContext,
    factors: &Tensor<T>,
    pivots: &Tensor<i32>,
    b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    let (n, batch_dims) = validate_square(factors)?;
    validate_lu_pivot_shape(pivots, n, batch_dims, "lu_solve")?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "lu_solve")?;
    let bc = batch_count(&rhs.output_batch_dims);
    let factor_batch_indexer =
        BroadcastBatchIndexer::new(batch_dims, &rhs.output_batch_dims, "lu_solve", "factors")?;
    let factors_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        factors,
        2,
        &factor_batch_indexer,
        "lu_solve",
        "factors",
    )?;
    let pivots_work = materialize_broadcasted_pivot_batches(
        pivots,
        n,
        batch_dims,
        &rhs.output_batch_dims,
        "lu_solve",
    )?;
    let x_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        b,
        rhs.structural_rank,
        &rhs.rhs_batch_indexer,
        "lu_solve",
        "b",
    )?;

    if n == 0 || bc == 0 {
        return Ok(x_work);
    }

    if !solve_supported::<T>() {
        return Err(Error::DeviceError(format!(
            "CUDA lu_solve currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )));
    }

    let dtype = T::cuda_data_type();
    let factor_stride = checked_mul(n, n, "lu_solve factor_stride")?;
    let rhs_stride = checked_mul(n, rhs.nrhs, "lu_solve rhs_stride")?;
    let lda = as_i32(n, "lu_solve lda")?;
    let n_i32 = as_i32(n, "lu_solve n")?;
    let nrhs_i32 = as_i32(rhs.nrhs, "lu_solve nrhs")?;

    let runtime = load_runtime(ctx)?;
    let factor_base = context_device_ptr(ctx, &factors_work, "lu_solve factors")?.cast::<T>();
    let pivot_base = context_device_ptr(ctx, &pivots_work, "lu_solve pivots")?.cast::<i32>();
    let x_base = context_device_ptr(ctx, &x_work, "lu_solve rhs")?.cast::<T>();
    let factor_offset = factors_work.offset() as usize;
    let pivot_offset = pivots_work.offset() as usize;
    let x_offset = x_work.offset() as usize;

    let info =
        DeviceAllocation::alloc(ctx, std::mem::size_of::<i32>(), "cudaMalloc(lu_solve info)")?;
    let mut host_info = [0_i32; 1];

    for batch in 0..bc {
        let factor_ptr =
            unsafe { factor_base.add(factor_offset + batch * factor_stride) }.cast::<c_void>();
        let pivot_ptr = unsafe { pivot_base.add(pivot_offset + batch * n) };
        let x_ptr = unsafe { x_base.add(x_offset + batch * rhs_stride) };

        runtime.cusolver_api().getrs(
            dtype,
            runtime.cusolver_handle.raw,
            n_i32,
            nrhs_i32,
            factor_ptr,
            lda,
            pivot_ptr,
            x_ptr.cast::<c_void>(),
            lda,
            info.as_mut_ptr().cast::<i32>(),
        )?;
        copy_device_to_host(
            ctx,
            info.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(lu_solve getrs info)",
        )?;
        check_info("lu_solve/getrs", host_info[0])?;
    }

    Ok(x_work)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn lu_solve<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _factors: &Tensor<T>,
    _pivots: &Tensor<i32>,
    _b: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("lu_solve")
}
