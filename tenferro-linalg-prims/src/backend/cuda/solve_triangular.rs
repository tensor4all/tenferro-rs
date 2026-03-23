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
use super::runtime::{context_device_ptr, copy_device_to_host, load_runtime};
#[cfg(feature = "cuda")]
use super::scalar_type::CudaDataType;
use super::scalar_type::CudaLinalgScalar;
#[cfg(feature = "cuda")]
use super::wrappers::{
    CUBLAS_DIAG_NON_UNIT, CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
    CUBLAS_SIDE_LEFT,
};
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{
    batch_count, materialize_broadcasted_batches_resolving_conj, validate_solve_rhs_shape,
    validate_square, BroadcastBatchIndexer,
};
#[cfg(feature = "cuda")]
use tenferro_device::cuda::runtime::ComplexRealUnaryOp;
#[cfg(feature = "cuda")]
use tenferro_device::cuda::runtime::{RealBinaryOp, RealReductionOp, RealUnaryOp};

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
fn solve_triangular_supported<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(feature = "cuda")]
pub(super) fn has_solve_triangular_support<T: CudaLinalgScalar>() -> bool {
    solve_triangular_supported::<T>()
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_solve_triangular_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn validate_nonzero_diagonal<T>(
    ctx: &mut tenferro_prims::CudaContext,
    tensor: &Tensor<T>,
) -> Result<()>
where
    T: CudaLinalgScalar,
{
    let diagonal = tensor.diagonal(&[(0, 1)])?;
    let runtime = ctx.shared_runtime();
    let dtype = T::cuda_data_type();
    let dims = diagonal.dims();
    let strides = diagonal.strides();
    let offset = diagonal.offset();
    let kept_axes: [usize; 0] = [];
    let reduced_axes: Vec<usize> = (0..diagonal.ndim()).collect();

    match dtype {
        CudaDataType::F32 => unsafe {
            let abs_diagonal: Tensor<f32> = Tensor::zeros(
                diagonal.dims(),
                diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let zero: Tensor<f32> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let nonzero_mask: Tensor<f32> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let reduced: Tensor<f32> = Tensor::zeros(
                &[],
                nonzero_mask.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            runtime.pointwise_unary_real_f32_raw(
                RealUnaryOp::Abs,
                1.0,
                context_device_ptr(ctx, &diagonal, "solve_triangular diagonal")?.cast::<f32>(),
                dims,
                strides,
                offset,
                0.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f32>(),
                abs_diagonal.strides(),
                abs_diagonal.offset(),
            )?;
            runtime.pointwise_binary_real_f32_raw(
                RealBinaryOp::Greater,
                1.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f32>(),
                dims,
                abs_diagonal.strides(),
                abs_diagonal.offset(),
                context_device_ptr(ctx, &zero, "solve_triangular zero buffer")?.cast::<f32>(),
                zero.strides(),
                zero.offset(),
                0.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f32>(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
            )?;
            runtime.reduce_real_f32_raw(
                RealReductionOp::Prod,
                1.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f32>(),
                nonzero_mask.dims(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
                0.0,
                context_device_ptr(ctx, &reduced, "solve_triangular reduced mask")?.cast::<f32>(),
                reduced.dims(),
                reduced.strides(),
                reduced.offset(),
                &kept_axes,
                &reduced_axes,
            )?;
            let mut host = [0.0f32; 1];
            copy_device_to_host(
                ctx,
                context_device_ptr(ctx, &reduced, "solve_triangular diagonal validation")?
                    .cast::<c_void>(),
                &mut host,
                "cudaMemcpyDtoH(solve_triangular diagonal validation)",
            )?;
            if host[0] != 1.0 {
                return Err(Error::InvalidArgument(
                    "solve_triangular: zero diagonal".into(),
                ));
            }
        },
        CudaDataType::F64 => unsafe {
            let abs_diagonal: Tensor<f64> = Tensor::zeros(
                diagonal.dims(),
                diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let zero: Tensor<f64> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let nonzero_mask: Tensor<f64> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let reduced: Tensor<f64> = Tensor::zeros(
                &[],
                nonzero_mask.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            runtime.pointwise_unary_real_f64_raw(
                RealUnaryOp::Abs,
                1.0,
                context_device_ptr(ctx, &diagonal, "solve_triangular diagonal")?.cast::<f64>(),
                dims,
                strides,
                offset,
                0.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f64>(),
                abs_diagonal.strides(),
                abs_diagonal.offset(),
            )?;
            runtime.pointwise_binary_real_f64_raw(
                RealBinaryOp::Greater,
                1.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f64>(),
                dims,
                abs_diagonal.strides(),
                abs_diagonal.offset(),
                context_device_ptr(ctx, &zero, "solve_triangular zero buffer")?.cast::<f64>(),
                zero.strides(),
                zero.offset(),
                0.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f64>(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
            )?;
            runtime.reduce_real_f64_raw(
                RealReductionOp::Prod,
                1.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f64>(),
                nonzero_mask.dims(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
                0.0,
                context_device_ptr(ctx, &reduced, "solve_triangular reduced mask")?.cast::<f64>(),
                reduced.dims(),
                reduced.strides(),
                reduced.offset(),
                &kept_axes,
                &reduced_axes,
            )?;
            let mut host = [0.0f64; 1];
            copy_device_to_host(
                ctx,
                context_device_ptr(ctx, &reduced, "solve_triangular diagonal validation")?
                    .cast::<c_void>(),
                &mut host,
                "cudaMemcpyDtoH(solve_triangular diagonal validation)",
            )?;
            if host[0] != 1.0 {
                return Err(Error::InvalidArgument(
                    "solve_triangular: zero diagonal".into(),
                ));
            }
        },
        CudaDataType::Complex32 => unsafe {
            let abs_diagonal: Tensor<f32> = Tensor::zeros(
                diagonal.dims(),
                diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let zero: Tensor<f32> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let nonzero_mask: Tensor<f32> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let reduced: Tensor<f32> = Tensor::zeros(
                &[],
                nonzero_mask.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            runtime.pointwise_unary_complex32_to_real_f32_raw(
                ComplexRealUnaryOp::Abs,
                1.0,
                context_device_ptr(ctx, &diagonal, "solve_triangular diagonal")?
                    .cast::<num_complex::Complex32>(),
                dims,
                strides,
                offset,
                0.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f32>(),
                abs_diagonal.strides(),
                abs_diagonal.offset(),
            )?;
            runtime.pointwise_binary_real_f32_raw(
                RealBinaryOp::Greater,
                1.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f32>(),
                dims,
                abs_diagonal.strides(),
                abs_diagonal.offset(),
                context_device_ptr(ctx, &zero, "solve_triangular zero buffer")?.cast::<f32>(),
                zero.strides(),
                zero.offset(),
                0.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f32>(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
            )?;
            runtime.reduce_real_f32_raw(
                RealReductionOp::Prod,
                1.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f32>(),
                nonzero_mask.dims(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
                0.0,
                context_device_ptr(ctx, &reduced, "solve_triangular reduced mask")?.cast::<f32>(),
                reduced.dims(),
                reduced.strides(),
                reduced.offset(),
                &kept_axes,
                &reduced_axes,
            )?;
            let mut host = [0.0f32; 1];
            copy_device_to_host(
                ctx,
                context_device_ptr(ctx, &reduced, "solve_triangular diagonal validation")?
                    .cast::<c_void>(),
                &mut host,
                "cudaMemcpyDtoH(solve_triangular diagonal validation)",
            )?;
            if host[0] != 1.0 {
                return Err(Error::InvalidArgument(
                    "solve_triangular: zero diagonal".into(),
                ));
            }
        },
        CudaDataType::Complex64 => unsafe {
            let abs_diagonal: Tensor<f64> = Tensor::zeros(
                diagonal.dims(),
                diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let zero: Tensor<f64> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let nonzero_mask: Tensor<f64> = Tensor::zeros(
                abs_diagonal.dims(),
                abs_diagonal.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            let reduced: Tensor<f64> = Tensor::zeros(
                &[],
                nonzero_mask.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )?;
            runtime.pointwise_unary_complex64_to_real_f64_raw(
                ComplexRealUnaryOp::Abs,
                1.0,
                context_device_ptr(ctx, &diagonal, "solve_triangular diagonal")?
                    .cast::<num_complex::Complex64>(),
                dims,
                strides,
                offset,
                0.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f64>(),
                abs_diagonal.strides(),
                abs_diagonal.offset(),
            )?;
            runtime.pointwise_binary_real_f64_raw(
                RealBinaryOp::Greater,
                1.0,
                context_device_ptr(ctx, &abs_diagonal, "solve_triangular abs diagonal")?
                    .cast::<f64>(),
                dims,
                abs_diagonal.strides(),
                abs_diagonal.offset(),
                context_device_ptr(ctx, &zero, "solve_triangular zero buffer")?.cast::<f64>(),
                zero.strides(),
                zero.offset(),
                0.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f64>(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
            )?;
            runtime.reduce_real_f64_raw(
                RealReductionOp::Prod,
                1.0,
                context_device_ptr(ctx, &nonzero_mask, "solve_triangular diagonal mask")?
                    .cast::<f64>(),
                nonzero_mask.dims(),
                nonzero_mask.strides(),
                nonzero_mask.offset(),
                0.0,
                context_device_ptr(ctx, &reduced, "solve_triangular reduced mask")?.cast::<f64>(),
                reduced.dims(),
                reduced.strides(),
                reduced.offset(),
                &kept_axes,
                &reduced_axes,
            )?;
            let mut host = [0.0f64; 1];
            copy_device_to_host(
                ctx,
                context_device_ptr(ctx, &reduced, "solve_triangular diagonal validation")?
                    .cast::<c_void>(),
                &mut host,
                "cudaMemcpyDtoH(solve_triangular diagonal validation)",
            )?;
            if host[0] != 1.0 {
                return Err(Error::InvalidArgument(
                    "solve_triangular: zero diagonal".into(),
                ));
            }
        },
    }

    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn solve_triangular<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    let (n, batch_dims) = validate_square(a)?;
    let rhs = validate_solve_rhs_shape(b, n, batch_dims, "solve_triangular")?;
    let bc = batch_count(&rhs.output_batch_dims);
    let x_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        b,
        rhs.structural_rank,
        &rhs.rhs_batch_indexer,
        "solve_triangular",
        "b",
    )?;

    if n == 0 || bc == 0 {
        return Ok(x_work);
    }

    if !solve_triangular_supported::<T>() {
        return Err(Error::DeviceError(format!(
            "CUDA solve_triangular currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )));
    }

    let dtype = T::cuda_data_type();
    let n_i32 = as_i32(n, "solve_triangular n")?;
    let nrhs_i32 = as_i32(rhs.nrhs, "solve_triangular nrhs")?;
    let lda = as_i32(n, "solve_triangular lda")?;
    let ldb = as_i32(n, "solve_triangular ldb")?;
    let a_stride = checked_mul(n, n, "solve_triangular a_stride")?;
    let b_stride = checked_mul(n, rhs.nrhs, "solve_triangular b_stride")?;

    let a_batch_indexer =
        BroadcastBatchIndexer::new(batch_dims, &rhs.output_batch_dims, "solve_triangular", "a")?;
    let a_work = materialize_broadcasted_batches_resolving_conj(
        ctx,
        a,
        2,
        &a_batch_indexer,
        "solve_triangular",
        "a",
    )?;
    validate_nonzero_diagonal(ctx, &a_work)?;

    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "solve_triangular a")?.cast::<T>();
    let x_base = context_device_ptr(ctx, &x_work, "solve_triangular x")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let x_offset = x_work.offset() as usize;
    let alpha = T::one();
    let uplo = if upper {
        CUBLAS_FILL_MODE_UPPER
    } else {
        CUBLAS_FILL_MODE_LOWER
    };

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        let x_ptr = unsafe { x_base.add(x_offset + batch * b_stride) }.cast::<c_void>();
        runtime.cublas_api().trsm(
            dtype,
            runtime.cublas_handle.raw,
            CUBLAS_SIDE_LEFT,
            uplo,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n_i32,
            nrhs_i32,
            (&alpha as *const T).cast::<c_void>(),
            a_ptr,
            lda,
            x_ptr,
            ldb,
        )?;
    }

    Ok(x_work)
}

#[cfg(not(feature = "cuda"))]
pub(super) fn solve_triangular<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
    _b: &Tensor<T>,
    _upper: bool,
) -> Result<Tensor<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("solve_triangular")
}
