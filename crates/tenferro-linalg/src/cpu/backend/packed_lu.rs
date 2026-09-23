//! Batched packed-LU solve entry points shared by both CPU providers.
//!
//! `lu_solve_prepared` and the fused `lu_factor_solve` both copy the RHS once
//! into the output buffer and hand the whole batch to one provider kernel,
//! which loops over matrices with scratch allocated once per call. No
//! per-matrix tensor, permutation vector, or triangular-solve intermediate is
//! materialized.

#[cfg(any(not(feature = "cpu-faer"), not(feature = "cpu-blas")))]
use super::unsupported_provider;
use super::{
    batched_vector_rhs_shape_of, checked_product, has_zero_dim, square_matrix_dim,
    CpuLinalgProvider,
};
#[cfg(any(not(feature = "cpu-faer"), not(feature = "cpu-blas")))]
use tenferro_cpu::CpuBackendKind;

#[allow(unused_imports)]
use super::linalg;
use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_cpu::CpuExecutionContext;
use tenferro_tensor::{Error, Tensor, TensorScalar, TypedTensor};

/// Per-dtype dispatch from a CPU provider to its packed-LU batch kernels.
pub(super) trait CpuPackedLu: TensorScalar + PoolScalar + Copy {
    /// Solve `op(A) X = B` in place for every batch from packed factors.
    #[allow(clippy::too_many_arguments)]
    fn prepared_solve(
        provider: CpuLinalgProvider,
        ctx: &CpuExecutionContext<'_>,
        op: &'static str,
        n: usize,
        nrhs: usize,
        packed_lu: &[Self],
        pivots: &[i32],
        output: &mut [Self],
        transpose_a: bool,
        conjugate_a: bool,
    ) -> tenferro_tensor::Result<()>;

    /// Factor `A` in place and solve `A X = B` for every batch.
    #[allow(clippy::too_many_arguments)]
    fn factor_solve(
        provider: CpuLinalgProvider,
        ctx: &CpuExecutionContext<'_>,
        op: &'static str,
        n: usize,
        nrhs: usize,
        packed_lu: &mut [Self],
        pivots: &mut [i32],
        output: &mut [Self],
    ) -> tenferro_tensor::Result<()>;
}

macro_rules! impl_cpu_packed_lu {
    ($($scalar:ty),* $(,)?) => {
        $(
            impl CpuPackedLu for $scalar {
                fn prepared_solve(
                    provider: CpuLinalgProvider,
                    ctx: &CpuExecutionContext<'_>,
                    op: &'static str,
                    n: usize,
                    nrhs: usize,
                    packed_lu: &[Self],
                    pivots: &[i32],
                    output: &mut [Self],
                    transpose_a: bool,
                    conjugate_a: bool,
                ) -> tenferro_tensor::Result<()> {
                    let _ = (&ctx, &packed_lu, &pivots, &output, n, nrhs, transpose_a, conjugate_a);
                    match provider {
                        CpuLinalgProvider::Faer => {
                            #[cfg(feature = "cpu-faer")]
                            {
                                linalg::faer::lu_solve_prepared_batched_in_place::<$scalar>(
                                    ctx,
                                    op,
                                    (n, nrhs),
                                    packed_lu,
                                    pivots,
                                    output,
                                    (transpose_a, conjugate_a),
                                )
                            }
                            #[cfg(not(feature = "cpu-faer"))]
                            {
                                Err(unsupported_provider(op, CpuBackendKind::Faer))
                            }
                        }
                        CpuLinalgProvider::Blas => {
                            #[cfg(feature = "cpu-blas")]
                            {
                                linalg::blas::lu_solve_prepared_batched_in_place::<$scalar>(
                                    op,
                                    (n, nrhs),
                                    packed_lu,
                                    pivots,
                                    output,
                                    (transpose_a, conjugate_a),
                                )
                            }
                            #[cfg(not(feature = "cpu-blas"))]
                            {
                                Err(unsupported_provider(op, CpuBackendKind::Blas))
                            }
                        }
                    }
                }

                fn factor_solve(
                    provider: CpuLinalgProvider,
                    ctx: &CpuExecutionContext<'_>,
                    op: &'static str,
                    n: usize,
                    nrhs: usize,
                    packed_lu: &mut [Self],
                    pivots: &mut [i32],
                    output: &mut [Self],
                ) -> tenferro_tensor::Result<()> {
                    let _ = (&ctx, &packed_lu, &pivots, &output, n, nrhs);
                    match provider {
                        CpuLinalgProvider::Faer => {
                            #[cfg(feature = "cpu-faer")]
                            {
                                linalg::faer::lu_factor_solve_batched_in_place::<$scalar>(
                                    ctx, op, n, nrhs, packed_lu, pivots, output,
                                )
                            }
                            #[cfg(not(feature = "cpu-faer"))]
                            {
                                Err(unsupported_provider(op, CpuBackendKind::Faer))
                            }
                        }
                        CpuLinalgProvider::Blas => {
                            #[cfg(feature = "cpu-blas")]
                            {
                                linalg::blas::lu_factor_solve_batched_in_place::<$scalar>(
                                    op, n, nrhs, packed_lu, pivots, output,
                                )
                            }
                            #[cfg(not(feature = "cpu-blas"))]
                            {
                                Err(unsupported_provider(op, CpuBackendKind::Blas))
                            }
                        }
                    }
                }
            }
        )*
    };
}

impl_cpu_packed_lu!(f32, f64, Complex32, Complex64);

/// A pooled compact copy of `source` carrying `template`'s placement.
fn pooled_output<T: CpuPackedLu>(
    buffers: &mut BufferPool,
    source: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<T>> {
    let data = source.host_data()?;
    let mut output = buffers.acquire_with_capacity::<T>(data.len());
    output.extend_from_slice(data);
    Ok(output)
}

fn tensor_like<T: TensorScalar, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(template.placement().clone());
    Ok(tensor)
}

/// The `(n, nrhs)` of an RHS `b` against square batched `a` storage.
///
/// A rank-1 `b` or a batched vector `[n, batch...]` is solved as one column;
/// its compact layout is identical to `[n, 1, batch...]`, so no reshape is
/// needed and the output keeps `b`'s shape.
pub(super) fn rhs_matrix_shape(a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
    batched_vector_rhs_shape_of(a_shape, b_shape).unwrap_or_else(|| b_shape.to_vec())
}

/// Solve from packed factors with one batched provider kernel call.
///
/// The caller has already validated the factor, pivot, and RHS shapes with
/// `validate_lu_solve_prepared_shapes` against [`rhs_matrix_shape`], and the
/// `U` diagonal with `validate_nonsingular_u`.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` for an out-of-range pivot and
/// `Error::Internal` if the precondition above does not hold.
#[allow(clippy::too_many_arguments)]
pub(super) fn lu_solve_prepared_typed<T: CpuPackedLu>(
    provider: CpuLinalgProvider,
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a_shape: &[usize],
    packed_lu: &TypedTensor<T>,
    pivots: &TypedTensor<i32>,
    b: &TypedTensor<T>,
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "lu_solve_prepared";
    let rhs_shape = rhs_matrix_shape(a_shape, b.shape());
    let (Some(&n), Some(&nrhs)) = (rhs_shape.first(), rhs_shape.get(1)) else {
        return Err(Error::Internal(format!(
            "{OP}: RHS shape {rhs_shape:?} reached the kernel without validation"
        )));
    };
    let mut output = pooled_output(buffers, b)?;
    T::prepared_solve(
        provider,
        ctx,
        OP,
        n,
        nrhs,
        packed_lu.host_data()?,
        pivots.host_data()?,
        &mut output,
        transpose_a,
        conjugate_a,
    )?;
    tensor_like(b.shape().to_vec(), output, b)
}

/// Factor `A` and solve `A X = B` in one batched provider kernel call.
///
/// Returns `(X, packed_lu, pivots)` in the `lu_factor` storage convention.
///
/// # Errors
///
/// Returns `Error::ShapeMismatch` or `Error::InvalidArgument` for a
/// non-square `A`, mismatched RHS rows or batch shape, and
/// `Error::Extension` with `crate::Error::Singular` for an exactly singular
/// matrix when the RHS is nonempty.
pub(super) fn lu_factor_solve_typed<T: CpuPackedLu>(
    provider: CpuLinalgProvider,
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>, TypedTensor<i32>)> {
    const OP: &str = "lu_factor_solve";
    let n = square_matrix_dim(OP, a.shape())?;
    let rhs_shape = rhs_matrix_shape(a.shape(), b.shape());
    if rhs_shape.len() < 2 {
        return Err(Error::rank_mismatch(OP, 2, rhs_shape.len()));
    }
    if rhs_shape[0] != n {
        return Err(Error::invalid_argument(
            OP,
            "rhs rows",
            format!("expected {n}, got {}", rhs_shape[0]),
        ));
    }
    if rhs_shape[2..] != a.shape()[2..] {
        return Err(Error::shape_mismatch(
            OP,
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }
    let nrhs = rhs_shape[1];
    let mut pivot_shape = Vec::with_capacity(a.shape().len() - 1);
    pivot_shape.push(n);
    pivot_shape.extend_from_slice(&a.shape()[2..]);
    let pivot_len = checked_product(OP, "pivot shape", &pivot_shape)?;

    let mut lu_data = pooled_output(buffers, a)?;
    let mut pivot_data = <i32 as PoolScalar>::pool_acquire_zeroed(buffers, pivot_len);
    let mut output = pooled_output(buffers, b)?;
    if !has_zero_dim(a.shape()) {
        T::factor_solve(
            provider,
            ctx,
            OP,
            n,
            nrhs,
            &mut lu_data,
            &mut pivot_data,
            &mut output,
        )?;
    }
    Ok((
        tensor_like(b.shape().to_vec(), output, b)?,
        tensor_like(a.shape().to_vec(), lu_data, a)?,
        tensor_like(pivot_shape, pivot_data, a)?,
    ))
}

/// Dispatch a dtype-erased prepared solve to [`lu_solve_prepared_typed`].
///
/// # Errors
///
/// Returns `Error::DTypeMismatch` or an unsupported-dtype error when the
/// operands do not share one supported linalg dtype, and the typed errors of
/// [`lu_solve_prepared_typed`].
#[allow(clippy::too_many_arguments)]
pub(super) fn lu_solve_prepared_entered(
    provider: CpuLinalgProvider,
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &Tensor,
    packed_lu: &Tensor,
    pivots: &Tensor,
    b: &Tensor,
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<Tensor> {
    const OP: &str = "lu_solve_prepared";
    let inconsistent = || {
        Error::Internal(format!(
            "{OP}: packed LU, pivots, and rhs dtypes are inconsistent"
        ))
    };
    let pivots = pivots.as_typed::<i32>().ok_or_else(inconsistent)?;
    macro_rules! solve_as {
        ($scalar:ty) => {{
            let lu = packed_lu.as_typed::<$scalar>().ok_or_else(inconsistent)?;
            let rhs = b.as_typed::<$scalar>().ok_or_else(inconsistent)?;
            lu_solve_prepared_typed(
                provider,
                ctx,
                buffers,
                a.shape(),
                lu,
                pivots,
                rhs,
                transpose_a,
                conjugate_a,
            )
            .map(Tensor::from_typed::<$scalar>)
        }};
    }
    match b.dtype() {
        tenferro_tensor::DType::F32 => solve_as!(f32),
        tenferro_tensor::DType::F64 => solve_as!(f64),
        tenferro_tensor::DType::C32 => solve_as!(Complex32),
        tenferro_tensor::DType::C64 => solve_as!(Complex64),
        other => Err(crate::backend::unsupported_dtype(OP, other)),
    }
}

/// Dispatch a dtype-erased fused factor-solve to [`lu_factor_solve_typed`].
///
/// # Errors
///
/// Returns an unsupported-dtype error for a non-linalg dtype, and the typed
/// errors of [`lu_factor_solve_typed`].
pub(super) fn lu_factor_solve_entered(
    provider: CpuLinalgProvider,
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &Tensor,
    b: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    const OP: &str = "lu_factor_solve";
    let inconsistent = || Error::dtype_mismatch(OP, a.dtype(), b.dtype());
    macro_rules! factor_solve_as {
        ($scalar:ty) => {{
            let a = a.as_typed::<$scalar>().ok_or_else(inconsistent)?;
            let b = b.as_typed::<$scalar>().ok_or_else(inconsistent)?;
            let (x, lu, pivots) = lu_factor_solve_typed(provider, ctx, buffers, a, b)?;
            Ok(vec![
                Tensor::from_typed::<$scalar>(x),
                Tensor::from_typed::<$scalar>(lu),
                Tensor::from_typed::<i32>(pivots),
            ])
        }};
    }
    match a.dtype() {
        tenferro_tensor::DType::F32 => factor_solve_as!(f32),
        tenferro_tensor::DType::F64 => factor_solve_as!(f64),
        tenferro_tensor::DType::C32 => factor_solve_as!(Complex32),
        tenferro_tensor::DType::C64 => factor_solve_as!(Complex64),
        other => Err(crate::backend::unsupported_dtype(OP, other)),
    }
}
