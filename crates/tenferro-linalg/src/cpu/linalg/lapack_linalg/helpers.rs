//! Shared host helpers of the LAPACK route and its per-scalar provider dispatch.
//!
//! Every numerical kernel, its FFI marshalling and the batch loop live in the extracted
//! `tlinalg-blas` crate. This module keeps tenferro's shape validation and tensor construction
//! helpers, and [`LapackLinalg`]: one trait per scalar that forwards each family to the provider
//! with the session pool as its workspace.

use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::{DType, Tensor, TypedTensor};
use tlinalg_blas::{Op, Result as BlasResult};

use crate::cpu::tlinalg_blas::workspace;

#[cfg(test)]
#[path = "helpers/tests.rs"]
mod tests;

/// The LAPACK route's scalar vocabulary: host facts plus one forwarding method per family.
///
/// The provider is generic over its own scalar and workspace traits; naming those bounds from a
/// generic host function would thread several where-clauses through every caller, so each method
/// is implemented once per concrete scalar, where the bounds are simply true.
pub(crate) trait LapackLinalg:
    tlinalg_blas::LapackScalar + PoolScalar + Default + PartialEq + std::ops::Mul<Output = Self>
{
    /// The real scalar of singular values and Hermitian eigenvalues.
    type RealScalar: PoolScalar + Default;
    /// The complex scalar of a general eigendecomposition.
    type ComplexScalar: PoolScalar;

    fn unit() -> Self;
    fn r_phase(diagonal: Self) -> Self;
    fn q_phase(diagonal: Self) -> Self;
    /// `|x|`, as the rank decision compares it.
    fn rank_magnitude(self) -> f64;
    /// Whether every component is finite (the rank-revealing QR input screen).
    fn is_finite_value(self) -> bool;
    /// Real values in this scalar type, reusing the buffer when the scalar is real.
    fn values_as_scalar(buffers: &mut BufferPool, values: Vec<Self::RealScalar>) -> Vec<Self>;
    /// Wrap a general-eigendecomposition output as an erased tensor.
    fn wrap_complex(tensor: TypedTensor<Self::ComplexScalar>) -> Tensor;
    /// `C = A B` for the compact Householder fold of `from_factors_2d` (host composition).
    fn gemm_2d(
        a: &[Self],
        a_rows: usize,
        a_cols: usize,
        b: &[Self],
        b_cols: usize,
        c: &mut [Self],
    ) -> tenferro_tensor::Result<()>;

    fn cholesky(
        pool: &mut BufferPool,
        a: RawStridedRef<'_, Self>,
        out: &mut Vec<Self>,
    ) -> BlasResult<()>;
    fn triangular_solve(
        pool: &mut BufferPool,
        options: tlinalg_blas::triangular_solve::TriangularSolveOptions,
        a: RawStridedRef<'_, Self>,
        b: RawStridedRef<'_, Self>,
        out: &mut Vec<Self>,
    ) -> BlasResult<()>;
    fn lu(
        pool: &mut BufferPool,
        a: RawStridedRef<'_, Self>,
        outputs: tlinalg_blas::lu::LuOutputs<'_, Self>,
    ) -> BlasResult<()>;
    fn solve(
        pool: &mut BufferPool,
        transpose_a: bool,
        a: RawStridedRef<'_, Self>,
        b: RawStridedRef<'_, Self>,
        out: &mut Vec<Self>,
    ) -> BlasResult<()>;
    fn solve_into(
        pool: &mut BufferPool,
        transpose_a: bool,
        a: RawStridedRef<'_, Self>,
        b: RawStridedRef<'_, Self>,
        out: RawStridedMut<'_, Self>,
    ) -> BlasResult<()>;
    fn householder_factor(
        pool: &mut BufferPool,
        rows: usize,
        cols: usize,
        data: &mut [Self],
        tau: &mut Vec<Self>,
    ) -> BlasResult<()>;
    // INVARIANT: the argument list mirrors the provider's reflector ABI one-to-one.
    #[allow(clippy::too_many_arguments)]
    fn apply_reflectors(
        pool: &mut BufferPool,
        m: usize,
        a_cols: usize,
        p: usize,
        k: usize,
        transpose: bool,
        a: &[Self],
        tau: &[Self],
        c: &mut [Self],
    ) -> BlasResult<()>;
    fn qr(
        pool: &mut BufferPool,
        a: RawStridedRef<'_, Self>,
        q: &mut Vec<Self>,
        r: &mut Vec<Self>,
    ) -> BlasResult<()>;
    fn rank_revealing_qr(
        pool: &mut BufferPool,
        a: RawStridedRef<'_, Self>,
        outputs: tlinalg_blas::qr::RankRevealingQrOutputs<'_, Self>,
    ) -> BlasResult<()>;
    fn eigh(
        pool: &mut BufferPool,
        op: Op,
        a: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::RealScalar>,
        vectors: Option<&mut Vec<Self>>,
    ) -> BlasResult<()>;
    fn svd(
        pool: &mut BufferPool,
        op: Op,
        mode: tlinalg_blas::svd::SvdMode,
        a: RawStridedRef<'_, Self>,
        outputs: tlinalg_blas::svd::SvdOutputs<'_, Self, Self::RealScalar>,
    ) -> BlasResult<()>;
    fn eig(
        pool: &mut BufferPool,
        op: Op,
        a: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::ComplexScalar>,
        vectors: Option<&mut Vec<Self::ComplexScalar>>,
    ) -> BlasResult<()>;
}

/// The forwarding methods, identical for every scalar.
macro_rules! forward_lapack_families {
    ($real:ty, $complex:ty) => {
        fn cholesky(
            pool: &mut BufferPool,
            a: RawStridedRef<'_, Self>,
            out: &mut Vec<Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::cholesky::cholesky(Op::Cholesky, a, out, &mut workspace(pool))
        }

        fn triangular_solve(
            pool: &mut BufferPool,
            options: tlinalg_blas::triangular_solve::TriangularSolveOptions,
            a: RawStridedRef<'_, Self>,
            b: RawStridedRef<'_, Self>,
            out: &mut Vec<Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::triangular_solve::triangular_solve(
                Op::TriangularSolve,
                options,
                a,
                b,
                out,
                &mut workspace(pool),
            )
        }

        fn lu(
            pool: &mut BufferPool,
            a: RawStridedRef<'_, Self>,
            outputs: tlinalg_blas::lu::LuOutputs<'_, Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::lu::lu(Op::Lu, a, outputs, &mut workspace(pool))
        }

        fn solve(
            pool: &mut BufferPool,
            transpose_a: bool,
            a: RawStridedRef<'_, Self>,
            b: RawStridedRef<'_, Self>,
            out: &mut Vec<Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::solve::solve(Op::Solve, transpose_a, a, b, out, &mut workspace(pool))
        }

        fn solve_into(
            pool: &mut BufferPool,
            transpose_a: bool,
            a: RawStridedRef<'_, Self>,
            b: RawStridedRef<'_, Self>,
            out: RawStridedMut<'_, Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::solve::solve_into(Op::Solve, transpose_a, a, b, out, &mut workspace(pool))
        }

        fn householder_factor(
            pool: &mut BufferPool,
            rows: usize,
            cols: usize,
            data: &mut [Self],
            tau: &mut Vec<Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::householder::factor(
                Op::HouseholderFactor,
                rows,
                cols,
                data,
                tau,
                &mut workspace(pool),
            )
        }

        fn apply_reflectors(
            pool: &mut BufferPool,
            m: usize,
            a_cols: usize,
            p: usize,
            k: usize,
            transpose: bool,
            a: &[Self],
            tau: &[Self],
            c: &mut [Self],
        ) -> BlasResult<()> {
            tlinalg_blas::householder::apply_reflectors(
                Op::HouseholderApply,
                m,
                a_cols,
                p,
                k,
                transpose,
                a,
                tau,
                c,
                &mut workspace(pool),
            )
        }

        fn qr(
            pool: &mut BufferPool,
            a: RawStridedRef<'_, Self>,
            q: &mut Vec<Self>,
            r: &mut Vec<Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::qr::qr(Op::Qr, a, q, r, &mut workspace(pool))
        }

        fn rank_revealing_qr(
            pool: &mut BufferPool,
            a: RawStridedRef<'_, Self>,
            outputs: tlinalg_blas::qr::RankRevealingQrOutputs<'_, Self>,
        ) -> BlasResult<()> {
            tlinalg_blas::qr::rank_revealing_qr(
                Op::RankRevealingQr,
                a,
                outputs,
                &mut workspace(pool),
            )
        }

        fn eigh(
            pool: &mut BufferPool,
            op: Op,
            a: RawStridedRef<'_, Self>,
            values: &mut Vec<$real>,
            vectors: Option<&mut Vec<Self>>,
        ) -> BlasResult<()> {
            tlinalg_blas::eigh::eigh(op, a, values, vectors, &mut workspace(pool))
        }

        fn svd(
            pool: &mut BufferPool,
            op: Op,
            mode: tlinalg_blas::svd::SvdMode,
            a: RawStridedRef<'_, Self>,
            outputs: tlinalg_blas::svd::SvdOutputs<'_, Self, $real>,
        ) -> BlasResult<()> {
            tlinalg_blas::svd::svd(op, mode, a, outputs, &mut workspace(pool))
        }

        fn eig(
            pool: &mut BufferPool,
            op: Op,
            a: RawStridedRef<'_, Self>,
            values: &mut Vec<$complex>,
            vectors: Option<&mut Vec<$complex>>,
        ) -> BlasResult<()> {
            tlinalg_blas::eig::eig(op, a, values, vectors, &mut workspace(pool))
        }
    };
}

/// Validate the three compact operands of a host GEMM.
fn validate_gemm_operands(
    a_len: usize,
    b_len: usize,
    c_len: usize,
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> tenferro_tensor::Result<(i32, i32, i32)> {
    validate_buffer_len(
        "from_factors_2d",
        "T",
        a_len,
        checked_product("from_factors_2d", "T", &[a_rows, a_cols])?,
    )?;
    validate_buffer_len(
        "from_factors_2d",
        "R",
        b_len,
        checked_product("from_factors_2d", "R", &[a_cols, b_cols])?,
    )?;
    validate_buffer_len(
        "from_factors_2d",
        "folded R",
        c_len,
        checked_product("from_factors_2d", "folded R", &[a_rows, b_cols])?,
    )?;
    Ok((
        dim_i32(a_rows, "from_factors_2d")?,
        dim_i32(b_cols, "from_factors_2d")?,
        dim_i32(a_cols, "from_factors_2d")?,
    ))
}

macro_rules! impl_real_lapack {
    ($scalar:ty, $complex:ty, $gemm:path) => {
        impl LapackLinalg for $scalar {
            type RealScalar = $scalar;
            type ComplexScalar = $complex;

            fn unit() -> Self {
                1.0
            }

            fn r_phase(diagonal: Self) -> Self {
                if diagonal < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }

            fn q_phase(diagonal: Self) -> Self {
                Self::r_phase(diagonal)
            }

            fn rank_magnitude(self) -> f64 {
                tlinalg_blas::magnitude(self)
            }

            fn is_finite_value(self) -> bool {
                self.is_finite()
            }

            fn values_as_scalar(_buffers: &mut BufferPool, values: Vec<Self>) -> Vec<Self> {
                values
            }

            fn wrap_complex(tensor: TypedTensor<$complex>) -> Tensor {
                Tensor::from_typed::<$complex>(tensor)
            }

            fn gemm_2d(
                a: &[Self],
                a_rows: usize,
                a_cols: usize,
                b: &[Self],
                b_cols: usize,
                c: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let (m, n, k) =
                    validate_gemm_operands(a.len(), b.len(), c.len(), a_rows, a_cols, b_cols)?;
                // SAFETY: the checked slice lengths cover m*k, k*n, and m*n column-major
                // elements; dimensions and leading dimensions fit LP64 i32.
                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        1.0,
                        a.as_ptr(),
                        m,
                        b.as_ptr(),
                        k,
                        0.0,
                        c.as_mut_ptr(),
                        m,
                    );
                }
                Ok(())
            }

            forward_lapack_families!($scalar, $complex);
        }
    };
}

macro_rules! impl_complex_lapack {
    ($complex:ty, $real:ty, $gemm:path) => {
        impl LapackLinalg for $complex {
            type RealScalar = $real;
            type ComplexScalar = $complex;

            fn unit() -> Self {
                <$complex>::new(1.0, 0.0)
            }

            fn r_phase(diagonal: Self) -> Self {
                let norm = diagonal.norm();
                if norm == 0.0 {
                    Self::unit()
                } else {
                    diagonal.conj() / norm
                }
            }

            fn q_phase(diagonal: Self) -> Self {
                let norm = diagonal.norm();
                if norm == 0.0 {
                    Self::unit()
                } else {
                    diagonal / norm
                }
            }

            fn rank_magnitude(self) -> f64 {
                tlinalg_blas::magnitude(self)
            }

            fn is_finite_value(self) -> bool {
                self.re.is_finite() && self.im.is_finite()
            }

            fn values_as_scalar(buffers: &mut BufferPool, values: Vec<$real>) -> Vec<Self> {
                let mut converted = buffers.acquire_with_capacity::<$complex>(values.len());
                converted.extend(values.iter().map(|&value| <$complex>::new(value, 0.0)));
                <$real as PoolScalar>::pool_release(buffers, values);
                converted
            }

            fn wrap_complex(tensor: TypedTensor<$complex>) -> Tensor {
                Tensor::from_typed::<$complex>(tensor)
            }

            fn gemm_2d(
                a: &[Self],
                a_rows: usize,
                a_cols: usize,
                b: &[Self],
                b_cols: usize,
                c: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let (m, n, k) =
                    validate_gemm_operands(a.len(), b.len(), c.len(), a_rows, a_cols, b_cols)?;
                let alpha = <$complex>::new(1.0, 0.0);
                let beta = <$complex>::new(0.0, 0.0);
                // SAFETY: the checked slice lengths cover m*k, k*n, and m*n column-major
                // elements; dimensions and leading dimensions fit LP64 i32, and `alpha`/`beta`
                // live for the call.
                unsafe {
                    $gemm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        m,
                        n,
                        k,
                        (&alpha as *const $complex).cast(),
                        a.as_ptr().cast(),
                        m,
                        b.as_ptr().cast(),
                        k,
                        (&beta as *const $complex).cast(),
                        c.as_mut_ptr().cast(),
                        m,
                    );
                }
                Ok(())
            }

            forward_lapack_families!($real, $complex);
        }
    };
}

impl_real_lapack!(f32, Complex32, cblas_sys::cblas_sgemm);
impl_real_lapack!(f64, Complex64, cblas_sys::cblas_dgemm);
impl_complex_lapack!(Complex32, f32, cblas_sys::cblas_cgemm);
impl_complex_lapack!(Complex64, f64, cblas_sys::cblas_zgemm);

/// Rebuild tenferro's error from a provider failure.
pub(crate) fn provider(op: Op) -> impl Fn(tlinalg_blas::Error) -> tenferro_tensor::Error {
    move |error| crate::cpu::tlinalg_error::map_blas_error(op, error)
}

pub(crate) fn validate_buffer_len(
    op: &'static str,
    role: &'static str,
    actual: usize,
    expected: usize,
) -> tenferro_tensor::Result<()> {
    if actual != expected {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            role,
            format!("expected {expected} elements, got {actual}"),
        ));
    }
    Ok(())
}

pub(crate) fn matrix_dims<T>(
    input: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    if input.shape().len() != 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            2,
            input.shape().len(),
        ));
    }
    Ok((input.shape()[0], input.shape()[1]))
}

pub(crate) fn tensor_from_vec_with_template<T: Clone + tenferro_tensor::TensorScalar, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(template.placement().clone());
    Ok(tensor)
}

pub(crate) fn split_core_and_batch_result<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &'static str,
) -> tenferro_tensor::Result<(&'a [usize], &'a [usize])> {
    if input.shape().len() < core_rank {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            core_rank,
            input.shape().len(),
        ));
    }
    Ok(input.shape().split_at(core_rank))
}

pub(crate) fn matrix_core_and_batch_result<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize, &'a [usize])> {
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, op)?;
    Ok((matrix_shape[0], matrix_shape[1], batch_shape))
}

pub(crate) fn square_core_and_batch_result<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, &'a [usize])> {
    let (rows, cols, batch_shape) = matrix_core_and_batch_result(input, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok((rows, batch_shape))
}

pub(crate) fn batch_element_count(
    op: &'static str,
    batch_shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    batch_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            tenferro_tensor::Error::validation(
                op,
                tenferro_tensor::ValidationError::IntegerOverflow,
            )
        })
    })
}

pub(crate) fn checked_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                op,
                "shape",
                format!("{role} element count overflow"),
            )
        })
    })
}

pub(crate) fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

pub(crate) fn matrix_with_batch_shape(
    rows: usize,
    cols: usize,
    batch_shape: &[usize],
) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch_shape);
    shape
}

pub(crate) fn vector_with_batch_shape(len: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch_shape);
    shape
}

pub(crate) fn dim_i32(value: usize, op: &'static str) -> tenferro_tensor::Result<i32> {
    i32::try_from(value).map_err(|_| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "dimension",
            format!("dimension {value} exceeds LAPACK i32 range"),
        )
    })
}

/// A pooled output buffer for `shape` elements; the provider clears and fills it.
pub(crate) fn pooled_output<T: PoolScalar>(
    buffers: &mut BufferPool,
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<Vec<T>> {
    Ok(buffers.acquire_with_capacity::<T>(checked_product(op, role, shape)?))
}

/// Return a buffer to the pool for the next call to reuse.
pub(crate) fn release_scratch<T: PoolScalar>(buffers: &mut BufferPool, data: Vec<T>) {
    <T as PoolScalar>::pool_release(buffers, data);
}

/// Empty `(values, vectors)` of a general eigendecomposition with a zero core dimension, tagged
/// with the complex dtype the nonempty route returns.
pub(crate) fn zero_dim_eig_outputs(
    input: &Tensor,
    op: &'static str,
    vectors: bool,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(op, 2, shape.len()));
    }
    let n = shape[0];
    if shape[1] != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![shape[1]],
        ));
    }
    let batch_shape = &shape[2..];
    let mut shapes = vec![vector_with_batch_shape(n, batch_shape)];
    if vectors {
        shapes.push(matrix_with_batch_shape(n, n, batch_shape));
    }
    match input.dtype() {
        DType::F32 | DType::C32 => shapes
            .into_iter()
            .map(|shape| {
                Ok(Tensor::from_typed::<Complex32>(
                    TypedTensor::from_vec_col_major(shape, Vec::new())?,
                ))
            })
            .collect(),
        DType::F64 | DType::C64 => shapes
            .into_iter()
            .map(|shape| {
                Ok(Tensor::from_typed::<Complex64>(
                    TypedTensor::from_vec_col_major(shape, Vec::new())?,
                ))
            })
            .collect(),
        _ => Err(super::unsupported_dtype(op, input.dtype())),
    }
}
