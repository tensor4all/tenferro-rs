//! The faer route's host side: tensors in, one batched `tlinalg` call, tensors out.
//!
//! Every numerical kernel and the batch loop live in the extracted `tlinalg` crate. What stays here
//! is what the host owns: shape and rank validation with tenferro's error shapes, the empty and
//! zero-batch results, pooled output allocation, borrowed strided descriptors of host tensors and
//! views, tensor construction, and the host compositions built on the extracted primitives (the
//! rank decision of rank-revealing QR and the compact Householder QR state operations).

use faer::{MatMut, MatRef};
use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};

use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar, PooledUninitOutput};
use tenferro_cpu::CpuExecutionContext;
use tenferro_tensor::{
    DType, Tensor, TensorView, TypedTensor, TypedTensorView, TypedTensorViewMut,
};
use tlinalg::Op;

use super::{finish_lease, pooled_copy, pooled_output, pooled_zeroed, raw_view};
use crate::cpu::tlinalg::{exec_from, PooledWorkspace};
use crate::cpu::tlinalg_error::map_error;

/// The host-side scalar vocabulary of the faer route.
///
/// The numerical work is `tlinalg`'s; this trait keeps only the scalar facts the host's own
/// compositions use (gauges, the rank screen, the compact QR fold) plus the two families whose
/// output scalar differs from the input scalar, which are dispatched per concrete type because the
/// provider's real-scalar projection is not nameable from here.
pub(crate) trait FaerLinalg:
    tlinalg::FaerScalar + PoolScalar + Default + PartialEq + std::ops::Mul<Output = Self>
{
    /// The real scalar of singular values and Hermitian eigenvalues.
    type RealScalar: PoolScalar + Default;

    fn parity_one() -> Self;
    fn is_finite(self) -> bool;
    fn one() -> Self;
    fn r_phase(diagonal: Self) -> Self;
    fn q_phase(diagonal: Self) -> Self;
    /// `C = A B` for the compact Householder fold of `from_factors_2d`.
    fn gemm_data(
        ctx: &CpuExecutionContext<'_>,
        a: &[Self],
        a_rows: usize,
        a_cols: usize,
        b: &[Self],
        b_cols: usize,
        c: &mut [Self],
    ) -> tenferro_tensor::Result<()>;
    /// Batched singular values in the real scalar.
    fn svd_values_into(
        input: RawStridedRef<'_, Self>,
        values: &mut [core::mem::MaybeUninit<Self::RealScalar>],
        workspace: &mut [core::mem::MaybeUninit<u8>],
        exec: Exec<'_>,
    ) -> tlinalg::Result<()>;
    /// Batched Hermitian eigenvalues in the real scalar.
    fn eigh_values_into(
        input: RawStridedRef<'_, Self>,
        values: &mut [core::mem::MaybeUninit<Self::RealScalar>],
        workspace: &mut [core::mem::MaybeUninit<u8>],
        exec: Exec<'_>,
    ) -> tlinalg::Result<()>;
}

macro_rules! impl_faer_linalg {
    (
        $scalar:ty,
        $real:ty,
        $one:expr,
        $faer_one:expr,
        |$d:ident| $r_phase:expr,
        |$q:ident| $q_phase:expr,
        |$f:ident| $is_finite:expr,
        $to_faer:ident,
        $to_faer_mut:ident
    ) => {
        impl FaerLinalg for $scalar {
            type RealScalar = $real;

            fn parity_one() -> Self {
                $one
            }

            fn is_finite(self) -> bool {
                let $f = self;
                $is_finite
            }

            fn one() -> Self {
                $one
            }

            fn r_phase($d: Self) -> Self {
                $r_phase
            }

            fn q_phase($q: Self) -> Self {
                $q_phase
            }

            fn gemm_data(
                ctx: &CpuExecutionContext<'_>,
                a: &[Self],
                a_rows: usize,
                a_cols: usize,
                b: &[Self],
                b_cols: usize,
                c: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                if a.len() != checked_product("from_factors_2d", "T", &[a_rows, a_cols])?
                    || b.len() != checked_product("from_factors_2d", "R", &[a_cols, b_cols])?
                    || c.len() != checked_product("from_factors_2d", "folded R", &[a_rows, b_cols])?
                {
                    return Err(invalid_config(
                        "from_factors_2d",
                        "matrix: input buffer length does not match dimensions",
                    ));
                }
                if a_rows == 0 || a_cols == 0 || b_cols == 0 {
                    return Ok(());
                }
                let lhs = MatRef::from_column_major_slice($to_faer(a), a_rows, a_cols);
                let rhs = MatRef::from_column_major_slice($to_faer(b), a_cols, b_cols);
                let out = MatMut::from_column_major_slice_mut($to_faer_mut(c), a_rows, b_cols);
                // The shared work model decides the width before any pool is
                // entered: a small product stays on the calling thread (#2206).
                let muladds = (a_rows as f64) * (a_cols as f64) * (b_cols as f64);
                let width = ctx
                    .contraction_exec()?
                    .batch_decision(&tcontract_exec::BatchQuery::new(1, muladds))
                    .item_threads;
                ctx.with_faer_width(width, |par| {
                    faer::linalg::matmul::matmul(
                        out,
                        faer::Accum::Replace,
                        lhs,
                        rhs,
                        $faer_one,
                        par,
                    );
                });
                Ok(())
            }

            fn svd_values_into(
                input: RawStridedRef<'_, Self>,
                values: &mut [core::mem::MaybeUninit<$real>],
                workspace: &mut [core::mem::MaybeUninit<u8>],
                exec: Exec<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::svd::svd_values_into(Op::SvdValues, input, values, workspace, exec)
            }

            fn eigh_values_into(
                input: RawStridedRef<'_, Self>,
                values: &mut [core::mem::MaybeUninit<$real>],
                workspace: &mut [core::mem::MaybeUninit<u8>],
                exec: Exec<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::eigh::eigh_values_into(Op::EighValues, input, values, workspace, exec)
            }
        }
    };
}

fn same_slice<T>(data: &[T]) -> &[T] {
    data
}

fn same_slice_mut<T>(data: &mut [T]) -> &mut [T] {
    data
}

macro_rules! impl_complex_faer_casts {
    ($to_faer_slice:ident, $to_faer_slice_mut:ident, $complex:ty, $faer_complex:ty) => {
        const _: () = {
            assert!(std::mem::size_of::<$complex>() == std::mem::size_of::<$faer_complex>());
            assert!(std::mem::align_of::<$complex>() == std::mem::align_of::<$faer_complex>());
            assert!(std::mem::offset_of!($complex, re) == std::mem::offset_of!($faer_complex, re));
            assert!(std::mem::offset_of!($complex, im) == std::mem::offset_of!($faer_complex, im));
        };

        fn $to_faer_slice(data: &[$complex]) -> &[$faer_complex] {
            // SAFETY: the const assertions above prove identical size, alignment and field
            // offsets, so both types represent one complex scalar over the same real type.
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<$faer_complex>(), data.len()) }
        }

        fn $to_faer_slice_mut(data: &mut [$complex]) -> &mut [$faer_complex] {
            // SAFETY: as above; the mutable receiver guarantees exclusive access.
            unsafe {
                std::slice::from_raw_parts_mut(
                    data.as_mut_ptr().cast::<$faer_complex>(),
                    data.len(),
                )
            }
        }
    };
}

impl_complex_faer_casts!(
    complex32_to_faer_slice,
    complex32_to_faer_slice_mut,
    Complex32,
    faer::c32
);
impl_complex_faer_casts!(
    complex64_to_faer_slice,
    complex64_to_faer_slice_mut,
    Complex64,
    faer::c64
);

impl_faer_linalg!(
    f32,
    f32,
    1.0,
    1.0,
    |d| if d < 0.0 { -1.0 } else { 1.0 },
    |d| if d < 0.0 { -1.0 } else { 1.0 },
    |value| value.is_finite(),
    same_slice,
    same_slice_mut
);
impl_faer_linalg!(
    f64,
    f64,
    1.0,
    1.0,
    |d| if d < 0.0 { -1.0 } else { 1.0 },
    |d| if d < 0.0 { -1.0 } else { 1.0 },
    |value| value.is_finite(),
    same_slice,
    same_slice_mut
);
impl_faer_linalg!(
    Complex32,
    f32,
    Complex32::new(1.0, 0.0),
    faer::c32::new(1.0, 0.0),
    |d| {
        let norm = d.norm();
        if norm == 0.0 {
            Complex32::new(1.0, 0.0)
        } else {
            d.conj() / norm
        }
    },
    |d| {
        let norm = d.norm();
        if norm == 0.0 {
            Complex32::new(1.0, 0.0)
        } else {
            d / norm
        }
    },
    |value| value.re.is_finite() && value.im.is_finite(),
    complex32_to_faer_slice,
    complex32_to_faer_slice_mut
);
impl_faer_linalg!(
    Complex64,
    f64,
    Complex64::new(1.0, 0.0),
    faer::c64::new(1.0, 0.0),
    |d| {
        let norm = d.norm();
        if norm == 0.0 {
            Complex64::new(1.0, 0.0)
        } else {
            d.conj() / norm
        }
    },
    |d| {
        let norm = d.norm();
        if norm == 0.0 {
            Complex64::new(1.0, 0.0)
        } else {
            d / norm
        }
    },
    |value| value.re.is_finite() && value.im.is_finite(),
    complex64_to_faer_slice,
    complex64_to_faer_slice_mut
);

/// The general eigendecomposition, whose outputs are in the complex counterpart of the input.
pub(crate) trait FaerEig: FaerLinalg {
    type ComplexScalar: PoolScalar;

    fn eig_batch(
        input: RawStridedRef<'_, Self>,
        values: &mut [core::mem::MaybeUninit<Self::ComplexScalar>],
        vectors: Option<&mut [core::mem::MaybeUninit<Self::ComplexScalar>]>,
        workspace: &mut [core::mem::MaybeUninit<u8>],
        exec: Exec<'_>,
    ) -> tlinalg::Result<()>;

    /// Put the provider's eigenvectors into the public contract's gauge.
    ///
    /// `eig` documents **unit 2-norm columns** - the vendored oracle records the
    /// same for PyTorch and tenferro - which the providers do not all guarantee,
    /// so the seam normalizes them here. A column the provider left non-finite (a
    /// defective, Jordan-like input can overflow inside its back-substitution) is
    /// reported as a numerical failure rather than returned as NaN (#2148).
    fn normalize_eig_vectors(
        vectors: &mut [Self::ComplexScalar],
        n: usize,
        batch_count: usize,
    ) -> tenferro_tensor::Result<()>;

    fn wrap(tensor: TypedTensor<Self::ComplexScalar>) -> Tensor;
}

macro_rules! impl_faer_eig {
    ($scalar:ty, $complex:ty, $real:ty) => {
        impl FaerEig for $scalar {
            type ComplexScalar = $complex;

            fn eig_batch(
                input: RawStridedRef<'_, Self>,
                values: &mut [core::mem::MaybeUninit<$complex>],
                vectors: Option<&mut [core::mem::MaybeUninit<$complex>]>,
                workspace: &mut [core::mem::MaybeUninit<u8>],
                exec: Exec<'_>,
            ) -> tlinalg::Result<()> {
                match vectors {
                    Some(vectors) => {
                        tlinalg::eig::eig_into(Op::Eig, input, values, vectors, workspace, exec)
                    }
                    None => {
                        tlinalg::eig::eig_values_into(Op::EigValues, input, values, workspace, exec)
                    }
                }
            }

            fn wrap(tensor: TypedTensor<$complex>) -> Tensor {
                Tensor::from_typed::<$complex>(tensor)
            }

            fn normalize_eig_vectors(
                vectors: &mut [$complex],
                n: usize,
                batch_count: usize,
            ) -> tenferro_tensor::Result<()> {
                // INVARIANT: `n >= 1` for a non-empty square input, so each chunk
                // below is a whole column of the column-major [n, n] block.
                for block in vectors.chunks_exact_mut(n * n).take(batch_count) {
                    for column in block.chunks_exact_mut(n) {
                        // Scale before summing squares. `norm_sqr` is evaluated in the
                        // operand dtype (`f32` for the `Complex32` instantiation), so a
                        // finite column of large magnitude squares to `inf` and one of
                        // tiny magnitude squares to `0`, and both were rejected as
                        // `NonFinite` even though the column is normalizable. Scaling by
                        // the largest component first keeps every squared term in [0, 1],
                        // and the scaling itself is done in f64 so it cannot overflow.
                        let magnitude = column
                            .iter()
                            .map(|value| f64::from(value.re.abs().max(value.im.abs())))
                            .fold(0.0_f64, f64::max);
                        if magnitude == 0.0 {
                            return Err(crate::error::into_tensor_error(
                                "eig",
                                crate::Error::NonFinite {
                                    op: "eig",
                                    role: "eigenvector column",
                                },
                            ));
                        }
                        let norm = magnitude
                            * column
                                .iter()
                                .map(|value| {
                                    let re = f64::from(value.re) / magnitude;
                                    let im = f64::from(value.im) / magnitude;
                                    re * re + im * im
                                })
                                .sum::<f64>()
                                .sqrt();
                        let scale = norm as $real;
                        if !norm.is_finite() || !scale.is_finite() || scale == 0.0 {
                            return Err(crate::error::into_tensor_error(
                                "eig",
                                crate::Error::NonFinite {
                                    op: "eig",
                                    role: "eigenvector column",
                                },
                            ));
                        }
                        for value in column.iter_mut() {
                            *value /= scale;
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

impl_faer_eig!(f32, Complex32, f32);
impl_faer_eig!(f64, Complex64, f64);
impl_faer_eig!(Complex32, Complex32, f32);
impl_faer_eig!(Complex64, Complex64, f64);

// ---------------------------------------------------------------------------------------------
// Shape and allocation helpers
// ---------------------------------------------------------------------------------------------

fn invalid_config(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op, "configuration", message)
}

fn checked_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| invalid_config(op, format!("{role} element count overflows usize")))
}

fn tensor_from_vec_with_template<T: PoolScalar>(
    shape: impl tenferro_tensor::IntoShapeVec,
    data: Vec<T>,
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(placement.clone());
    Ok(tensor)
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

/// `[rows, cols, batch...]` as an inline shape (no heap allocation up to `INLINE_RANK`).
fn matrix_with_batch_shape(
    rows: usize,
    cols: usize,
    batch_shape: &[usize],
) -> tenferro_tensor::ShapeVec {
    let mut shape = tenferro_tensor::ShapeVec::new();
    shape.push(rows);
    shape.push(cols);
    shape.extend_from_slice(batch_shape);
    shape
}

/// `[len, batch...]` as an inline shape (no heap allocation up to `INLINE_RANK`).
fn vector_with_batch_shape(len: usize, batch_shape: &[usize]) -> tenferro_tensor::ShapeVec {
    let mut shape = tenferro_tensor::ShapeVec::new();
    shape.push(len);
    shape.extend_from_slice(batch_shape);
    shape
}

/// `(rows, cols, batch shape)` of a rank-`2 + B` operand.
fn matrix_core_and_batch<'s>(
    shape: &'s [usize],
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize, &'s [usize])> {
    if shape.len() < 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(op, 2, shape.len()));
    }
    Ok((shape[0], shape[1], &shape[2..]))
}

/// `(n, batch shape)` of a batch of square matrices.
fn square_core_and_batch<'s>(
    shape: &'s [usize],
    op: &'static str,
) -> tenferro_tensor::Result<(usize, &'s [usize])> {
    let (rows, cols, batch_shape) = matrix_core_and_batch(shape, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok((rows, batch_shape))
}

/// The dims of an exactly rank-2 operand, as the borrowed view routes require.
fn matrix_dims_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    if view.shape().len() != 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            2,
            view.shape().len(),
        ));
    }
    Ok((view.shape()[0], view.shape()[1]))
}

fn square_matrix_dim_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    let (rows, cols) = matrix_dims_view(view, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok(rows)
}

/// The `(rows, cols)` of a right-hand side that may be a vector.
fn rhs_matrix_dims_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    match view.shape() {
        [rows] => Ok((*rows, 1)),
        _ => matrix_dims_view(view, op),
    }
}

/// The pooled bytes one provider `*_into` call borrows for its scratch.
///
/// The bytes come from the session pool and return to it when the call ends. The query and the
/// call must take the same `Exec`, which is what makes the size the route actually needs.
fn pooled_workspace<T: PoolScalar>(
    buffers: &BufferPool,
    bytes: tenferro_tensor::Result<usize>,
) -> tenferro_tensor::Result<PooledWorkspace<T>> {
    PooledWorkspace::<T>::new(buffers, bytes?)
}

/// Finish a destination a completed provider call wrote in full, carrying `placement`.
fn tensor_from_lease<T: PoolScalar>(
    lease: PooledUninitOutput<T>,
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = finish_lease(lease)?;
    tensor.set_placement(placement.clone());
    Ok(tensor)
}

fn provider(op: Op) -> impl Fn(tlinalg::Error) -> tenferro_tensor::Error {
    move |error| map_error(op, error)
}

// ---------------------------------------------------------------------------------------------
// Cholesky
// ---------------------------------------------------------------------------------------------

fn cholesky_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "cholesky";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(view.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), view.placement());
    }
    let mut l = pooled_output::<T>(buffers, OP, "matrix", &shape)?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::cholesky::cholesky_workspace::<T>(n, batch, par).map_err(provider(Op::Cholesky)),
    )?;
    tlinalg::cholesky::cholesky_into(
        Op::Cholesky,
        raw_view(OP, view)?,
        l.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Cholesky))?;
    tensor_from_lease(l, view.placement())
}

pub(crate) fn cholesky<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    cholesky_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn cholesky_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    square_matrix_dim_view(&view, "cholesky")?;
    cholesky_impl(ctx, buffers, &view)
}

/// Lower Cholesky factor of one compact `n x n` matrix, for the managed (prepared) route.
///
/// The factor stays a pooled tensor, so the host's staging copy into the managed domain returns its
/// storage to the session pool instead of allocating a fresh vector per call.
pub(crate) fn cholesky_compact_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &[T],
    n: usize,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let expected_len = checked_product("cholesky", "matrix", &[n, n])?;
    if input.len() != expected_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input storage",
            format!("expected {expected_len} elements, got {}", input.len()),
        ));
    }
    if n == 0 {
        return finish_lease(pooled_output::<T>(buffers, "cholesky", "matrix", &[0, 0])?);
    }
    let dims = [n, n];
    let strides = [1, n as isize];
    let descriptor = RawStridedRef::new(input, &dims, &strides, 0).map_err(|error| {
        tenferro_tensor::Error::invalid_argument("cholesky", "layout", error.to_string())
    })?;
    let mut l = pooled_output::<T>(buffers, "cholesky", "matrix", &dims)?;
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::cholesky::cholesky_workspace::<T>(n, 1, par).map_err(provider(Op::Cholesky)),
    )?;
    tlinalg::cholesky::cholesky_into(
        Op::Cholesky,
        descriptor,
        l.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Cholesky))?;
    finish_lease(l)
}

// ---------------------------------------------------------------------------------------------
// LU
// ---------------------------------------------------------------------------------------------

fn lu_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "lu";
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let k = m.min(n);
    let placement = view.placement();
    let p_shape = matrix_with_batch_shape(m, m, batch_shape);
    let l_shape = matrix_with_batch_shape(m, k, batch_shape);
    let u_shape = matrix_with_batch_shape(k, n, batch_shape);
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    if has_zero_dim(view.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(p_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(l_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(u_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); batch],
                placement,
            )?,
        ]);
    }
    let mut p = pooled_output::<T>(buffers, OP, "permutation matrix", &p_shape)?;
    let mut l = pooled_output::<T>(buffers, OP, "L", &l_shape)?;
    let mut u = pooled_output::<T>(buffers, OP, "U", &u_shape)?;
    let mut parity = pooled_output::<T>(buffers, OP, "parity", batch_shape)?;
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::lu::lu_workspace::<T>(m, n, batch, par).map_err(provider(Op::Lu)),
    )?;
    tlinalg::lu::lu_into(
        Op::Lu,
        raw_view(OP, view)?,
        p.as_uninit_slice_mut(),
        l.as_uninit_slice_mut(),
        u.as_uninit_slice_mut(),
        parity.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Lu))?;
    Ok(vec![
        tensor_from_lease(p, placement)?,
        tensor_from_lease(l, placement)?,
        tensor_from_lease(u, placement)?,
        tensor_from_lease(parity, placement)?,
    ])
}

pub(crate) fn lu<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    lu_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn lu_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    matrix_dims_view(&view, "lu")?;
    lu_impl(ctx, buffers, &view)
}

pub(crate) fn lu_factor<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    let (m, n, batch_shape) = matrix_core_and_batch(input.shape(), "lu_factor")?;
    let k = m.min(n);
    // An empty batch has no parity entries; an unbatched empty matrix has one.
    let batch_total = checked_product("lu_factor", "batch shape", batch_shape)?;
    if has_zero_dim(input.shape()) {
        return Ok((
            tensor_from_vec_with_template(input.shape().to_vec(), Vec::new(), input.placement())?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); batch_total],
                input.placement(),
            )?,
        ));
    }

    let pivot_shape = vector_with_batch_shape(k, batch_shape);
    let mut lu_data = pooled_copy::<T>(
        buffers,
        "lu_factor",
        "matrix",
        input.shape(),
        input.host_data()?,
    )?;
    let mut pivot_data = pooled_zeroed::<i32>(buffers, "lu_factor", "pivots", &pivot_shape)?;
    // Every item's parity is written by the factorization, so the prefill only has to leave the
    // buffer initialized.
    let mut parity_data = pooled_zeroed::<T>(buffers, "lu_factor", "parity", &batch_shape)?;
    crate::cpu::tlinalg::factor_batch::<T>(
        ctx,
        buffers,
        Op::LuFactor,
        m,
        n,
        lu_data.host_data_mut()?,
        pivot_data.host_data_mut()?,
        parity_data.host_data_mut()?,
    )?;

    let placement = input.placement();
    lu_data.set_placement(placement.clone());
    pivot_data.set_placement(placement.clone());
    parity_data.set_placement(placement.clone());
    Ok((lu_data, pivot_data, parity_data))
}

// ---------------------------------------------------------------------------------------------
// Full-pivot LU
// ---------------------------------------------------------------------------------------------

fn full_piv_lu_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "full_piv_lu";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let placement = view.placement();
    let shape = matrix_with_batch_shape(n, n, batch_shape);
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    if has_zero_dim(view.shape()) {
        let empty = || tensor_from_vec_with_template(shape.clone(), Vec::new(), placement);
        return Ok(vec![
            empty()?,
            empty()?,
            empty()?,
            empty()?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); batch],
                placement,
            )?,
        ]);
    }
    let mut p = pooled_output::<T>(buffers, OP, "permutation matrix", &shape)?;
    let mut l = pooled_output::<T>(buffers, OP, "L", &shape)?;
    let mut u = pooled_output::<T>(buffers, OP, "U", &shape)?;
    let mut q = pooled_output::<T>(buffers, OP, "permutation matrix", &shape)?;
    let mut parity = pooled_output::<T>(buffers, OP, "parity", batch_shape)?;
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::full_piv_lu::full_piv_lu_workspace::<T>(n, batch, par)
            .map_err(provider(Op::FullPivLu)),
    )?;
    tlinalg::full_piv_lu::full_piv_lu_into(
        Op::FullPivLu,
        raw_view(OP, view)?,
        p.as_uninit_slice_mut(),
        l.as_uninit_slice_mut(),
        u.as_uninit_slice_mut(),
        q.as_uninit_slice_mut(),
        parity.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::FullPivLu))?;
    Ok(vec![
        tensor_from_lease(p, placement)?,
        tensor_from_lease(l, placement)?,
        tensor_from_lease(u, placement)?,
        tensor_from_lease(q, placement)?,
        tensor_from_lease(parity, placement)?,
    ])
}

pub(crate) fn full_piv_lu<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    full_piv_lu_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn full_piv_lu_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    square_matrix_dim_view(&view, "full_piv_lu")?;
    full_piv_lu_impl(ctx, buffers, &view)
}

/// Validate a batched binary solve: square `A`, an `n`-row matrix RHS, and equal batch shapes.
///
/// Returns `(n, nrhs, batch shape)`.
fn solve_shapes<'s>(
    op: &'static str,
    a_shape: &'s [usize],
    b_shape: &'s [usize],
    rhs_core_is_rows: bool,
) -> tenferro_tensor::Result<(usize, usize, &'s [usize])> {
    let (_, _, a_batch_shape) = matrix_core_and_batch(a_shape, op)?;
    let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch(b_shape, op)?;
    let (n, _) = square_core_and_batch(a_shape, op)?;
    let rhs_core_dim = if rhs_core_is_rows { b_rows } else { b_cols };
    if rhs_core_dim != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![rhs_core_dim],
        ));
    }
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }
    Ok((
        n,
        if rhs_core_is_rows { b_cols } else { b_rows },
        b_batch_shape,
    ))
}

pub(crate) fn full_piv_lu_solve<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "full_piv_lu_solve";
    let (n, nrhs, batch_shape) = solve_shapes(OP, a.shape(), b.shape(), true)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut x = pooled_output::<T>(buffers, OP, "solution", b.shape())?;
    let (a_view, b_view) = (a.as_view(), b.as_view());
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::full_piv_lu::full_piv_lu_solve_workspace::<T>(n, nrhs, batch, par)
            .map_err(provider(Op::FullPivLuSolve)),
    )?;
    tlinalg::full_piv_lu::full_piv_lu_solve_into(
        Op::FullPivLuSolve,
        raw_view(OP, &a_view)?,
        raw_view(OP, &b_view)?,
        transpose_a,
        x.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::FullPivLuSolve))?;
    tensor_from_lease(x, b.placement())
}

// ---------------------------------------------------------------------------------------------
// Solve (partial-pivot LU)
// ---------------------------------------------------------------------------------------------

pub(crate) fn solve<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "solve";
    let (n, nrhs, batch_shape) = solve_shapes(OP, a.shape(), b.shape(), true)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    // The right-hand side is copied once into the output, which the solve then overwrites in
    // place: the same single copy the per-item route made.
    let mut output = pooled_copy::<T>(buffers, OP, "solution", b.shape(), b.host_data()?)?;
    let a_view = a.as_view();
    let b_view = b.as_view();
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::lu::lu_solve_workspace::<T>(n, nrhs, transpose_a, batch, par)
            .map_err(provider(Op::Solve)),
    )?;
    {
        // INVARIANT: `output` is a compact copy of `b`, so `b`'s own compact layout describes it.
        let out = RawStridedMut::new(output.host_data_mut()?, b_view.shape(), b_view.strides(), 0)
            .map_err(|error| {
                tenferro_tensor::Error::invalid_argument(OP, "layout", error.to_string())
            })?;
        tlinalg::lu::solve_into(
            Op::Solve,
            raw_view(OP, &a_view)?,
            None,
            out,
            transpose_a,
            workspace.bytes_mut(),
            par,
        )
        .map_err(provider(Op::Solve))?;
    }
    output.set_placement(b.placement().clone());
    Ok(output)
}

/// The `(dims, strides)` a rank-1 or rank-2 right-hand side view presents as an `n x nrhs` matrix.
fn rhs_layout(shape: &[usize], strides: &[isize], n: usize) -> ([usize; 2], [isize; 2]) {
    match (shape, strides) {
        ([rows], [row_stride]) => ([*rows, 1], [*row_stride, n as isize]),
        _ => ([shape[0], shape[1]], [strides[0], strides[1]]),
    }
}

/// Solve one system from borrowed views into a destination the solve fills in place.
///
/// `copy_rhs` selects where the right-hand side comes from: copied from `b` by the provider only
/// after the factorization and singularity check succeed (the direct-output route, which leaves
/// the caller's buffer untouched on failure), or already present in `out`.
// INVARIANT: context, pool, the three operands, the transpose flag, the RHS source and the route
// name are distinct inputs of the one direct-solve boundary both view routes share.
#[allow(clippy::too_many_arguments)]
fn solve_in_place_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensorView<'_, T>,
    b: &TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
    copy_rhs: bool,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    let n = square_matrix_dim_view(a, op)?;
    let (b_rows, _) = rhs_matrix_dims_view(b, op)?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![n],
            vec![b_rows],
        ));
    }
    let a_copy = super::provider_readable(buffers, a, op)?;
    let b_copy = if copy_rhs {
        super::provider_readable(buffers, b, op)?
    } else {
        None
    };
    let a_owned;
    let a = match &a_copy {
        Some(copy) => {
            a_owned = copy.as_view();
            &a_owned
        }
        None => a,
    };
    let b_owned;
    let b = match &b_copy {
        Some(copy) => {
            b_owned = copy.as_view();
            &b_owned
        }
        None => b,
    };
    let (b_dims, b_strides) = rhs_layout(b.shape(), b.strides(), n);
    let (out_dims, out_strides) = rhs_layout(out.shape(), out.strides(), n);
    let out_offset = out.offset();
    if out_strides.iter().any(|&stride| stride < 0) {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "out",
            "output strides must be non-negative",
        ));
    }
    let a_descriptor = raw_view(op, a)?;
    let b_storage = b.host_storage()?;
    let rhs = if copy_rhs {
        Some(
            RawStridedRef::new(b_storage, &b_dims, &b_strides, b.offset()).map_err(|error| {
                tenferro_tensor::Error::invalid_argument(op, "b", error.to_string())
            })?,
        )
    } else {
        None
    };
    let par = exec_from(ctx)?;
    let out = RawStridedMut::new(out.host_storage_mut()?, &out_dims, &out_strides, out_offset)
        .map_err(|error| tenferro_tensor::Error::invalid_argument(op, "out", error.to_string()))?;
    // The view routes are rank-1 or rank-2 only, so one item per call; the query and the call take
    // the same `Exec`, and the provider resolves the same route for both.
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::lu::lu_solve_workspace::<T>(n, out_dims[1], transpose_a, 1, par)
            .map_err(provider(Op::Solve)),
    )?;
    tlinalg::lu::solve_into(
        Op::Solve,
        a_descriptor,
        rhs,
        out,
        transpose_a,
        workspace.bytes_mut(),
        par,
    )
    .map_err(|error| match error {
        // The provider names its own operation; the host reports the route the caller used.
        tlinalg::Error::Singular { .. } => {
            crate::error::into_tensor_error(op, crate::Error::Singular { op })
        }
        other => map_error(Op::Solve, other),
    })
}

pub(crate) fn solve_from_views<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut output = super::output_from_rhs_view(buffers, &b, "solve")?;
    let mut out = output.as_view_mut();
    solve_in_place_view(ctx, buffers, &a, &b, &mut out, transpose_a, false, "solve")?;
    Ok(output)
}

/// Solve a single matrix system directly into a column-major output view. The destination is
/// populated only after factorization and singularity checks have completed, so validation and
/// provider failures do not partially modify the caller-owned buffer.
pub(crate) fn solve_into<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<()> {
    solve_in_place_view(
        ctx,
        buffers,
        &a,
        &b,
        out,
        transpose_a,
        true,
        "solve_read_into",
    )
}

// ---------------------------------------------------------------------------------------------
// Triangular solve
// ---------------------------------------------------------------------------------------------

fn triangular_solve_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensorView<'_, T>,
    b: &TypedTensorView<'_, T>,
    flags: tlinalg::triangular_solve::TriangularSolveFlags,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "triangular_solve";
    let (_, _, batch_shape) = solve_shapes(OP, a.shape(), b.shape(), flags.left_side)?;
    let placement = b.placement();
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), placement);
    }
    let mut x = pooled_output::<T>(buffers, OP, "solution", b.shape())?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::triangular_solve::triangular_solve_workspace::<T>(
            a.shape()[0],
            b.shape()[0],
            b.shape()[1],
            flags,
            batch,
            par,
        )
        .map_err(provider(Op::TriangularSolve)),
    )?;
    tlinalg::triangular_solve::triangular_solve_into(
        Op::TriangularSolve,
        raw_view(OP, a)?,
        raw_view(OP, b)?,
        flags,
        x.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::TriangularSolve))?;
    tensor_from_lease(x, placement)
}

// Keeps triangular-solve operands and flags explicit at the CPU backend boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn triangular_solve<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    triangular_solve_impl(
        ctx,
        buffers,
        &a.as_view(),
        &b.as_view(),
        tlinalg::triangular_solve::TriangularSolveFlags {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        },
    )
}

/// Triangular solve with a borrowed coefficient matrix and right-hand side.
///
/// Both reach the provider as strided descriptors with no host copy; the provider makes the
/// single copy of `b` the destructive solve needs, straight into the output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn triangular_solve_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "triangular_solve";
    square_matrix_dim_view(&a, OP)?;
    // Triangular solve takes a matrix right-hand side on every provider; the borrowed route must
    // not accept a rank the owned route rejects.
    matrix_dims_view(&b, OP)?;
    triangular_solve_impl(
        ctx,
        buffers,
        &a,
        &b,
        tlinalg::triangular_solve::TriangularSolveFlags {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        },
    )
}

// ---------------------------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------------------------

fn svd_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    full: bool,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let op = if full { "svd_full" } else { "svd" };
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), op)?;
    let k = m.min(n);
    let placement = view.placement();
    if has_zero_dim(view.shape()) {
        if full {
            return empty_svd_full_outputs(op, m, n, batch_shape, placement);
        }
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                placement,
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                placement,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                placement,
            )?,
        ]);
    }
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    let u_shape = matrix_with_batch_shape(m, u_cols, batch_shape);
    let s_shape = vector_with_batch_shape(k, batch_shape);
    let vt_shape = matrix_with_batch_shape(v_cols, n, batch_shape);
    let batch = checked_product(op, "batch shape", batch_shape)?;
    let mut u = pooled_output::<T>(buffers, "svd", "left singular vectors", &u_shape)?;
    let mut s = pooled_output::<T>(buffers, "svd", "singular values", &s_shape)?;
    let mut vt = pooled_output::<T>(buffers, "svd", "right singular vectors", &vt_shape)?;
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::svd::svd_workspace::<T>(m, n, full, batch, par).map_err(provider(Op::Svd)),
    )?;
    tlinalg::svd::svd_into(
        Op::Svd,
        raw_view(op, view)?,
        full,
        u.as_uninit_slice_mut(),
        s.as_uninit_slice_mut(),
        vt.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Svd))?;
    Ok(vec![
        tensor_from_lease(u, placement)?,
        tensor_from_lease(s, placement)?,
        tensor_from_lease(vt, placement)?,
    ])
}

pub(crate) fn svd<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    svd_impl(ctx, buffers, &input.as_view(), false)
}

pub(crate) fn svd_full<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    svd_impl(ctx, buffers, &input.as_view(), true)
}

pub(crate) fn svd_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    matrix_dims_view(&view, "svd")?;
    svd_impl(ctx, buffers, &view, false)
}

pub(crate) fn svd_full_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    matrix_dims_view(&view, "svd_full")?;
    svd_impl(ctx, buffers, &view, true)
}

/// Full-SVD factors for an input with an empty dimension.
///
/// The full variant keeps its `m x m` and `n x n` output shapes even when the other core
/// dimension is zero, so a degenerate factor is not empty: the canonical choice is the identity,
/// which preserves `U Uᴴ = I` and `Vᴴ V = I` and reconstructs the (empty) input exactly.
fn empty_svd_full_outputs<T: FaerLinalg>(
    op: &'static str,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let blocks = checked_product(op, "batch shape", batch_shape)?;
    Ok(vec![
        tensor_from_vec_with_template(
            matrix_with_batch_shape(m, m, batch_shape),
            identity_blocks(op, m, blocks)?,
            placement,
        )?,
        tensor_from_vec_with_template(
            vector_with_batch_shape(m.min(n), batch_shape),
            Vec::new(),
            placement,
        )?,
        tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            identity_blocks(op, n, blocks)?,
            placement,
        )?,
    ])
}

/// `blocks` column-major `dim x dim` identity matrices laid out back to back.
fn identity_blocks<T: FaerLinalg>(
    op: &'static str,
    dim: usize,
    blocks: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let per_block = checked_product(op, "identity block", &[dim, dim])?;
    let len = checked_product(op, "identity stack", &[per_block, blocks])?;
    let mut data = vec![T::default(); len];
    for block in 0..blocks {
        let base = block * per_block;
        for index in 0..dim {
            data[base + index + index * dim] = T::parity_one();
        }
    }
    Ok(data)
}

fn svd_values_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    const OP: &str = "svd_values";
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let shape = vector_with_batch_shape(m.min(n), batch_shape);
    if has_zero_dim(view.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), view.placement());
    }
    let mut s = pooled_output::<T::RealScalar>(buffers, OP, "singular values", &shape)?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::svd::svd_values_workspace::<T>(m, n, batch, par).map_err(provider(Op::SvdValues)),
    )?;
    T::svd_values_into(
        raw_view(OP, view)?,
        s.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::SvdValues))?;
    tensor_from_lease(s, view.placement())
}

pub(crate) fn svd_values<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    svd_values_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn svd_values_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    matrix_dims_view(&view, "svd_values")?;
    svd_values_impl(ctx, buffers, &view)
}

// ---------------------------------------------------------------------------------------------
// QR and rank-revealing QR
// ---------------------------------------------------------------------------------------------

fn qr_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "qr";
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let k = m.min(n);
    let placement = view.placement();
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(view.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(q_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(r_shape, Vec::new(), placement)?,
        ]);
    }
    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::qr::qr_workspace::<T>(m, n, batch, par).map_err(provider(Op::Qr)),
    )?;
    tlinalg::qr::qr_into(
        Op::Qr,
        raw_view(OP, view)?,
        q.as_uninit_slice_mut(),
        r.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Qr))?;
    Ok(vec![
        tensor_from_lease(q, placement)?,
        tensor_from_lease(r, placement)?,
    ])
}

pub(crate) fn qr<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    qr_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn qr_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    matrix_dims_view(&view, "qr")?;
    qr_impl(ctx, buffers, &view)
}

/// Column-pivoted QR with the host's rank decision.
///
/// The host screens every item for non-finite entries before the provider runs (the same screen
/// the LAPACK route applies, so both report the same error first). The provider factors the whole
/// batch in one call and gives an all-zero item the canonical zero-rank factors (leading identity
/// columns of `Q`, zero `R`, identity permutation); the host then decides each rank from the `R`
/// diagonal, which is zero for such an item.
fn rank_revealing_qr_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<T>> {
    const OP: &str = "rank_revealing_qr";
    crate::rank_revealing_qr::validate_rank_revealing_qr_options(OP, options)?;
    let (m, n, batch_shape) = matrix_core_and_batch(view.shape(), OP)?;
    let k = m.min(n);
    let placement = view.placement();
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let q_shape = matrix_with_batch_shape(m, k, batch_shape);
    let r_shape = matrix_with_batch_shape(k, n, batch_shape);
    let p_shape = vector_with_batch_shape(n, batch_shape);
    if has_zero_dim(view.shape()) {
    }
    super::rank_revealing_qr::screen_non_finite(OP, view, batch, |value: T| value.is_finite())?;

    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let mut permutation = pooled_output::<i64>(buffers, OP, "permutation", &p_shape)?;
    if batch > 0 {
        let par = exec_from(ctx)?;
        let mut workspace = pooled_workspace::<T>(
            buffers,
            tlinalg::qr::rank_revealing_qr_workspace::<T>(m, n, batch, par)
                .map_err(provider(Op::RankRevealingQr)),
        )?;
        tlinalg::qr::rank_revealing_qr_into(
            Op::RankRevealingQr,
            raw_view(OP, view)?,
            q.as_uninit_slice_mut(),
            r.as_uninit_slice_mut(),
            permutation.as_uninit_slice_mut(),
            workspace.bytes_mut(),
            par,
        )
        .map_err(provider(Op::RankRevealingQr))?;
    }
    let r_filled = tensor_from_lease(r, placement)?;
    let ranks = super::rank_revealing_qr::batch_ranks(
        r_filled.host_data()?,
        k,
        n,
        batch,
        options,
        |value: T| tlinalg::qr::magnitude(value),
    )?;
    Ok(crate::RankRevealingQrResult {
        q: tensor_from_lease(q, placement)?,
        r: r_filled,
        column_permutation: tensor_from_lease(permutation, placement)?,
        rank: tensor_from_vec_with_template(batch_shape.to_vec(), ranks, placement)?,
    })
}

pub(crate) fn rank_revealing_qr<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<T>> {
    rank_revealing_qr_impl(ctx, buffers, &input.as_view(), options)
}

/// Column-pivoted QR of a borrowed 2-D host view, read in place by the provider.
pub(crate) fn rank_revealing_qr_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<T>> {
    crate::rank_revealing_qr::validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
    matrix_dims_view(&view, "rank_revealing_qr")?;
    rank_revealing_qr_impl(ctx, buffers, &view, options)
}

// ---------------------------------------------------------------------------------------------
// Hermitian eigendecomposition
// ---------------------------------------------------------------------------------------------

fn eigh_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "eigh";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let placement = view.placement();
    let values_shape = vector_with_batch_shape(n, batch_shape);
    let vectors_shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(view.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(values_shape, Vec::new(), placement)?,
            tensor_from_vec_with_template(vectors_shape, Vec::new(), placement)?,
        ]);
    }
    let mut values = pooled_output::<T>(buffers, OP, "eigenvalues", &values_shape)?;
    let mut vectors = pooled_output::<T>(buffers, OP, "eigenvectors", &vectors_shape)?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::eigh::eigh_workspace::<T>(n, true, batch, par).map_err(provider(Op::Eigh)),
    )?;
    tlinalg::eigh::eigh_into(
        Op::Eigh,
        raw_view(OP, view)?,
        values.as_uninit_slice_mut(),
        vectors.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::Eigh))?;
    Ok(vec![
        tensor_from_lease(values, placement)?,
        tensor_from_lease(vectors, placement)?,
    ])
}

pub(crate) fn eigh<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    eigh_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn eigh_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    square_matrix_dim_view(&view, "eigh")?;
    eigh_impl(ctx, buffers, &view)
}

fn eigh_values_impl<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    const OP: &str = "eigh_values";
    let (n, batch_shape) = square_core_and_batch(view.shape(), OP)?;
    let shape = vector_with_batch_shape(n, batch_shape);
    if has_zero_dim(view.shape()) {
        return tensor_from_vec_with_template(shape, Vec::new(), view.placement());
    }
    let mut values = pooled_output::<T::RealScalar>(buffers, OP, "eigenvalues", &shape)?;
    let par = exec_from(ctx)?;
    let batch = checked_product(OP, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::eigh::eigh_values_workspace::<T>(n, batch, par).map_err(provider(Op::EighValues)),
    )?;
    T::eigh_values_into(
        raw_view(OP, view)?,
        values.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::EighValues))?;
    tensor_from_lease(values, view.placement())
}

pub(crate) fn eigh_values<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    eigh_values_impl(ctx, buffers, &input.as_view())
}

pub(crate) fn eigh_values_view<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T::RealScalar>> {
    square_matrix_dim_view(&view, "eigh_values")?;
    eigh_values_impl(ctx, buffers, &view)
}

// ---------------------------------------------------------------------------------------------
// General eigendecomposition
// ---------------------------------------------------------------------------------------------

/// Eigenvalues (and with `vectors`, eigenvectors) of a batch, as erased complex tensors.
///
/// General eigendecomposition always returns complex factors, so even empty outputs are tagged
/// complex, exactly as the owned entry point tags them.
fn eig_impl<T: FaerEig>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: &TypedTensorView<'_, T>,
    vectors: bool,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let (op, provider_op) = if vectors {
        ("eig", Op::Eig)
    } else {
        ("eig_values", Op::EigValues)
    };
    let (n, batch_shape) = square_core_and_batch(view.shape(), op)?;
    let values_shape = vector_with_batch_shape(n, batch_shape);
    let vectors_shape = if vectors {
        matrix_with_batch_shape(n, n, batch_shape)
    } else {
        tenferro_tensor::ShapeVec::new()
    };
    if has_zero_dim(view.shape()) {
        let mut outputs = vec![T::wrap(TypedTensor::from_vec_col_major(
            values_shape,
            Vec::new(),
        )?)];
        if vectors {
            outputs.push(T::wrap(TypedTensor::from_vec_col_major(
                vectors_shape,
                Vec::new(),
            )?));
        }
        return Ok(outputs);
    }
    let mut values = pooled_output::<T::ComplexScalar>(buffers, op, "eigenvalues", &values_shape)?;
    let mut vector_lease = if vectors {
        Some(pooled_output::<T::ComplexScalar>(
            buffers,
            op,
            "eigenvectors",
            &vectors_shape,
        )?)
    } else {
        None
    };
    let par = exec_from(ctx)?;
    let batch_count = checked_product(op, "batch shape", batch_shape)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        if vectors {
            tlinalg::eig::eig_workspace::<T>(n, true, batch_count, par)
        } else {
            tlinalg::eig::eig_values_workspace::<T>(n, batch_count, par)
        }
        .map_err(provider(provider_op)),
    )?;
    T::eig_batch(
        raw_view(op, view)?,
        values.as_uninit_slice_mut(),
        vector_lease
            .as_mut()
            .map(|lease| lease.as_uninit_slice_mut()),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(provider_op))?;
    let placement = view.placement();
    let mut outputs = vec![T::wrap(tensor_from_lease(values, placement)?)];
    if let Some(vectors) = vector_lease {
        let mut filled = tensor_from_lease(vectors, placement)?;
        // The public contract's gauge, applied at the seam so every provider
        // agrees (#2148); see `FaerEig::normalize_eig_vectors`.
        T::normalize_eig_vectors(filled.host_data_mut()?, n, batch_count)?;
        outputs.push(T::wrap(filled));
    }
    Ok(outputs)
}
/// Dispatch an erased eig over the four supported dtypes.
macro_rules! eig_dispatch {
    ($tensor:expr, $op:literal, |$typed:ident| $body:expr) => {
        match $tensor {
            TensorView::F32($typed) => $body,
            TensorView::F64($typed) => $body,
            TensorView::C32($typed) => $body,
            TensorView::C64($typed) => $body,
            unsupported => Err(crate::error::unsupported_dtype($op, unsupported.dtype())),
        }
    };
}

/// General eigendecomposition of a borrowed 2-D host view.
pub(crate) fn eig_view(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TensorView<'_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    eig_dispatch!(view, "eig", |typed| {
        square_matrix_dim_view(&typed, "eig")?;
        eig_impl(ctx, buffers, &typed, true)
    })
}

/// Eigenvalues of a borrowed 2-D host view, without the eigenvector solve.
pub(crate) fn eig_values_view(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TensorView<'_>,
) -> tenferro_tensor::Result<Tensor> {
    let mut outputs = eig_dispatch!(view, "eig_values", |typed| {
        square_matrix_dim_view(&typed, "eig_values")?;
        eig_impl(ctx, buffers, &typed, false)
    })?;
    outputs.pop().ok_or_else(|| {
        tenferro_tensor::Error::runtime_state("eig_values", "eigenvalue output missing")
    })
}

fn typed_or_unsupported<'t, T: PoolScalar>(
    input: &'t Tensor,
    op: &'static str,
) -> tenferro_tensor::Result<&'t TypedTensor<T>> {
    input
        .as_typed::<T>()
        .ok_or_else(|| crate::error::unsupported_dtype(op, input.dtype()))
}

pub(crate) fn eig(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match input.dtype() {
        DType::F32 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<f32>(input, "eig")?.as_view(),
            true,
        ),
        DType::F64 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<f64>(input, "eig")?.as_view(),
            true,
        ),
        DType::C32 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<Complex32>(input, "eig")?.as_view(),
            true,
        ),
        DType::C64 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<Complex64>(input, "eig")?.as_view(),
            true,
        ),
        _ => Err(crate::error::unsupported_dtype("eig", input.dtype())),
    }
}

pub(crate) fn eig_values(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    const OP: &str = "eig_values";
    let mut outputs = match input.dtype() {
        DType::F32 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<f32>(input, OP)?.as_view(),
            false,
        ),
        DType::F64 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<f64>(input, OP)?.as_view(),
            false,
        ),
        DType::C32 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<Complex32>(input, OP)?.as_view(),
            false,
        ),
        DType::C64 => eig_impl(
            ctx,
            buffers,
            &typed_or_unsupported::<Complex64>(input, OP)?.as_view(),
            false,
        ),
        _ => Err(crate::error::unsupported_dtype(OP, input.dtype())),
    }?;
    outputs
        .pop()
        .ok_or_else(|| tenferro_tensor::Error::runtime_state(OP, "eigenvalue output missing"))
}

// ---------------------------------------------------------------------------------------------
// Compact Householder QR state (host compositions over the extracted reflector kernels)
// ---------------------------------------------------------------------------------------------

/// Factor one compact `rows x cols` matrix in place; returns its reflector coefficients.
///
/// The coefficients stay a pooled tensor, so their storage returns to the session pool when the
/// caller drops them instead of being exported into an ordinary vector.
fn compact_factor_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &BufferPool,
    data: &mut [T],
    rows: usize,
    cols: usize,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let k = rows.min(cols);
    let mut coeff = pooled_output::<T>(buffers, "compact_factor", "coefficients", &[k])?;
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::householder::compact_factor_workspace::<T>(rows, cols, 1, par)
            .map_err(provider(Op::HouseholderQr)),
    )?;
    tlinalg::householder::compact_factor_into(
        Op::HouseholderQr,
        rows,
        cols,
        1,
        data,
        coeff.as_uninit_slice_mut(),
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::HouseholderQr))?;
    finish_lease(coeff)
}

/// Apply the first `k` reflectors of one compact state to `c` in place.
// INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
#[allow(clippy::too_many_arguments)]
fn apply_reflectors_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &BufferPool,
    a: &[T],
    a_cols: usize,
    coeff: &[T],
    c: &mut [T],
    rows: usize,
    cols: usize,
    k: usize,
    transpose: bool,
) -> tenferro_tensor::Result<()> {
    let shape = tlinalg::householder::ReflectorShape {
        rows,
        a_cols,
        cols,
        k,
    };
    let par = exec_from(ctx)?;
    let mut workspace = pooled_workspace::<T>(
        buffers,
        tlinalg::householder::apply_reflectors_workspace::<T>(shape, 1, transpose, par)
            .map_err(provider(Op::HouseholderQr)),
    )?;
    tlinalg::householder::apply_reflectors_into(
        Op::HouseholderQr,
        shape,
        1,
        a,
        coeff,
        c,
        transpose,
        workspace.bytes_mut(),
        par,
    )
    .map_err(provider(Op::HouseholderQr))
}

fn matrix_dims<T>(
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

fn validate_compact_state<T>(
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize, usize)> {
    let (rows, cols) = matrix_dims(packed, op)?;
    if coeff.shape().len() != 1 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            1,
            coeff.shape().len(),
        ));
    }
    let k = rows.min(cols);
    if coeff.shape()[0] != k {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![k],
            vec![coeff.shape()[0]],
        ));
    }
    Ok((rows, cols, k))
}

pub(crate) fn compact_factor_2d<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    const OP: &str = "compact_factor_2d";
    let (rows, cols) = matrix_dims(input, OP)?;
    let mut packed = pooled_copy::<T>(
        buffers,
        OP,
        "packed state",
        &[rows, cols],
        input.host_data()?,
    )?;
    let mut coeff = compact_factor_data(ctx, buffers, packed.host_data_mut()?, rows, cols)?;
    let k = rows.min(cols);
    if coeff.shape().len() != 1 || coeff.shape()[0] != k {
        return Err(invalid_config(
            OP,
            "coefficients: provider returned an invalid coefficient count",
        ));
    }
    let placement = input.placement();
    packed.set_placement(placement.clone());
    coeff.set_placement(placement.clone());
    Ok((packed, coeff))
}

pub(crate) fn append_2d<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    let (rows, old_cols, old_k) = validate_compact_state(packed, coeff, "append_2d")?;
    let (input_rows, block_cols) = matrix_dims(input, "append_2d")?;
    if input_rows != rows {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "append_2d",
            vec![rows, block_cols],
            vec![input_rows, block_cols],
        ));
    }
    const OP: &str = "append_2d";
    if block_cols == 0 {
        let mut packed_copy = pooled_copy::<T>(
            buffers,
            OP,
            "packed state",
            packed.shape(),
            packed.host_data()?,
        )?;
        let mut coeff_copy = pooled_copy::<T>(
            buffers,
            OP,
            "coefficients",
            coeff.shape(),
            coeff.host_data()?,
        )?;
        packed_copy.set_placement(packed.placement().clone());
        coeff_copy.set_placement(coeff.placement().clone());
        return Ok((packed_copy, coeff_copy));
    }
    let new_cols = old_cols
        .checked_add(block_cols)
        .ok_or_else(|| invalid_config(OP, "shape: column count overflows usize"))?;
    let new_k = rows.min(new_cols);
    let mut transformed = pooled_copy::<T>(
        buffers,
        OP,
        "transformed block",
        &[rows, block_cols],
        input.host_data()?,
    )?;
    apply_reflectors_data(
        ctx,
        buffers,
        packed.host_data()?,
        old_cols,
        coeff.host_data()?,
        transformed.host_data_mut()?,
        rows,
        block_cols,
        old_k,
        true,
    )?;
    let trailing_rows = rows - old_k;
    let mut trailing =
        pooled_zeroed::<T>(buffers, OP, "trailing block", &[trailing_rows, block_cols])?;
    {
        let transformed_data = transformed.host_data()?;
        let trailing_data = trailing.host_data_mut()?;
        for col in 0..block_cols {
            trailing_data[col * trailing_rows..(col + 1) * trailing_rows]
                .copy_from_slice(&transformed_data[col * rows + old_k..(col + 1) * rows]);
        }
    }
    let new_coeff = if trailing_rows == 0 {
        None
    } else {
        Some(compact_factor_data(
            ctx,
            buffers,
            trailing.host_data_mut()?,
            trailing_rows,
            block_cols,
        )?)
    };
    let mut output = pooled_zeroed::<T>(buffers, OP, "packed state", &[rows, new_cols])?;
    {
        let packed_data = packed.host_data()?;
        let transformed_data = transformed.host_data()?;
        let trailing_data = trailing.host_data()?;
        let output_data = output.host_data_mut()?;
        for col in 0..old_cols {
            output_data[col * rows..(col + 1) * rows]
                .copy_from_slice(&packed_data[col * rows..(col + 1) * rows]);
        }
        for col in 0..block_cols {
            let destination = (old_cols + col) * rows;
            output_data[destination..destination + old_k]
                .copy_from_slice(&transformed_data[col * rows..col * rows + old_k]);
            output_data[destination + old_k..destination + rows]
                .copy_from_slice(&trailing_data[col * trailing_rows..(col + 1) * trailing_rows]);
        }
    }
    let mut output_coeff = pooled_zeroed::<T>(buffers, OP, "coefficients", &[new_k])?;
    {
        let coeff_data = coeff.host_data()?;
        let new_coeff_len = match &new_coeff {
            Some(new_coeff) => new_coeff.host_data()?.len(),
            None => 0,
        };
        if coeff_data.len() + new_coeff_len != new_k {
            return Err(invalid_config(
                OP,
                "coefficients: provider returned an invalid coefficient count",
            ));
        }
        let output_coeff_data = output_coeff.host_data_mut()?;
        output_coeff_data[..coeff_data.len()].copy_from_slice(coeff_data);
        if let Some(new_coeff) = &new_coeff {
            let start = coeff_data.len();
            output_coeff_data[start..].copy_from_slice(new_coeff.host_data()?);
        }
    }
    let placement = packed.placement();
    output.set_placement(placement.clone());
    output_coeff.set_placement(placement.clone());
    Ok((output, output_coeff))
}

pub(crate) fn from_factors_2d<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    q: &TypedTensor<T>,
    r: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    let (rows, q_cols) = matrix_dims(q, "from_factors_2d")?;
    let (r_rows, cols) = matrix_dims(r, "from_factors_2d")?;
    if r_rows != q_cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "from_factors_2d",
            vec![q_cols, cols],
            vec![r_rows, cols],
        ));
    }
    if q_cols > rows.min(cols) {
        return Err(invalid_config(
            "from_factors_2d",
            "shape: Q column count must not exceed min(Q rows, R columns)",
        ));
    }
    let k = rows.min(cols);
    let r_data = r.host_data()?;
    for col in 0..cols.min(q_cols) {
        for row in col + 1..q_cols {
            if r_data[row + col * q_cols] != T::default() {
                return Err(invalid_config(
                    "from_factors_2d",
                    "R must be upper trapezoidal",
                ));
            }
        }
    }
    const OP: &str = "from_factors_2d";
    let mut q_packed = pooled_copy::<T>(buffers, OP, "packed Q", &[rows, q_cols], q.host_data()?)?;
    let q_coeff = compact_factor_data(ctx, buffers, q_packed.host_data_mut()?, rows, q_cols)?;
    let mut folded = pooled_zeroed::<T>(buffers, OP, "folded R", &[q_cols, cols])?;
    let mut t = pooled_zeroed::<T>(buffers, OP, "triangular factor", &[q_cols, q_cols])?;
    {
        let q_packed_data = q_packed.host_data()?;
        let t_data = t.host_data_mut()?;
        for col in 0..q_cols {
            for row in 0..q_cols {
                if row <= col {
                    t_data[row + col * q_cols] = q_packed_data[row + col * rows];
                }
            }
        }
    }
    T::gemm_data(
        ctx,
        t.host_data()?,
        q_cols,
        q_cols,
        r_data,
        cols,
        folded.host_data_mut()?,
    )?;
    let mut output = pooled_zeroed::<T>(buffers, OP, "packed state", &[rows, cols])?;
    {
        let q_packed_data = q_packed.host_data()?;
        let folded_data = folded.host_data()?;
        let output_data = output.host_data_mut()?;
        for col in 0..cols {
            for row in 0..rows {
                output_data[row + col * rows] = if col < q_cols && row > col {
                    q_packed_data[row + col * rows]
                } else if row < q_cols {
                    folded_data[row + col * q_cols]
                } else {
                    T::default()
                };
            }
        }
    }
    let mut output_coeff = pooled_zeroed::<T>(buffers, OP, "coefficients", &[k])?;
    {
        let q_coeff_data = q_coeff.host_data()?;
        let output_coeff_data = output_coeff.host_data_mut()?;
        output_coeff_data[..q_coeff_data.len()].copy_from_slice(q_coeff_data);
    }
    let placement = q.placement();
    output.set_placement(placement.clone());
    output_coeff.set_placement(placement.clone());
    Ok((output, output_coeff))
}

pub(crate) fn raw_r_2d<T: FaerLinalg>(
    buffers: &BufferPool,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "raw_r_2d";
    let (rows, cols, k) = validate_compact_state(packed, coeff, OP)?;
    let mut r = pooled_zeroed::<T>(buffers, OP, "R", &[k, cols])?;
    {
        let packed_data = packed.host_data()?;
        let r_data = r.host_data_mut()?;
        for col in 0..cols {
            for row in 0..k.min(col + 1) {
                r_data[row + col * k] = packed_data[row + col * rows];
            }
        }
    }
    if positive_diagonal {
        let r_data = r.host_data_mut()?;
        for diag in 0..k {
            let phase = T::r_phase(r_data[diag + diag * k]);
            for col in 0..cols {
                r_data[diag + col * k] = r_data[diag + col * k] * phase;
            }
        }
    }
    r.set_placement(packed.placement().clone());
    Ok(r)
}

pub(crate) fn q_columns_2d<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    start: usize,
    end: usize,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (rows, _cols, k) = validate_compact_state(packed, coeff, "q_columns_2d")?;
    // Full-Q width: columns `k..rows` span the orthogonal complement of the input's column space,
    // which the compact reflectors represent exactly as well as the thin columns do.
    if start > end || end > rows {
        return Err(invalid_config(
            "q_columns_2d",
            format!("range: range {start}..{end} is outside 0..{rows}"),
        ));
    }
    const OP: &str = "q_columns_2d";
    let columns = end - start;
    let mut q = pooled_zeroed::<T>(buffers, OP, "Q", &[rows, columns])?;
    {
        let q_data = q.host_data_mut()?;
        for col in 0..columns {
            q_data[start + col + col * rows] = T::one();
        }
    }
    if columns != 0 {
        apply_reflectors_data(
            ctx,
            buffers,
            packed.host_data()?,
            packed.shape()[1],
            coeff.host_data()?,
            q.host_data_mut()?,
            rows,
            columns,
            k,
            false,
        )?;
        if positive_diagonal {
            let packed_data = packed.host_data()?;
            let q_data = q.host_data_mut()?;
            // The gauge is defined by R's diagonal, so it fixes only the first `k` columns; a
            // complement column has no diagonal to fix and is left as the reflector product
            // produced it.
            for col in 0..columns {
                let diagonal = start + col;
                if diagonal >= k {
                    break;
                }
                let phase = T::q_phase(packed_data[diagonal + diagonal * rows]);
                for row in 0..rows {
                    q_data[row + col * rows] = q_data[row + col * rows] * phase;
                }
            }
        }
    }
    q.set_placement(packed.placement().clone());
    Ok(q)
}

/// Predicate: can this view be fed to the provider as a strided descriptor?
pub(crate) fn faer_strided_ok<T: 'static>(view: &TypedTensorView<'_, T>) -> bool {
    view.backend_buffer().is_none()
        && view.shape().len() == 2
        && view.strides().iter().all(|&s| s >= 0)
}

#[cfg(test)]
#[path = "faer_linalg/tests.rs"]
mod tests;
