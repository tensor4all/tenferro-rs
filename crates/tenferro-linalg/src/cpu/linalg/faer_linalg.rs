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

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_cpu::CpuExecutionContext;
use tenferro_tensor::{
    DType, Tensor, TensorView, TypedTensor, TypedTensorView, TypedTensorViewMut,
};
use tlinalg::{Op, Parallel};

use super::raw_view;
use crate::cpu::tlinalg::parallel_from;
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
        values: &mut Vec<Self::RealScalar>,
        par: Parallel<'_>,
    ) -> tlinalg::Result<()>;
    /// Batched Hermitian eigenvalues in the real scalar.
    fn eigh_values_into(
        input: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::RealScalar>,
        par: Parallel<'_>,
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
                // faer fans out on the ambient registry, so install the
                // selected pool for this one numerical call. The call is
                // operation-local: the caller's continuation is never installed.
                let run = || {
                    faer::linalg::matmul::matmul(
                        out,
                        faer::Accum::Replace,
                        lhs,
                        rhs,
                        $faer_one,
                        ctx.faer_parallelism(),
                    );
                };
                match ctx.rayon_pool() {
                    Some(pool) => pool.install(run),
                    None => run(),
                }
                Ok(())
            }

            fn svd_values_into(
                input: RawStridedRef<'_, Self>,
                values: &mut Vec<$real>,
                par: Parallel<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::svd::svd_values(Op::SvdValues, input, values, par)
            }

            fn eigh_values_into(
                input: RawStridedRef<'_, Self>,
                values: &mut Vec<$real>,
                par: Parallel<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::eigh::eigh_values(Op::EighValues, input, values, par)
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
        values: &mut Vec<Self::ComplexScalar>,
        vectors: Option<&mut Vec<Self::ComplexScalar>>,
        par: Parallel<'_>,
    ) -> tlinalg::Result<()>;

    fn wrap(tensor: TypedTensor<Self::ComplexScalar>) -> Tensor;
}

macro_rules! impl_faer_eig {
    ($scalar:ty, $complex:ty) => {
        impl FaerEig for $scalar {
            type ComplexScalar = $complex;

            fn eig_batch(
                input: RawStridedRef<'_, Self>,
                values: &mut Vec<$complex>,
                vectors: Option<&mut Vec<$complex>>,
                par: Parallel<'_>,
            ) -> tlinalg::Result<()> {
                match vectors {
                    Some(vectors) => tlinalg::eig::eig(Op::Eig, input, values, vectors, par),
                    None => tlinalg::eig::eig_values(Op::EigValues, input, values, par),
                }
            }

            fn wrap(tensor: TypedTensor<$complex>) -> Tensor {
                Tensor::from_typed::<$complex>(tensor)
            }
        }
    };
}

impl_faer_eig!(f32, Complex32);
impl_faer_eig!(f64, Complex64);
impl_faer_eig!(Complex32, Complex32);
impl_faer_eig!(Complex64, Complex64);

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
    shape: Vec<usize>,
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

fn matrix_with_batch_shape(rows: usize, cols: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch_shape);
    shape
}

fn vector_with_batch_shape(len: usize, batch_shape: &[usize]) -> Vec<usize> {
    let mut shape = vec![len];
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

/// A pooled output buffer for `shape` elements; the provider clears and fills it.
fn pooled_output<T: PoolScalar>(
    buffers: &mut BufferPool,
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<Vec<T>> {
    Ok(buffers.acquire_with_capacity::<T>(checked_product(op, role, shape)?))
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
    let par = parallel_from(ctx);
    tlinalg::cholesky::cholesky(Op::Cholesky, raw_view(OP, view)?, &mut l, par)
        .map_err(provider(Op::Cholesky))?;
    tensor_from_vec_with_template(shape, l, view.placement())
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
pub(crate) fn cholesky_compact_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &[T],
    n: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let expected_len = checked_product("cholesky", "matrix", &[n, n])?;
    if input.len() != expected_len {
        return Err(tenferro_tensor::Error::invalid_argument(
            "cholesky",
            "input storage",
            format!("expected {expected_len} elements, got {}", input.len()),
        ));
    }
    let mut l = buffers.acquire_with_capacity::<T>(expected_len);
    if n == 0 {
        return Ok(l);
    }
    let dims = [n, n];
    let strides = [1, n as isize];
    let descriptor = RawStridedRef::new(input, &dims, &strides, 0).map_err(|error| {
        tenferro_tensor::Error::invalid_argument("cholesky", "layout", error.to_string())
    })?;
    let par = parallel_from(ctx);
    tlinalg::cholesky::cholesky(Op::Cholesky, descriptor, &mut l, par)
        .map_err(provider(Op::Cholesky))?;
    Ok(l)
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
    let mut parity = buffers.acquire_with_capacity::<T>(batch);
    let par = parallel_from(ctx);
    tlinalg::lu::lu(
        Op::Lu,
        raw_view(OP, view)?,
        tlinalg::lu::LuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            parity: &mut parity,
        },
        par,
    )
    .map_err(provider(Op::Lu))?;
    Ok(vec![
        tensor_from_vec_with_template(p_shape, p, placement)?,
        tensor_from_vec_with_template(l_shape, l, placement)?,
        tensor_from_vec_with_template(u_shape, u, placement)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity, placement)?,
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

    let pivot_len = checked_product("lu_factor", "pivots", &[k, batch_total])?;
    let mut lu_data = buffers.acquire_with_capacity::<T>(input.n_elements());
    lu_data.extend_from_slice(input.host_data()?);
    let mut pivot_data = <i32 as PoolScalar>::pool_acquire_zeroed(buffers, pivot_len);
    let mut parity_data = buffers.acquire_with_capacity::<T>(batch_total);
    parity_data.resize(batch_total, T::parity_one());
    crate::cpu::tlinalg::factor_batch::<T>(
        ctx,
        Op::LuFactor,
        m,
        n,
        &mut lu_data,
        &mut pivot_data,
        &mut parity_data,
    )?;

    Ok((
        tensor_from_vec_with_template(input.shape().to_vec(), lu_data, input.placement())?,
        tensor_from_vec_with_template(
            vector_with_batch_shape(k, batch_shape),
            pivot_data,
            input.placement(),
        )?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity_data, input.placement())?,
    ))
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
    let mut parity = buffers.acquire_with_capacity::<T>(batch);
    let par = parallel_from(ctx);
    tlinalg::full_piv_lu::full_piv_lu(
        Op::FullPivLu,
        raw_view(OP, view)?,
        tlinalg::full_piv_lu::FullPivLuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
        par,
    )
    .map_err(provider(Op::FullPivLu))?;
    Ok(vec![
        tensor_from_vec_with_template(shape.clone(), p, placement)?,
        tensor_from_vec_with_template(shape.clone(), l, placement)?,
        tensor_from_vec_with_template(shape.clone(), u, placement)?,
        tensor_from_vec_with_template(shape, q, placement)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity, placement)?,
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
    solve_shapes(OP, a.shape(), b.shape(), true)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    let mut x = pooled_output::<T>(buffers, OP, "solution", b.shape())?;
    let (a_view, b_view) = (a.as_view(), b.as_view());
    let par = parallel_from(ctx);
    tlinalg::full_piv_lu::full_piv_lu_solve(
        Op::FullPivLuSolve,
        raw_view(OP, &a_view)?,
        raw_view(OP, &b_view)?,
        transpose_a,
        &mut x,
        par,
    )
    .map_err(provider(Op::FullPivLuSolve))?;
    tensor_from_vec_with_template(b.shape().to_vec(), x, b.placement())
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
    solve_shapes(OP, a.shape(), b.shape(), true)?;
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    // The right-hand side is copied once into the output, which the solve then overwrites in
    // place: the same single copy the per-item route made.
    let mut output = buffers.acquire_with_capacity::<T>(b.n_elements());
    output.extend_from_slice(b.host_data()?);
    let a_view = a.as_view();
    let b_view = b.as_view();
    let par = parallel_from(ctx);
    {
        // INVARIANT: `output` is a compact copy of `b`, so `b`'s own compact layout describes it.
        let out = RawStridedMut::new(&mut output, b_view.shape(), b_view.strides(), 0).map_err(
            |error| tenferro_tensor::Error::invalid_argument(OP, "layout", error.to_string()),
        )?;
        tlinalg::lu::solve(
            Op::Solve,
            raw_view(OP, &a_view)?,
            None,
            out,
            transpose_a,
            par,
        )
        .map_err(provider(Op::Solve))?;
    }
    tensor_from_vec_with_template(b.shape().to_vec(), output, b.placement())
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
    let par = parallel_from(ctx);
    let out = RawStridedMut::new(out.host_storage_mut()?, &out_dims, &out_strides, out_offset)
        .map_err(|error| tenferro_tensor::Error::invalid_argument(op, "out", error.to_string()))?;
    tlinalg::lu::solve(Op::Solve, a_descriptor, rhs, out, transpose_a, par).map_err(|error| {
        match error {
            // The provider names its own operation; the host reports the route the caller used.
            tlinalg::Error::Singular { .. } => {
                crate::error::into_tensor_error(op, crate::Error::Singular { op })
            }
            other => map_error(Op::Solve, other),
        }
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
    solve_shapes(OP, a.shape(), b.shape(), flags.left_side)?;
    let placement = b.placement();
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), placement);
    }
    let mut x = pooled_output::<T>(buffers, OP, "solution", b.shape())?;
    let par = parallel_from(ctx);
    tlinalg::triangular_solve::triangular_solve(
        Op::TriangularSolve,
        raw_view(OP, a)?,
        raw_view(OP, b)?,
        flags,
        &mut x,
        par,
    )
    .map_err(provider(Op::TriangularSolve))?;
    tensor_from_vec_with_template(b.shape().to_vec(), x, placement)
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
    let mut u = pooled_output::<T>(buffers, "svd", "left singular vectors", &u_shape)?;
    let mut s = pooled_output::<T>(buffers, "svd", "singular values", &s_shape)?;
    let mut vt = pooled_output::<T>(buffers, "svd", "right singular vectors", &vt_shape)?;
    let par = parallel_from(ctx);
    tlinalg::svd::svd(
        Op::Svd,
        raw_view(op, view)?,
        full,
        &mut u,
        &mut s,
        &mut vt,
        par,
    )
    .map_err(provider(Op::Svd))?;
    Ok(vec![
        tensor_from_vec_with_template(u_shape, u, placement)?,
        tensor_from_vec_with_template(s_shape, s, placement)?,
        tensor_from_vec_with_template(vt_shape, vt, placement)?,
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
    let par = parallel_from(ctx);
    T::svd_values_into(raw_view(OP, view)?, &mut s, par).map_err(provider(Op::SvdValues))?;
    tensor_from_vec_with_template(shape, s, view.placement())
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
    let par = parallel_from(ctx);
    tlinalg::qr::qr(Op::Qr, raw_view(OP, view)?, &mut q, &mut r, par).map_err(provider(Op::Qr))?;
    Ok(vec![
        tensor_from_vec_with_template(q_shape, q, placement)?,
        tensor_from_vec_with_template(r_shape, r, placement)?,
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
    super::rank_revealing_qr::screen_non_finite(OP, view, batch, |value: T| value.is_finite())?;

    let mut q = pooled_output::<T>(buffers, OP, "Q", &q_shape)?;
    let mut r = pooled_output::<T>(buffers, OP, "R", &r_shape)?;
    let mut permutation = Vec::with_capacity(checked_product(OP, "permutation", &p_shape)?);
    if batch > 0 {
        let par = parallel_from(ctx);
        tlinalg::qr::rank_revealing_qr(
            Op::RankRevealingQr,
            raw_view(OP, view)?,
            &mut q,
            &mut r,
            &mut permutation,
            par,
        )
        .map_err(provider(Op::RankRevealingQr))?;
    }
    let ranks = super::rank_revealing_qr::batch_ranks(&r, k, n, batch, options, |value: T| {
        tlinalg::qr::magnitude(value)
    })?;
    Ok(crate::RankRevealingQrResult {
        q: tensor_from_vec_with_template(q_shape, q, placement)?,
        r: tensor_from_vec_with_template(r_shape, r, placement)?,
        column_permutation: tensor_from_vec_with_template(p_shape, permutation, placement)?,
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
    let par = parallel_from(ctx);
    tlinalg::eigh::eigh(
        Op::Eigh,
        raw_view(OP, view)?,
        &mut values,
        &mut vectors,
        par,
    )
    .map_err(provider(Op::Eigh))?;
    Ok(vec![
        tensor_from_vec_with_template(values_shape, values, placement)?,
        tensor_from_vec_with_template(vectors_shape, vectors, placement)?,
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
    let par = parallel_from(ctx);
    T::eigh_values_into(raw_view(OP, view)?, &mut values, par).map_err(provider(Op::EighValues))?;
    tensor_from_vec_with_template(shape, values, view.placement())
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
        Vec::new()
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
    let mut vector_data = if vectors {
        pooled_output::<T::ComplexScalar>(buffers, op, "eigenvectors", &vectors_shape)?
    } else {
        Vec::new()
    };
    let par = parallel_from(ctx);
    T::eig_batch(
        raw_view(op, view)?,
        &mut values,
        vectors.then_some(&mut vector_data),
        par,
    )
    .map_err(provider(provider_op))?;
    let placement = view.placement();
    let mut outputs = vec![T::wrap(tensor_from_vec_with_template(
        values_shape,
        values,
        placement,
    )?)];
    if vectors {
        outputs.push(T::wrap(tensor_from_vec_with_template(
            vectors_shape,
            vector_data,
            placement,
        )?));
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
fn compact_factor_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    data: &mut [T],
    rows: usize,
    cols: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let mut coeff = Vec::new();
    let par = parallel_from(ctx);
    tlinalg::householder::compact_factor(Op::HouseholderQr, rows, cols, 1, data, &mut coeff, par)
        .map_err(provider(Op::HouseholderQr))?;
    Ok(coeff)
}

/// Apply the first `k` reflectors of one compact state to `c` in place.
// INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
#[allow(clippy::too_many_arguments)]
fn apply_reflectors_data<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    a: &[T],
    a_cols: usize,
    coeff: &[T],
    c: &mut [T],
    rows: usize,
    cols: usize,
    k: usize,
    transpose: bool,
) -> tenferro_tensor::Result<()> {
    let par = parallel_from(ctx);
    tlinalg::householder::apply_reflectors(
        Op::HouseholderQr,
        tlinalg::householder::ReflectorShape {
            rows,
            a_cols,
            cols,
            k,
        },
        1,
        a,
        coeff,
        c,
        transpose,
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
    let (rows, cols) = matrix_dims(input, "compact_factor_2d")?;
    let input_data = input.host_data()?;
    let mut packed = buffers.acquire_with_capacity::<T>(input_data.len());
    packed.extend_from_slice(input_data);
    let coeff = compact_factor_data(ctx, &mut packed, rows, cols)?;
    let k = rows.min(cols);
    if coeff.len() != k {
        return Err(invalid_config(
            "compact_factor_2d",
            "coefficients: provider returned an invalid coefficient count",
        ));
    }
    Ok((
        tensor_from_vec_with_template(vec![rows, cols], packed, input.placement())?,
        tensor_from_vec_with_template(vec![k], coeff, input.placement())?,
    ))
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
    if block_cols == 0 {
        return Ok((
            tensor_from_vec_with_template(
                packed.shape().to_vec(),
                packed.host_data()?.to_vec(),
                packed.placement(),
            )?,
            tensor_from_vec_with_template(
                coeff.shape().to_vec(),
                coeff.host_data()?.to_vec(),
                coeff.placement(),
            )?,
        ));
    }
    let new_cols = old_cols
        .checked_add(block_cols)
        .ok_or_else(|| invalid_config("append_2d", "shape: column count overflows usize"))?;
    let new_k = rows.min(new_cols);
    let mut transformed = buffers.acquire_with_capacity::<T>(checked_product(
        "append_2d",
        "transformed block",
        &[rows, block_cols],
    )?);
    transformed.extend_from_slice(input.host_data()?);
    apply_reflectors_data(
        ctx,
        packed.host_data()?,
        old_cols,
        coeff.host_data()?,
        &mut transformed,
        rows,
        block_cols,
        old_k,
        true,
    )?;
    let trailing_rows = rows - old_k;
    let trailing_len =
        checked_product("append_2d", "trailing block", &[trailing_rows, block_cols])?;
    let mut trailing = buffers.acquire_with_capacity::<T>(trailing_len);
    for col in 0..block_cols {
        trailing.extend_from_slice(&transformed[col * rows + old_k..(col + 1) * rows]);
    }
    let new_coeff = if trailing_rows == 0 {
        Vec::new()
    } else {
        compact_factor_data(ctx, &mut trailing, trailing_rows, block_cols)?
    };
    let mut output = buffers.acquire_with_capacity::<T>(checked_product(
        "append_2d",
        "packed state",
        &[rows, new_cols],
    )?);
    output.extend_from_slice(packed.host_data()?);
    for col in 0..block_cols {
        output.extend_from_slice(&transformed[col * rows..col * rows + old_k]);
        output.extend_from_slice(&trailing[col * trailing_rows..(col + 1) * trailing_rows]);
    }
    let mut output_coeff = buffers.acquire_with_capacity::<T>(new_k);
    output_coeff.extend_from_slice(coeff.host_data()?);
    output_coeff.extend_from_slice(&new_coeff);
    if output_coeff.len() != new_k {
        return Err(invalid_config(
            "append_2d",
            "coefficients: provider returned an invalid coefficient count",
        ));
    }
    Ok((
        tensor_from_vec_with_template(vec![rows, new_cols], output, packed.placement())?,
        tensor_from_vec_with_template(vec![new_k], output_coeff, packed.placement())?,
    ))
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
    let q_data = q.host_data()?;
    let mut q_packed = buffers.acquire_with_capacity::<T>(q_data.len());
    q_packed.extend_from_slice(q_data);
    let q_coeff = compact_factor_data(ctx, &mut q_packed, rows, q_cols)?;
    let folded_len = checked_product("from_factors_2d", "folded R", &[q_cols, cols])?;
    let mut folded = buffers.acquire_with_capacity::<T>(folded_len);
    folded.resize(folded_len, T::default());
    let triangular_len =
        checked_product("from_factors_2d", "triangular factor", &[q_cols, q_cols])?;
    let mut t = buffers.acquire_with_capacity::<T>(triangular_len);
    t.resize(triangular_len, T::default());
    for col in 0..q_cols {
        for row in 0..q_cols {
            if row <= col {
                t[row + col * q_cols] = q_packed[row + col * rows];
            }
        }
    }
    T::gemm_data(ctx, &t, q_cols, q_cols, r_data, cols, &mut folded)?;
    let output_len = checked_product("from_factors_2d", "packed state", &[rows, cols])?;
    let mut output = buffers.acquire_with_capacity::<T>(output_len);
    output.resize(output_len, T::default());
    for col in 0..cols {
        for row in 0..rows {
            output[row + col * rows] = if col < q_cols && row > col {
                q_packed[row + col * rows]
            } else if row < q_cols {
                folded[row + col * q_cols]
            } else {
                T::default()
            };
        }
    }
    let mut output_coeff = buffers.acquire_with_capacity::<T>(k);
    output_coeff.resize(k, T::default());
    output_coeff[..q_cols].copy_from_slice(&q_coeff);
    Ok((
        tensor_from_vec_with_template(vec![rows, cols], output, q.placement())?,
        tensor_from_vec_with_template(vec![k], output_coeff, q.placement())?,
    ))
}

pub(crate) fn raw_r_2d<T: FaerLinalg>(
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (rows, cols, k) = validate_compact_state(packed, coeff, "raw_r_2d")?;
    let mut r = vec![T::default(); checked_product("raw_r_2d", "R", &[k, cols])?];
    let packed_data = packed.host_data()?;
    for col in 0..cols {
        for row in 0..k.min(col + 1) {
            r[row + col * k] = packed_data[row + col * rows];
        }
    }
    if positive_diagonal {
        for diag in 0..k {
            let phase = T::r_phase(r[diag + diag * k]);
            for col in 0..cols {
                r[diag + col * k] = r[diag + col * k] * phase;
            }
        }
    }
    tensor_from_vec_with_template(vec![k, cols], r, packed.placement())
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
    let columns = end - start;
    let q_len = checked_product("q_columns_2d", "Q", &[rows, columns])?;
    let mut q = buffers.acquire_with_capacity::<T>(q_len);
    q.resize(q_len, T::default());
    for col in 0..columns {
        q[start + col + col * rows] = T::one();
    }
    if columns != 0 {
        apply_reflectors_data(
            ctx,
            packed.host_data()?,
            packed.shape()[1],
            coeff.host_data()?,
            &mut q,
            rows,
            columns,
            k,
            false,
        )?;
        if positive_diagonal {
            let packed_data = packed.host_data()?;
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
                    q[row + col * rows] = q[row + col * rows] * phase;
                }
            }
        }
    }
    tensor_from_vec_with_template(vec![rows, columns], q, packed.placement())
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
