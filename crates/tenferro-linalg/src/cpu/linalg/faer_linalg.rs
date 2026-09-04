use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::ReborrowMut;
use faer::{
    diag::{Diag, DiagRef},
    Conj, Mat, MatMut, MatRef,
};
use num_complex::{Complex32, Complex64};
use std::ops::Range;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_cpu::CpuExecutionContext;
use tenferro_tensor::{Tensor, TensorScalar, TypedTensor, TypedTensorView, TypedTensorViewMut};

pub(crate) trait FaerLinalg:
    Copy + Clone + Default + PartialEq + PoolScalar + std::ops::Mul<Output = Self>
{
    type Real: Copy + Clone + PoolScalar;

    fn parity_one() -> Self;
    fn is_finite(self) -> bool;
    fn magnitude(self) -> f64;
    fn compact_factor_data(
        ctx: &CpuExecutionContext<'_>,
        data: &mut [Self],
        rows: usize,
        cols: usize,
    ) -> tenferro_tensor::Result<Vec<Self>>;
    // INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
    #[allow(clippy::too_many_arguments)]
    fn apply_reflectors_data(
        ctx: &CpuExecutionContext<'_>,
        a: &[Self],
        a_cols: usize,
        coeff: &[Self],
        c: &mut [Self],
        rows: usize,
        cols: usize,
        k: usize,
        transpose: bool,
    ) -> tenferro_tensor::Result<()>;
    fn gemm_data(
        ctx: &CpuExecutionContext<'_>,
        a: &[Self],
        a_rows: usize,
        a_cols: usize,
        b: &[Self],
        b_cols: usize,
        c: &mut [Self],
    ) -> tenferro_tensor::Result<()>;
    fn one() -> Self;
    fn r_phase(diagonal: Self) -> Self;
    fn q_phase(diagonal: Self) -> Self;
    fn cholesky_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn lu_factor_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<(TypedTensor<Self>, TypedTensor<i32>, TypedTensor<Self>)>;
    fn full_piv_lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn full_piv_lu_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn solve_2d_into_view(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: TypedTensorView<'_, Self>,
        b: TypedTensorView<'_, Self>,
        out: &mut TypedTensorViewMut<'_, Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<()> {
        Self::solve_2d_in_place_view(
            ctx,
            buffers,
            a,
            b,
            out,
            transpose_a,
            true,
            "solve_read_into",
        )
    }
    // INVARIANT: these parameters keep the provider kernel's context, pooled
    // scratch, typed views, destination, and solve contract explicit at the
    // single provider boundary; the method is private to this trait.
    #[allow(clippy::too_many_arguments)]
    fn solve_2d_in_place_view(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: TypedTensorView<'_, Self>,
        b: TypedTensorView<'_, Self>,
        out: &mut TypedTensorViewMut<'_, Self>,
        transpose_a: bool,
        copy_rhs: bool,
        op: &'static str,
    ) -> tenferro_tensor::Result<()>;
    // Mirrors triangular-solve math flags directly at the scalar backend boundary.
    #[allow(clippy::too_many_arguments)]
    fn triangular_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>>;
    fn svd_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        full: bool,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn svd_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>>;
    fn qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn rank_revealing_qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<Self>>;
    fn eigh_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn eigh_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>>;

    /// Wrap a compact column-major slice as a faer MatRef.
    fn faer_mat_ref_compact<'a>(data: &'a [Self], m: usize, n: usize) -> MatRef<'a, Self>;
    /// Wrap an arbitrarily-strided 2D host view as a faer MatRef (no copy).
    /// # Safety
    /// Caller must guarantee host placement, exactly two dimensions, finite
    /// non-negative strides, a non-null properly aligned base pointer, and
    /// that every address reached by the shape/stride rectangle is within the
    /// backing allocation for at least `'a`. The shape/stride span and offset
    /// must already have passed `TypedTensorView` layout validation; callers
    /// must not rely on this constructor to check pointer arithmetic or ranges.
    unsafe fn faer_mat_ref_strided<'a>(
        base: *const Self,
        m: usize,
        n: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, Self>;
    fn svd_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        full: bool,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn svd_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>>;
    fn qr_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn eigh_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn eigh_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>>;
    fn cholesky_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
    ) -> tenferro_tensor::Result<Vec<Self>>;
    fn lu_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
    fn full_piv_lu_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>>;
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

fn square_matrix_dim<T>(
    input: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    let (rows, cols) = matrix_dims(input, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok(rows)
}

fn tensor_from_vec_with_template<T: Clone + TensorScalar>(
    shape: Vec<usize>,
    data: Vec<T>,
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(placement.clone());
    Ok(tensor)
}

fn tensor_from_pooled_slice_with_template<T: Clone + PoolScalar>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
    data: &[T],
    placement: &tenferro_tensor::Placement,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut owned = buffers.acquire_with_capacity::<T>(data.len());
    owned.extend_from_slice(data);
    tensor_from_vec_with_template(shape, owned, placement)
}

fn refill_tensor_from_slice<T: Copy + TensorScalar>(
    tensor: &mut TypedTensor<T>,
    data: &[T],
) -> tenferro_tensor::Result<()> {
    tensor.host_data_mut()?.copy_from_slice(data);
    Ok(())
}

fn col_major_vec_from_mat<T: Copy + PoolScalar>(
    buffers: &mut BufferPool,
    mat: MatRef<'_, T>,
) -> tenferro_tensor::Result<Vec<T>> {
    let (rows, cols) = mat.shape();
    let len = checked_product("faer_linalg", "matrix", &[rows, cols])?;
    let mut data = buffers.acquire_with_capacity::<T>(len);
    for j in 0..cols {
        for i in 0..rows {
            data.push(mat[(i, j)]);
        }
    }
    Ok(data)
}

fn vec_from_diag<T: Copy + PoolScalar>(buffers: &mut BufferPool, diag: DiagRef<'_, T>) -> Vec<T> {
    let col = diag.column_vector();
    let mut data = buffers.acquire_with_capacity::<T>(col.nrows());
    for i in 0..col.nrows() {
        data.push(col[i]);
    }
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
            assert_eq!(
                std::mem::size_of::<$complex>(),
                std::mem::size_of::<$faer_complex>()
            );
            assert_eq!(
                std::mem::align_of::<$complex>(),
                std::mem::align_of::<$faer_complex>()
            );

            // SAFETY: size and alignment are checked above in release builds,
            // and both types represent one complex scalar over the same real type.
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<$faer_complex>(), data.len()) }
        }

        fn $to_faer_slice_mut(data: &mut [$complex]) -> &mut [$faer_complex] {
            assert_eq!(
                std::mem::size_of::<$complex>(),
                std::mem::size_of::<$faer_complex>()
            );
            assert_eq!(
                std::mem::align_of::<$complex>(),
                std::mem::align_of::<$faer_complex>()
            );

            // SAFETY: size and alignment are checked above in release builds,
            // and both types represent one complex scalar over the same real type.
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

fn decomposition_failed(op: &'static str) -> tenferro_tensor::Error {
    crate::error::into_tensor_error(op, crate::Error::NonConvergence { op })
}

fn singular_matrix(op: &'static str) -> tenferro_tensor::Error {
    crate::error::into_tensor_error(op, crate::Error::Singular { op })
}

fn invalid_config(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op, "configuration", message)
}

fn eig_imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

fn real_pivot_is_effectively_singular(pivot: f64, max_diagonal: f64, eps: f64) -> bool {
    pivot.abs() <= eps * max_diagonal
}

fn complex_pivot_is_effectively_singular(
    real: f64,
    imag: f64,
    max_diagonal: f64,
    eps: f64,
) -> bool {
    real.hypot(imag) <= eps * max_diagonal
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

fn batch_count(op: &'static str, batch_shape: &[usize]) -> tenferro_tensor::Result<usize> {
    Ok(checked_product(op, "batch shape", batch_shape)?.max(1))
}

fn checked_repeated_len(
    op: &'static str,
    role: &'static str,
    per_batch: usize,
    batch_count: usize,
) -> tenferro_tensor::Result<usize> {
    per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| invalid_config(op, format!("{role} repeated batch length overflows usize")))
}

fn checked_slice_range(
    op: &'static str,
    batch_idx: usize,
    slice_size: usize,
) -> tenferro_tensor::Result<Range<usize>> {
    let start = batch_idx
        .checked_mul(slice_size)
        .ok_or_else(|| invalid_config(op, "batch slice start overflows usize"))?;
    let end = start
        .checked_add(slice_size)
        .ok_or_else(|| invalid_config(op, "batch slice end overflows usize"))?;
    Ok(start..end)
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

pub(crate) fn compact_factor_2d<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<T>)> {
    let (rows, cols) = matrix_dims(input, "compact_factor_2d")?;
    let input_data = input.host_data()?;
    let mut packed = buffers.acquire_with_capacity::<T>(input_data.len());
    packed.extend_from_slice(input_data);
    let coeff = T::compact_factor_data(ctx, &mut packed, rows, cols)?;
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

pub(crate) fn append_2d<T: FaerLinalg + 'static>(
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
    T::apply_reflectors_data(
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
        T::compact_factor_data(ctx, &mut trailing, trailing_rows, block_cols)?
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

pub(crate) fn from_factors_2d<T: FaerLinalg + 'static>(
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
    let q_coeff = T::compact_factor_data(ctx, &mut q_packed, rows, q_cols)?;
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

pub(crate) fn raw_r_2d<T: FaerLinalg + 'static>(
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

pub(crate) fn q_columns_2d<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    packed: &TypedTensor<T>,
    coeff: &TypedTensor<T>,
    start: usize,
    end: usize,
    positive_diagonal: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (rows, _cols, k) = validate_compact_state(packed, coeff, "q_columns_2d")?;
    if start > end || end > k {
        return Err(invalid_config(
            "q_columns_2d",
            format!("range: range {start}..{end} is outside 0..{k}"),
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
        T::apply_reflectors_data(
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
            for col in 0..columns {
                let phase = T::q_phase(packed_data[(start + col) + (start + col) * rows]);
                for row in 0..rows {
                    q[row + col * rows] = q[row + col * rows] * phase;
                }
            }
        }
    }
    tensor_from_vec_with_template(vec![rows, columns], q, packed.placement())
}

macro_rules! impl_complex_vec_helpers {
    (
        $vec_from_real_diag:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident,
        $matrix_from_predicate:ident,
        $complex:ty,
        $faer_complex:ty
    ) => {
        fn $vec_from_real_diag(
            buffers: &mut BufferPool,
            diag: DiagRef<'_, $faer_complex>,
        ) -> Vec<$complex> {
            let col = diag.column_vector();
            let mut data = buffers.acquire_with_capacity::<$complex>(col.nrows());
            for i in 0..col.nrows() {
                data.push(<$complex>::new(col[i].re, 0.0));
            }
            data
        }

        fn $vec_from_diag(
            buffers: &mut BufferPool,
            diag: DiagRef<'_, $faer_complex>,
        ) -> Vec<$complex> {
            let col = diag.column_vector();
            let mut data = buffers.acquire_with_capacity::<$complex>(col.nrows());
            for i in 0..col.nrows() {
                data.push(<$complex>::new(col[i].re, col[i].im));
            }
            data
        }

        fn $vec_from_mat(
            buffers: &mut BufferPool,
            mat: MatRef<'_, $faer_complex>,
        ) -> tenferro_tensor::Result<Vec<$complex>> {
            let (rows, cols) = mat.shape();
            let len = checked_product("faer_linalg", "complex matrix", &[rows, cols])?;
            let mut data = buffers.acquire_with_capacity::<$complex>(len);
            for j in 0..cols {
                for i in 0..rows {
                    let value = mat[(i, j)];
                    data.push(<$complex>::new(value.re, value.im));
                }
            }
            Ok(data)
        }

        fn $matrix_from_predicate(
            mat: MatRef<'_, $faer_complex>,
            rows: usize,
            cols: usize,
            predicate: impl Fn(usize, usize) -> bool,
        ) -> tenferro_tensor::Result<Vec<$complex>> {
            let len = checked_product("faer_linalg", "complex matrix", &[rows, cols])?;
            let mut data = vec![<$complex>::new(0.0, 0.0); len];
            for j in 0..cols {
                for i in 0..rows {
                    if predicate(i, j) {
                        let value = mat[(i, j)];
                        data[i + j * rows] = <$complex>::new(value.re, value.im);
                    }
                }
            }
            Ok(data)
        }
    };
}

fn matrix_from_predicate<T: Copy + Default>(
    mat: MatRef<'_, T>,
    rows: usize,
    cols: usize,
    predicate: impl Fn(usize, usize) -> bool,
) -> tenferro_tensor::Result<Vec<T>> {
    let len = checked_product("faer_linalg", "matrix", &[rows, cols])?;
    let mut data = vec![T::default(); len];
    for j in 0..cols {
        for i in 0..rows {
            if predicate(i, j) {
                data[i + j * rows] = mat[(i, j)];
            }
        }
    }
    Ok(data)
}

fn lower_triangle_vec_from_mat<T: Copy + Default>(
    mat: MatRef<'_, T>,
) -> tenferro_tensor::Result<Vec<T>> {
    let (rows, cols) = mat.shape();
    matrix_from_predicate(mat, rows, cols, |row, col| row >= col)
}

fn upper_triangle_vec_from_mat<T: Copy + Default>(
    mat: MatRef<'_, T>,
) -> tenferro_tensor::Result<Vec<T>> {
    let (rows, cols) = mat.shape();
    matrix_from_predicate(mat, rows, cols, |row, col| row <= col)
}

fn permutation_matrix<T: Copy + Default>(
    perm: &[usize],
    one: T,
) -> tenferro_tensor::Result<Vec<T>> {
    let n = perm.len();
    let len = checked_product("faer_linalg", "permutation matrix", &[n, n])?;
    let mut data = vec![T::default(); len];
    for (row, &source) in perm.iter().enumerate() {
        data[row + source * n] = one;
    }
    Ok(data)
}

fn swap_sequence_from_permutation(
    perm: &[usize],
    k: usize,
    op: &'static str,
) -> tenferro_tensor::Result<Vec<i32>> {
    let mut current: Vec<usize> = (0..perm.len()).collect();
    let mut pivots = Vec::with_capacity(k);
    for (step, &wanted) in perm.iter().take(k).enumerate() {
        let pivot = current
            .iter()
            .position(|&row| row == wanted)
            .ok_or_else(|| invalid_config(op, "invalid row permutation"))?;
        current.swap(step, pivot);
        let pivot_one_based = i32::try_from(pivot + 1)
            .map_err(|_| invalid_config(op, "pivot index exceeds i32 range"))?;
        pivots.push(pivot_one_based);
    }
    Ok(pivots)
}

impl_complex_vec_helpers!(
    complex32_vec_from_real_diag,
    complex32_vec_from_diag,
    complex32_vec_from_mat,
    complex32_matrix_from_predicate,
    Complex32,
    faer::c32
);
impl_complex_vec_helpers!(
    complex64_vec_from_real_diag,
    complex64_vec_from_diag,
    complex64_vec_from_mat,
    complex64_matrix_from_predicate,
    Complex64,
    faer::c64
);

macro_rules! impl_real_eig_to_complex_outputs {
    ($name:ident, $real:ty, $complex:ty) => {
        fn $name(
            buffers: &mut BufferPool,
            u_real: MatRef<'_, $real>,
            s_re: DiagRef<'_, $real>,
            s_im: DiagRef<'_, $real>,
        ) -> tenferro_tensor::Result<(Vec<$complex>, Vec<$complex>)> {
            let n = u_real.nrows();
            let matrix_len = checked_product("eig", "eigenvector matrix", &[n, n])?;
            let mut u = buffers.acquire_with_capacity::<$complex>(matrix_len);
            let mut s = buffers.acquire_with_capacity::<$complex>(n);
            let mut j = 0;
            while j < n {
                if j + 1 >= n
                    || eig_imag_is_effectively_zero(
                        s_re[j] as f64,
                        s_im[j] as f64,
                        <$real>::EPSILON as f64,
                    )
                {
                    s.push(<$complex>::new(s_re[j], 0.0));
                    for i in 0..n {
                        u.push(<$complex>::new(u_real[(i, j)], 0.0));
                    }
                    j += 1;
                } else {
                    s.push(<$complex>::new(s_re[j], s_im[j]));
                    s.push(<$complex>::new(s_re[j], -s_im[j]));
                    for i in 0..n {
                        u.push(<$complex>::new(u_real[(i, j)], u_real[(i, j + 1)]));
                    }
                    for i in 0..n {
                        u.push(<$complex>::new(u_real[(i, j)], -u_real[(i, j + 1)]));
                    }
                    j += 2;
                }
            }
            Ok((u, s))
        }
    };
}

macro_rules! impl_real_eig_to_complex_values {
    ($name:ident, $real:ty, $complex:ty) => {
        fn $name(
            buffers: &mut BufferPool,
            s_re: DiagRef<'_, $real>,
            s_im: DiagRef<'_, $real>,
        ) -> Vec<$complex> {
            let n = s_re.column_vector().nrows();
            let mut s = buffers.acquire_with_capacity::<$complex>(n);
            for j in 0..n {
                s.push(<$complex>::new(s_re[j], s_im[j]));
            }
            s
        }
    };
}

impl_real_eig_to_complex_outputs!(real32_eig_to_complex_outputs, f32, Complex32);
impl_real_eig_to_complex_outputs!(real64_eig_to_complex_outputs, f64, Complex64);
impl_real_eig_to_complex_values!(real32_eig_to_complex_values, f32, Complex32);
impl_real_eig_to_complex_values!(real64_eig_to_complex_values, f64, Complex64);

fn split_shape_core_and_batch<'a>(
    shape: &'a [usize],
    core_rank: usize,
    op: &'static str,
) -> tenferro_tensor::Result<(&'a [usize], &'a [usize])> {
    if shape.len() < core_rank {
        return Err(tenferro_tensor::Error::rank_mismatch(
            op,
            core_rank,
            shape.len(),
        ));
    }
    Ok(shape.split_at(core_rank))
}

fn split_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &'static str,
) -> tenferro_tensor::Result<(&'a [usize], &'a [usize])> {
    split_shape_core_and_batch(input.shape(), core_rank, op)
}

fn matrix_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize, &'a [usize])> {
    let (matrix_shape, batch_shape) = split_core_and_batch(input, 2, op)?;
    Ok((matrix_shape[0], matrix_shape[1], batch_shape))
}

fn square_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, &'a [usize])> {
    let (rows, cols, batch_shape) = matrix_core_and_batch(input, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op,
            vec![rows],
            vec![cols],
        ));
    }
    Ok((rows, batch_shape))
}

fn transpose_col_major_data<T: Copy + PoolScalar>(
    buffers: &mut BufferPool,
    data: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    let mut transposed = buffers.acquire_with_capacity::<T>(data.len());
    for j in 0..rows {
        for i in 0..cols {
            transposed.push(data[j + i * rows]);
        }
    }
    transposed
}

fn batched_single<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    core_rank: usize,
    op: F,
) -> tenferro_tensor::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> tenferro_tensor::Result<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input.placement(),
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_output = op(buffers, &batch_input)?;

        if let Some(expected_shape) = &out_core_shape {
            if batch_output.shape() != expected_shape.as_slice() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    batch_output.shape().to_vec(),
                    expected_shape.clone(),
                ));
            }
        } else {
            let output_elements =
                checked_product(op_name, "output core shape", batch_output.shape())?;
            let capacity =
                checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
            out_data = Some(buffers.acquire_with_capacity::<T>(capacity));
            out_core_shape = Some(batch_output.shape().to_vec());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()?),
            None => {
                return Err(invalid_config(
                    op_name,
                    "missing output buffer after first batch",
                ));
            }
        }
    }

    let mut out_shape =
        out_core_shape.ok_or_else(|| invalid_config(op_name, "missing output shape"))?;
    out_shape.extend_from_slice(batch_shape);
    let out_data = out_data.ok_or_else(|| invalid_config(op_name, "missing output data"))?;
    tensor_from_vec_with_template(out_shape, out_data, input.placement())
}

fn batched_multi_result<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    core_rank: usize,
    op: F,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>
where
    T: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<T>> = Vec::new();

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input.placement(),
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape().to_vec())
                .collect();
            let mut pooled_outputs = Vec::with_capacity(batch_outputs.len());
            for tensor in &batch_outputs {
                let output_elements =
                    checked_product(op_name, "output core shape", tensor.shape())?;
                let capacity =
                    checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
                pooled_outputs.push(buffers.acquire_with_capacity::<T>(capacity));
            }
            out_data = pooled_outputs;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    vec![batch_outputs.len()],
                    vec![out_shapes.len()],
                ));
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape() != out_shapes[idx].as_slice() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    batch_output.shape().to_vec(),
                    out_shapes[idx].clone(),
                ));
            }
            out_data[idx].extend_from_slice(batch_output.host_data()?);
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input.placement())
        })
        .collect()
}

fn batched_multi_convert_result<InT, OutT, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<InT>,
    core_rank: usize,
    op: F,
) -> tenferro_tensor::Result<Vec<TypedTensor<OutT>>>
where
    InT: Clone + PoolScalar,
    OutT: Clone + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<InT>) -> tenferro_tensor::Result<Vec<TypedTensor<OutT>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, core_rank, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<OutT>> = Vec::new();

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input.placement(),
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape().to_vec())
                .collect();
            let mut pooled_outputs = Vec::with_capacity(batch_outputs.len());
            for tensor in &batch_outputs {
                let output_elements =
                    checked_product(op_name, "output core shape", tensor.shape())?;
                let capacity =
                    checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
                pooled_outputs.push(buffers.acquire_with_capacity::<OutT>(capacity));
            }
            out_data = pooled_outputs;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    vec![batch_outputs.len()],
                    vec![out_shapes.len()],
                ));
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape() != out_shapes[idx].as_slice() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    batch_output.shape().to_vec(),
                    out_shapes[idx].clone(),
                ));
            }
            out_data[idx].extend_from_slice(batch_output.host_data()?);
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input.placement())
        })
        .collect()
}

fn batched_binary_result<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    core_rank_a: usize,
    core_rank_b: usize,
    op: F,
) -> tenferro_tensor::Result<TypedTensor<T>>
where
    T: Clone + PoolScalar,
    F: Fn(
        &mut BufferPool,
        &TypedTensor<T>,
        &TypedTensor<T>,
    ) -> tenferro_tensor::Result<TypedTensor<T>>,
{
    let (a_core_shape, a_batch_shape) = split_core_and_batch(a, core_rank_a, op_name)?;
    let (b_core_shape, b_batch_shape) = split_core_and_batch(b, core_rank_b, op_name)?;
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            op_name,
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }

    if a_batch_shape.is_empty() {
        return op(buffers, a, b);
    }

    let a_slice_size = checked_product(op_name, "lhs core shape", a_core_shape)?;
    let b_slice_size = checked_product(op_name, "rhs core shape", b_core_shape)?;
    let batch_count = checked_product(op_name, "batch shape", a_batch_shape)?;
    if batch_count == 0 {
        return Err(invalid_config(
            op_name,
            "zero-sized batch dims must be handled by the caller",
        ));
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    let first_a_range = checked_slice_range(op_name, 0, a_slice_size)?;
    let first_b_range = checked_slice_range(op_name, 0, b_slice_size)?;
    let mut batch_a = tensor_from_pooled_slice_with_template(
        buffers,
        a_core_shape.to_vec(),
        &a.host_data()?[first_a_range],
        a.placement(),
    )?;
    let mut batch_b = tensor_from_pooled_slice_with_template(
        buffers,
        b_core_shape.to_vec(),
        &b.host_data()?[first_b_range],
        b.placement(),
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let a_range = checked_slice_range(op_name, batch_idx, a_slice_size)?;
            let b_range = checked_slice_range(op_name, batch_idx, b_slice_size)?;
            refill_tensor_from_slice(&mut batch_a, &a.host_data()?[a_range])?;
            refill_tensor_from_slice(&mut batch_b, &b.host_data()?[b_range])?;
        }
        let batch_output = op(buffers, &batch_a, &batch_b)?;

        if let Some(expected_shape) = &out_core_shape {
            if batch_output.shape() != expected_shape.as_slice() {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op_name,
                    batch_output.shape().to_vec(),
                    expected_shape.clone(),
                ));
            }
        } else {
            let output_elements =
                checked_product(op_name, "output core shape", batch_output.shape())?;
            let capacity =
                checked_repeated_len(op_name, "output buffer", output_elements, batch_count)?;
            out_data = Some(buffers.acquire_with_capacity::<T>(capacity));
            out_core_shape = Some(batch_output.shape().to_vec());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()?),
            None => {
                return Err(invalid_config(
                    op_name,
                    "missing output buffer after first batch",
                ));
            }
        }
    }

    let mut out_shape =
        out_core_shape.ok_or_else(|| invalid_config(op_name, "missing output shape"))?;
    out_shape.extend_from_slice(a_batch_shape);
    let out_data = out_data.ok_or_else(|| invalid_config(op_name, "missing output data"))?;
    tensor_from_vec_with_template(out_shape, out_data, b.placement())
}

macro_rules! impl_solve_into_view {
    ($scalar:ty, $faer_scalar:ty, $to_faer_slice:ident, $to_faer_slice_mut:ident) => {
        // INVARIANT: this implementation mirrors the explicit provider
        // boundary declared above so the macro cannot silently drop context,
        // destination, or validation parameters.
        #[allow(clippy::too_many_arguments)]
        fn solve_2d_in_place_view(
            ctx: &CpuExecutionContext<'_>,
            _buffers: &mut BufferPool,
            a: TypedTensorView<'_, Self>,
            b: TypedTensorView<'_, Self>,
            out: &mut TypedTensorViewMut<'_, Self>,
            transpose_a: bool,
            copy_rhs: bool,
            op: &'static str,
        ) -> tenferro_tensor::Result<()> {
            let n = square_matrix_dim_view(&a, op)?;
            let (b_rows, b_cols) = rhs_matrix_dims_view(&b, op)?;
            if b_rows != n {
                return Err(tenferro_tensor::Error::shape_mismatch(
                    op,
                    vec![n],
                    vec![b_rows],
                ));
            }

            let mut lu = Mat::<$faer_scalar>::zeros(n, n);
            for col in 0..n {
                for row in 0..n {
                    let value = a.get(&[row, col]).ok_or_else(|| {
                        tenferro_tensor::Error::runtime_state(
                            op,
                            "CPU Faer solve input view is not host-addressable",
                        )
                    })?;
                    lu[(row, col)] = $to_faer_slice(std::slice::from_ref(value))[0];
                }
            }

            let mut row_perm = vec![0usize; n];
            let mut row_perm_inv = vec![0usize; n];
            let mut mem = MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<
                    usize,
                    $faer_scalar,
                >(n, n, ctx.faer_parallelism(), Default::default()),
            );
            let stack = MemStack::new(&mut mem);
            let (_, row_perm_ref) = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
                lu.as_mut(),
                &mut row_perm,
                &mut row_perm_inv,
                ctx.faer_parallelism(),
                stack,
                Default::default(),
            );
            for i in 0..n {
                if lu[(i, i)] == <$faer_scalar as Default>::default() {
                    return Err(singular_matrix(op));
                }
            }

            if copy_rhs {
                copy_rhs_view_into(&b, out, n, b_cols, op)?;
            }
            let out_col_stride = if out.shape().len() == 1 {
                isize::try_from(n).map_err(|_| {
                    tenferro_tensor::Error::invalid_argument(
                        op,
                        "out",
                        "output row count does not fit a Faer stride",
                    )
                })?
            } else {
                out.strides()[1]
            };
            let out_storage = output_storage_slice_mut(out, n, b_cols, out_col_stride, op)?;
            let out_storage = $to_faer_slice_mut(out_storage);
            // SAFETY: the CPU boundary verifies host storage and a positive
            // column-major destination. The aliasing contract rejects
            // input/output overlap before this function is called.
            let rhs = unsafe {
                MatMut::<$faer_scalar>::from_raw_parts_mut(
                    out_storage.as_mut_ptr(),
                    n,
                    b_cols,
                    1,
                    out_col_stride,
                )
            };
            let mut mem = MemBuffer::new(if transpose_a {
                faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
                    usize,
                    $faer_scalar,
                >(n, b_cols, ctx.faer_parallelism())
            } else {
                faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<
                    usize,
                    $faer_scalar,
                >(n, b_cols, ctx.faer_parallelism())
            });
            let stack = MemStack::new(&mut mem);
            if transpose_a {
                faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
                    lu.as_ref(),
                    lu.as_ref(),
                    row_perm_ref,
                    rhs,
                    ctx.faer_parallelism(),
                    stack,
                );
            } else {
                faer::linalg::lu::partial_pivoting::solve::solve_in_place(
                    lu.as_ref(),
                    lu.as_ref(),
                    row_perm_ref,
                    rhs,
                    ctx.faer_parallelism(),
                    stack,
                );
            }
            Ok(())
        }
    };
}

pub(crate) fn solve_from_views<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut output = super::output_from_rhs_view(buffers, &b, "solve")?;
    let a = a;
    let b = b;
    let mut out = output.as_view_mut();
    T::solve_2d_in_place_view(ctx, buffers, a, b, &mut out, transpose_a, false, "solve")?;
    Ok(output)
}

macro_rules! impl_faer_linalg_for_real {
    ($scalar:ty) => {
        impl FaerLinalg for $scalar {
    type Real = $scalar;

    fn parity_one() -> Self {
        1.0
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn magnitude(self) -> f64 {
        self.abs() as f64
    }

    fn compact_factor_data(
        ctx: &CpuExecutionContext<'_>,
        data: &mut [Self],
        rows: usize,
        cols: usize,
    ) -> tenferro_tensor::Result<Vec<Self>> {
        let expected = checked_product("compact_factor_2d", "matrix", &[rows, cols])?;
        if data.len() != expected {
            return Err(invalid_config("compact_factor_2d", "matrix: input buffer length does not match dimensions"));
        }
        let k = rows.min(cols);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut qr = MatMut::from_column_major_slice_mut(data, rows, cols);
        let mut coeff = Vec::with_capacity(k);
        for j in 0..k {
            let mut column = qr.as_mut().col_mut(j).subrows_mut(j, rows - j);
            let (mut head, tail) = column.rb_mut().split_at_row_mut(1);
            let info = faer::linalg::householder::make_householder_in_place(&mut head[0], tail);
            let beta = head[0];
            head[0] = 1.0 as $scalar;
            // Faer stores H = I - vv^H/tau; compact state stores 1/tau.
            coeff.push((1.0 as $scalar) / info.tau);
            if j + 1 < cols {
                let mut basis = Mat::<$scalar>::zeros(rows - j, 1);
                basis[(0, 0)] = 1.0 as $scalar;
                for row in 1..rows - j {
                    basis[(row, 0)] = qr[(j + row, j)];
                }
                let factor = Mat::from_fn(1, 1, |_, _| info.tau);
                let mut mem = MemBuffer::new(faer::linalg::householder::apply_block_householder_on_the_left_in_place_scratch::<$scalar>(rows - j, 1, cols - j - 1));
                let stack = MemStack::new(&mut mem);
                faer::linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
                    basis.as_ref(), factor.as_ref(), Conj::No,
                    qr.as_mut().submatrix_mut(j, j + 1, rows - j, cols - j - 1),
                    ctx.faer_parallelism(), stack,
                );
            }
            qr[(j, j)] = beta;
        }
        Ok(coeff)
    }

    // INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
    #[allow(clippy::too_many_arguments)]
    fn apply_reflectors_data(
        ctx: &CpuExecutionContext<'_>,
        a: &[Self],
        a_cols: usize,
        coeff: &[Self],
        c: &mut [Self],
        rows: usize,
        cols: usize,
        k: usize,
        transpose: bool,
    ) -> tenferro_tensor::Result<()> {
        if k > rows || k > a_cols || coeff.len() != k {
            return Err(invalid_config("apply_reflectors_2d", "dimensions: reflector count exceeds matrix dimensions"));
        }
        if a.len() != checked_product("apply_reflectors_2d", "A", &[rows, a_cols])?
            || c.len() != checked_product("apply_reflectors_2d", "C", &[rows, cols])?
        {
            return Err(invalid_config("apply_reflectors_2d", "matrix: input buffer length does not match dimensions"));
        }
        if rows == 0 || cols == 0 || k == 0 {
            return Ok(());
        }
        let basis = MatRef::from_column_major_slice(a, rows, a_cols).subcols(0, k);
        let factors = Mat::from_fn(1, k, |_, col| {
            // Rebuild Faer's denominator form transiently from state coeff.
            (1.0 as $scalar) / coeff[col]
        });
        let mut matrix = MatMut::from_column_major_slice_mut(c, rows, cols);
        let scratch = if transpose {
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<$scalar>(rows, 1, cols)
        } else {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$scalar>(rows, 1, cols)
        };
        let mut mem = MemBuffer::new(scratch);
        let stack = MemStack::new(&mut mem);
        if transpose {
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                basis.as_ref(), factors.as_ref(), Conj::No, matrix.as_mut(), ctx.faer_parallelism(), stack,
            );
        } else {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                basis.as_ref(), factors.as_ref(), Conj::No, matrix.as_mut(), ctx.faer_parallelism(), stack,
            );
        }
        Ok(())
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
            return Err(invalid_config("from_factors_2d", "matrix: input buffer length does not match dimensions"));
        }
        if a_rows == 0 || a_cols == 0 || b_cols == 0 {
            return Ok(());
        }
        let lhs = MatRef::from_column_major_slice(a, a_rows, a_cols);
        let rhs = MatRef::from_column_major_slice(b, a_cols, b_cols);
        let out = MatMut::from_column_major_slice_mut(c, a_rows, b_cols);
        faer::linalg::matmul::matmul(out, faer::Accum::Replace, lhs, rhs, 1.0 as $scalar, ctx.faer_parallelism());
        Ok(())
    }

    fn one() -> Self { 1.0 }
    fn r_phase(diagonal: Self) -> Self { if diagonal < 0.0 { -1.0 } else { 1.0 } }
    fn q_phase(diagonal: Self) -> Self { Self::r_phase(diagonal) }

    fn cholesky_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        tensor_from_vec_with_template(
            vec![n, n],
            Self::cholesky_core(ctx, buffers, mat, n)?,
            input.placement(),
        )
    }

    fn cholesky_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
    ) -> tenferro_tensor::Result<Vec<Self>> {
        let mut l = Mat::zeros(n, n);
        l.copy_from(mat);
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<Self>(
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("cholesky"))?;
        lower_triangle_vec_from_mat(l.as_ref())
    }

    fn lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "lu")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::lu_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn lu_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(mat);
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                m,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;

        let p_len = checked_product("lu", "permutation matrix", &[m, m])?;
        let mut p_data = vec![0.0; p_len];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = 1.0;
        }
        let parity = if info.transposition_count % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        let mut l_data = matrix_from_predicate(lu.as_ref(), m, k, |row, col| row >= col)?;
        for i in 0..k {
            l_data[i + i * m] = 1.0;
        }
        let u_data = upper_triangle_vec_from_mat(lu.as_ref().get(..k, ..))?;

        Ok(vec![
            tensor_from_vec_with_template(vec![m, m], p_data, placement)?,
            tensor_from_vec_with_template(vec![m, k], l_data, placement)?,
            tensor_from_vec_with_template(vec![k, n], u_data, placement)?,
            tensor_from_vec_with_template(vec![], vec![parity], placement)?,
        ])
    }

    fn lu_factor_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<(TypedTensor<Self>, TypedTensor<i32>, TypedTensor<Self>)> {
        let (m, n) = matrix_dims(input, "lu_factor")?;
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(MatRef::from_column_major_slice(input.host_data()?, m, n));
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                m,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;
        let parity = if info.transposition_count % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let pivots = swap_sequence_from_permutation(&perm, k, "lu_factor")?;

        Ok((
            tensor_from_vec_with_template(vec![m, n], col_major_vec_from_mat(buffers, lu.as_ref())?, input.placement())?,
            tensor_from_vec_with_template(vec![k], pivots, input.placement())?,
            tensor_from_vec_with_template(vec![], vec![parity], input.placement())?,
        ))
    }

    fn full_piv_lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "full_piv_lu")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        Self::full_piv_lu_core(ctx, buffers, mat, n, input.placement())
    }

    fn full_piv_lu_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let mut lu = Mat::zeros(n, n);
        lu.copy_from(mat);
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                n,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;

        let mut l_data = lower_triangle_vec_from_mat(lu.as_ref())?;
        for i in 0..n {
            l_data[i + i * n] = 1.0;
        }
        let u_data = upper_triangle_vec_from_mat(lu.as_ref())?;
        let parity = if info.transposition_count % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        Ok(vec![
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&row_perm, 1.0)?, placement)?,
            tensor_from_vec_with_template(vec![n, n], l_data, placement)?,
            tensor_from_vec_with_template(vec![n, n], u_data, placement)?,
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&col_perm, 1.0)?, placement)?,
            tensor_from_vec_with_template(vec![], vec![parity], placement)?,
        ])
    }

    fn full_piv_lu_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "full_piv_lu_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(tenferro_tensor::Error::shape_mismatch("full_piv_lu_solve", vec![n], vec![b_rows]));
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(a.host_data()?, n, n));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, Self>(
                n,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref, col_perm_ref) = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        );
        let max_diagonal = (0..n)
            .map(|i| lu[(i, i)].abs() as f64)
            .fold(0.0, f64::max);
        for i in 0..n {
            if real_pivot_is_effectively_singular(
                lu[(i, i)] as f64,
                max_diagonal,
                <$scalar>::EPSILON as f64,
            ) {
                return Err(singular_matrix("full_piv_lu_solve"));
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data()?.len());
        rhs_data.extend_from_slice(b.host_data()?);
        let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, Self>(
                n,
                b_cols,
                ctx.faer_parallelism(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_parallelism(),
                stack,
            );
        } else {
            faer::linalg::lu::full_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_parallelism(),
                stack,
            );
        }
        tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b.placement())
    }

    fn solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        solve_from_views(ctx, buffers, a.as_view(), b.as_view(), transpose_a)
    }

    impl_solve_into_view!($scalar, $scalar, same_slice, same_slice_mut);

    fn triangular_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "triangular_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
        let a_mat = MatRef::from_column_major_slice(a.host_data()?, n, n);

        if left_side {
            if b_rows != n {
                return Err(tenferro_tensor::Error::shape_mismatch("triangular_solve", vec![n], vec![b_rows]));
            }
            let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data()?.len());
            rhs_data.extend_from_slice(b.host_data()?);
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_data, n, b_cols);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
            }
            Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b.placement())?)
        } else {
            if b_cols != n {
                return Err(tenferro_tensor::Error::shape_mismatch("triangular_solve", vec![n], vec![b_cols]));
            }
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(buffers, b.host_data()?, nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(&mut rhs_transposed, n, nrhs);
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
            }
            let result = transpose_col_major_data(buffers, &rhs_transposed, n, nrhs);
            <Self as PoolScalar>::pool_release(buffers, rhs_transposed);
            Ok(tensor_from_vec_with_template(vec![nrhs, n], result, b.placement())?)
        }
    }

    fn svd_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        full: bool,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "svd")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::svd_core(ctx, buffers, mat, m, n, full, input.placement())
    }

    fn svd_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let (m, n) = matrix_dims(input, "svd_values")?;
        let mat = MatRef::from_column_major_slice(input.host_data()?, m, n);
        Self::svd_values_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn svd_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let k = m.min(n);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<Self>(
            m,
            n,
            faer::linalg::svd::ComputeSvdVectors::No,
            faer::linalg::svd::ComputeSvdVectors::No,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            None,
            None,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd_values"))?;

        tensor_from_vec_with_template(vec![k], vec_from_diag(buffers, s.as_ref()), placement)
    }

    fn qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "qr")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::qr_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn rank_revealing_qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<Self>> {
        let (m, n) = matrix_dims(input, "rank_revealing_qr")?;
        let k = m.min(n);
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<Self>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(Self::faer_mat_ref_compact(input.host_data()?, m, n));
        let mut coeff = Mat::zeros(block_size, k);
        let mut permutation = vec![0usize; n];
        let mut inverse_permutation = vec![0usize; n];
        // Following faer 0.24's public column-pivoted QR factor API; the
        // provider's forward permutation is the public gather convention.
        let mut mem = MemBuffer::new(
            faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, Self>(
                m,
                n,
                block_size,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        faer::linalg::qr::col_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            &mut permutation,
            &mut inverse_permutation,
            ctx.faer_parallelism(),
            MemStack::new(&mut mem),
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Self>(
                m,
                block_size,
                k,
            ),
        );
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_parallelism(),
            MemStack::new(&mut mem),
        );
        let rank = super::rank_revealing_qr::prefix_rank(
            (0..k).map(|index| qr[(index, index)].abs() as f64),
            options,
        )?;
        let permutation = permutation
            .into_iter()
            .map(|column| {
                i64::try_from(column).map_err(|_| {
                    invalid_config(
                        "rank_revealing_qr",
                        "column permutation exceeds i64 range",
                    )
                })
            })
            .collect::<tenferro_tensor::Result<Vec<_>>>()?;
        Ok(crate::RankRevealingQrResult {
            q: tensor_from_vec_with_template(
                vec![m, k],
                col_major_vec_from_mat(buffers, q.as_ref())?,
                input.placement(),
            )?,
            r: tensor_from_vec_with_template(
                vec![k, n],
                upper_triangle_vec_from_mat(qr.as_ref().get(..k, ..))?,
                input.placement(),
            )?,
            column_permutation: tensor_from_vec_with_template(
                vec![n],
                permutation,
                input.placement(),
            )?,
            rank: tensor_from_vec_with_template(vec![], vec![rank], input.placement())?,
        })
    }

    fn eigh_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "eigh")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        Self::eigh_core(ctx, buffers, mat, n, input.placement())
    }

    fn eigh_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let n = square_matrix_dim(input, "eigh_values")?;
        let mat = MatRef::from_column_major_slice(input.host_data()?, n, n);
        Self::eigh_values_core(ctx, buffers, mat, n, input.placement())
    }

    fn eigh_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let mut values = Diag::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<Self>(
            n,
            faer::linalg::evd::ComputeEigenvectors::No,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            None,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh_values"))?;

        tensor_from_vec_with_template(vec![n], vec_from_diag(buffers, values.as_ref()), placement)
    }

    fn faer_mat_ref_compact<'a>(data: &'a [Self], m: usize, n: usize) -> MatRef<'a, Self> {
        MatRef::from_column_major_slice(data, m, n)
    }

    unsafe fn faer_mat_ref_strided<'a>(
        base: *const Self,
        m: usize,
        n: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, Self> {
        // SAFETY: caller guarantees host placement, rank 2, non-negative strides, and 'a lifetime.
        unsafe { MatRef::from_raw_parts(base, m, n, row_stride, col_stride) }
    }

    fn svd_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        full: bool,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let k = m.min(n);
        // Full mode returns the square unitary factors `U (m x m)` and
        // `V (n x n)`; thin mode keeps only the leading `k` vectors. The
        // singular-value count is `k` in both modes.
        let (u_cols, v_cols, vectors) = if full {
            (m, n, faer::linalg::svd::ComputeSvdVectors::Full)
        } else {
            (k, k, faer::linalg::svd::ComputeSvdVectors::Thin)
        };
        let mut u = Mat::zeros(m, u_cols);
        let mut v = Mat::zeros(n, v_cols);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<Self>(
            m,
            n,
            vectors,
            vectors,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd"))?;

        let u = tensor_from_vec_with_template(
            vec![m, u_cols],
            col_major_vec_from_mat(buffers, u.as_ref())?,
            placement,
        )?;
        let s =
            tensor_from_vec_with_template(vec![k], vec_from_diag(buffers, s.as_ref()), placement)?;
        let vt_len = checked_product("svd", "right singular vectors", &[v_cols, n])?;
        let mut vt_data = buffers.acquire_with_capacity::<Self>(vt_len);
        for j in 0..n {
            for i in 0..v_cols {
                vt_data.push(v[(j, i)]);
            }
        }
        let vt = tensor_from_vec_with_template(vec![v_cols, n], vt_data, placement)?;

        Ok(vec![u, s, vt])
    }

    fn qr_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let k = m.min(n);
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<Self>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(mat);
        let mut coeff = Mat::zeros(block_size, k);
        let mut mem = MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<Self>(
                m,
                n,
                block_size,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<Self>(
                m,
                block_size,
                k,
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_parallelism(),
            stack,
        );
        let q = tensor_from_vec_with_template(
            vec![m, k],
            col_major_vec_from_mat(buffers, q.as_ref())?,
            placement,
        )?;
        let r = tensor_from_vec_with_template(
            vec![k, n],
            upper_triangle_vec_from_mat(qr.as_ref().get(..k, ..))?,
            placement,
        )?;

        Ok(vec![q, r])
    }

    fn eigh_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let mut values = Diag::zeros(n);
        let mut vectors = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<Self>(
            n,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            Some(vectors.as_mut()),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh"))?;

        let values = tensor_from_vec_with_template(
            vec![n],
            vec_from_diag(buffers, values.as_ref()),
            placement,
        )?;
        let vectors = tensor_from_vec_with_template(
            vec![n, n],
            col_major_vec_from_mat(buffers, vectors.as_ref())?,
            placement,
        )?;

        Ok(vec![values, vectors])
    }
        }
    };
}

impl_faer_linalg_for_real!(f32);
impl_faer_linalg_for_real!(f64);

macro_rules! impl_faer_linalg_for_complex {
    (
        $complex:ty,
        $real:ty,
        $faer_complex:ty,
        $to_faer_slice:ident,
        $to_faer_slice_mut:ident,
        $vec_from_real_diag:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident,
        $matrix_from_predicate:ident
    ) => {
        impl FaerLinalg for $complex {
    type Real = $real;

    fn parity_one() -> Self {
        <$complex>::new(1.0, 0.0)
    }

    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    fn magnitude(self) -> f64 {
        self.norm() as f64
    }

    fn compact_factor_data(
        ctx: &CpuExecutionContext<'_>,
        data: &mut [Self],
        rows: usize,
        cols: usize,
    ) -> tenferro_tensor::Result<Vec<Self>> {
        let expected = checked_product("compact_factor_2d", "matrix", &[rows, cols])?;
        if data.len() != expected {
            return Err(invalid_config("compact_factor_2d", "matrix: input buffer length does not match dimensions"));
        }
        let k = rows.min(cols);
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut qr = MatMut::from_column_major_slice_mut(
            $to_faer_slice_mut(data),
            rows,
            cols,
        );
        let mut coeff = Vec::with_capacity(k);
        for j in 0..k {
            let mut column = qr.as_mut().col_mut(j).subrows_mut(j, rows - j);
            let (mut head, tail) = column.rb_mut().split_at_row_mut(1);
            let info = faer::linalg::householder::make_householder_in_place(&mut head[0], tail);
            let beta = head[0];
            head[0] = <$faer_complex>::new(1.0, 0.0);
            // Faer stores H = I - vv^H/tau; compact state stores 1/tau.
            coeff.push(<$complex>::new(1.0 / info.tau, 0.0));
            if j + 1 < cols {
                let mut basis = Mat::<$faer_complex>::zeros(rows - j, 1);
                basis[(0, 0)] = <$faer_complex>::new(1.0, 0.0);
                for row in 1..rows - j {
                    basis[(row, 0)] = qr[(j + row, j)];
                }
                let factor = Mat::from_fn(1, 1, |_, _| <$faer_complex>::new(info.tau, 0.0));
                let mut mem = MemBuffer::new(faer::linalg::householder::apply_block_householder_on_the_left_in_place_scratch::<$faer_complex>(rows - j, 1, cols - j - 1));
                let stack = MemStack::new(&mut mem);
                faer::linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
                    basis.as_ref(), factor.as_ref(), Conj::No,
                    qr.as_mut().submatrix_mut(j, j + 1, rows - j, cols - j - 1),
                    ctx.faer_parallelism(), stack,
                );
            }
            qr[(j, j)] = beta;
        }
        Ok(coeff)
    }

    // INVARIANT: these buffers and dimensions mirror the provider reflector ABI.
    #[allow(clippy::too_many_arguments)]
    fn apply_reflectors_data(
        ctx: &CpuExecutionContext<'_>,
        a: &[Self],
        a_cols: usize,
        coeff: &[Self],
        c: &mut [Self],
        rows: usize,
        cols: usize,
        k: usize,
        transpose: bool,
    ) -> tenferro_tensor::Result<()> {
        if k > rows || k > a_cols || coeff.len() != k {
            return Err(invalid_config("apply_reflectors_2d", "dimensions: reflector count exceeds matrix dimensions"));
        }
        if a.len() != checked_product("apply_reflectors_2d", "A", &[rows, a_cols])?
            || c.len() != checked_product("apply_reflectors_2d", "C", &[rows, cols])?
        {
            return Err(invalid_config("apply_reflectors_2d", "matrix: input buffer length does not match dimensions"));
        }
        if rows == 0 || cols == 0 || k == 0 {
            return Ok(());
        }
        let basis = MatRef::from_column_major_slice($to_faer_slice(a), rows, a_cols)
            .subcols(0, k);
        let coeff = $to_faer_slice(coeff);
        let factors = Mat::from_fn(1, k, |_, col| {
            // Rebuild Faer's denominator form transiently from state coeff.
            <$faer_complex>::new(1.0 / coeff[col].re, 0.0)
        });
        let mut matrix = MatMut::from_column_major_slice_mut(
            $to_faer_slice_mut(c),
            rows,
            cols,
        );
        let scratch = if transpose {
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<$faer_complex>(rows, 1, cols)
        } else {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$faer_complex>(rows, 1, cols)
        };
        let mut mem = MemBuffer::new(scratch);
        let stack = MemStack::new(&mut mem);
        if transpose {
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                basis.as_ref(), factors.as_ref(), Conj::Yes, matrix.as_mut(), ctx.faer_parallelism(), stack,
            );
        } else {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                basis.as_ref(), factors.as_ref(), Conj::No, matrix.as_mut(), ctx.faer_parallelism(), stack,
            );
        }
        Ok(())
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
            return Err(invalid_config("from_factors_2d", "matrix: input buffer length does not match dimensions"));
        }
        if a_rows == 0 || a_cols == 0 || b_cols == 0 {
            return Ok(());
        }
        let lhs = MatRef::from_column_major_slice($to_faer_slice(a), a_rows, a_cols);
        let rhs = MatRef::from_column_major_slice($to_faer_slice(b), a_cols, b_cols);
        let out = MatMut::from_column_major_slice_mut($to_faer_slice_mut(c), a_rows, b_cols);
        faer::linalg::matmul::matmul(out, faer::Accum::Replace, lhs, rhs, <$faer_complex>::new(1.0, 0.0), ctx.faer_parallelism());
        Ok(())
    }

    fn one() -> Self { <$complex>::new(1.0, 0.0) }
    fn r_phase(diagonal: Self) -> Self {
        let norm = diagonal.norm();
        if norm == 0.0 { Self::one() } else { diagonal.conj() / norm }
    }
    fn q_phase(diagonal: Self) -> Self {
        let norm = diagonal.norm();
        if norm == 0.0 { Self::one() } else { diagonal / norm }
    }

    fn cholesky_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(input, "cholesky")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        tensor_from_vec_with_template(
            vec![n, n],
            Self::cholesky_core(ctx, buffers, mat, n)?,
            input.placement(),
        )
    }

    fn cholesky_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
    ) -> tenferro_tensor::Result<Vec<Self>> {
        // Cast Self (Complex32/64) to $faer_complex (faer::c32/c64) for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                n,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let mut l = Mat::zeros(n, n);
        l.copy_from(mat);
        let mut mem = MemBuffer::new(
            faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<$faer_complex>(
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            l.as_mut(),
            Default::default(),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("cholesky"))?;
        $matrix_from_predicate(l.as_ref(), n, n, |row, col| row >= col)
    }

    fn lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "lu")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::lu_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn lu_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        // Cast Self (Complex32/64) to $faer_complex (faer::c32/c64) for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                m,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(mat);
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                m,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;

        let p_len = checked_product("lu", "permutation matrix", &[m, m])?;
        let mut p_data = vec![<$complex>::new(0.0, 0.0); p_len];
        for (row, &col) in perm.iter().enumerate() {
            p_data[row + col * m] = <$complex>::new(1.0, 0.0);
        }
        let parity = if info.transposition_count % 2 == 0 {
            <$complex>::new(1.0, 0.0)
        } else {
            <$complex>::new(-1.0, 0.0)
        };
        let mut l_data = $matrix_from_predicate(lu.as_ref(), m, k, |row, col| row >= col)?;
        for i in 0..k {
            l_data[i + i * m] = <$complex>::new(1.0, 0.0);
        }
        let u_data = $matrix_from_predicate(lu.as_ref(), k, n, |row, col| row <= col)?;

        Ok(vec![
            tensor_from_vec_with_template(vec![m, m], p_data, placement)?,
            tensor_from_vec_with_template(vec![m, k], l_data, placement)?,
            tensor_from_vec_with_template(vec![k, n], u_data, placement)?,
            tensor_from_vec_with_template(vec![], vec![parity], placement)?,
        ])
    }

    fn lu_factor_2d(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<(TypedTensor<Self>, TypedTensor<i32>, TypedTensor<Self>)> {
        let (m, n) = matrix_dims(input, "lu_factor")?;
        let k = m.min(n);
        let mut lu = Mat::zeros(m, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(input.host_data()?),
            m,
            n,
        ));
        let mut perm = vec![0usize; m];
        let mut perm_inv = vec![0usize; m];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                m,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;
        let parity = if info.transposition_count % 2 == 0 {
            <$complex>::new(1.0, 0.0)
        } else {
            <$complex>::new(-1.0, 0.0)
        };
        let pivots = swap_sequence_from_permutation(&perm, k, "lu_factor")?;

        Ok((
            tensor_from_vec_with_template(vec![m, n], $vec_from_mat(_buffers, lu.as_ref())?, input.placement())?,
            tensor_from_vec_with_template(vec![k], pivots, input.placement())?,
            tensor_from_vec_with_template(vec![], vec![parity], input.placement())?,
        ))
    }

    fn full_piv_lu_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "full_piv_lu")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        Self::full_piv_lu_core(ctx, buffers, mat, n, input.placement())
    }

    fn full_piv_lu_core(
        ctx: &CpuExecutionContext<'_>,
        _buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        // Cast Self (Complex32/64) to $faer_complex (faer::c32/c64) for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                n,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let mut lu = Mat::zeros(n, n);
        lu.copy_from(mat);
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                n,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let info = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .0;

        let mut l_data = $matrix_from_predicate(lu.as_ref(), n, n, |row, col| row >= col)?;
        for i in 0..n {
            l_data[i + i * n] = <$complex>::new(1.0, 0.0);
        }
        let u_data = $matrix_from_predicate(lu.as_ref(), n, n, |row, col| row <= col)?;
        let one = <$complex>::new(1.0, 0.0);
        let parity = if info.transposition_count % 2 == 0 {
            one
        } else {
            <$complex>::new(-1.0, 0.0)
        };

        Ok(vec![
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&row_perm, one)?, placement)?,
            tensor_from_vec_with_template(vec![n, n], l_data, placement)?,
            tensor_from_vec_with_template(vec![n, n], u_data, placement)?,
            tensor_from_vec_with_template(vec![n, n], permutation_matrix(&col_perm, one)?, placement)?,
            tensor_from_vec_with_template(vec![], vec![parity], placement)?,
        ])
    }

    fn full_piv_lu_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "full_piv_lu_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(tenferro_tensor::Error::shape_mismatch("full_piv_lu_solve", vec![n], vec![b_rows]));
        }

        let mut lu = Mat::zeros(n, n);
        lu.copy_from(MatRef::from_column_major_slice(
            $to_faer_slice(a.host_data()?),
            n,
            n,
        ));
        let mut row_perm = vec![0usize; n];
        let mut row_perm_inv = vec![0usize; n];
        let mut col_perm = vec![0usize; n];
        let mut col_perm_inv = vec![0usize; n];
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, $faer_complex>(
                n,
                n,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        let (_, row_perm_ref, col_perm_ref) = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        );
        let max_diagonal = (0..n)
            .map(|i| {
                let value = lu[(i, i)];
                (value.re as f64).hypot(value.im as f64)
            })
            .fold(0.0, f64::max);
        for i in 0..n {
            let value = lu[(i, i)];
            if complex_pivot_is_effectively_singular(
                value.re as f64,
                value.im as f64,
                max_diagonal,
                <$real>::EPSILON as f64,
            ) {
                return Err(singular_matrix("full_piv_lu_solve"));
            }
        }

        let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data()?.len());
        rhs_data.extend_from_slice(b.host_data()?);
        let rhs = MatMut::from_column_major_slice_mut(
            $to_faer_slice_mut(&mut rhs_data),
            n,
            b_cols,
        );
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, $faer_complex>(
                n,
                b_cols,
                ctx.faer_parallelism(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_parallelism(),
                stack,
            );
        } else {
            faer::linalg::lu::full_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                rhs,
                ctx.faer_parallelism(),
                stack,
            );
        }
        tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b.placement())
    }

    fn solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        solve_from_views(ctx, buffers, a.as_view(), b.as_view(), transpose_a)
    }

    impl_solve_into_view!(
        $complex,
        $faer_complex,
        $to_faer_slice,
        $to_faer_slice_mut
    );

    fn triangular_solve_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        a: &TypedTensor<Self>,
        b: &TypedTensor<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<TypedTensor<Self>> {
        let n = square_matrix_dim(a, "triangular_solve")?;
        let (b_rows, b_cols) = matrix_dims(b, "triangular_solve")?;
        let a_mat = MatRef::from_column_major_slice($to_faer_slice(a.host_data()?), n, n);

        if left_side {
            if b_rows != n {
                return Err(tenferro_tensor::Error::shape_mismatch("triangular_solve", vec![n], vec![b_rows]));
            }
            let mut rhs_data = buffers.acquire_with_capacity::<Self>(b.host_data()?.len());
            rhs_data.extend_from_slice(b.host_data()?);
            let rhs = MatMut::from_column_major_slice_mut(
                $to_faer_slice_mut(&mut rhs_data),
                n,
                b_cols,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
            }
            Ok(tensor_from_vec_with_template(vec![n, b_cols], rhs_data, b.placement())?)
        } else {
            if b_cols != n {
                return Err(tenferro_tensor::Error::shape_mismatch("triangular_solve", vec![n], vec![b_cols]));
            }
            let nrhs = b_rows;
            let mut rhs_transposed = transpose_col_major_data(buffers, b.host_data()?, nrhs, n);
            let rhs = MatMut::from_column_major_slice_mut(
                $to_faer_slice_mut(&mut rhs_transposed),
                n,
                nrhs,
            );
            match (transpose_a, lower, unit_diagonal) {
                (false, true, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (false, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat.transpose(),
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, false) => {
                    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, true, true) => {
                    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, false) => {
                    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
                (true, false, true) => {
                    faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(
                        a_mat,
                        rhs,
                        ctx.faer_parallelism(),
                    );
                }
            }
            let result = transpose_col_major_data(buffers, &rhs_transposed, n, nrhs);
            <Self as PoolScalar>::pool_release(buffers, rhs_transposed);
            Ok(tensor_from_vec_with_template(vec![nrhs, n], result, b.placement())?)
        }
    }

    fn svd_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        full: bool,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "svd")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::svd_core(ctx, buffers, mat, m, n, full, input.placement())
    }

    fn svd_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let (m, n) = matrix_dims(input, "svd_values")?;
        let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()?), m, n);
        Self::svd_values_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn svd_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        // SAFETY: `mat` is a validated host view with an in-bounds shape/stride
        // span and aligned base; the `impl_complex_faer_casts` const assertions
        // prove the scalar layouts match for this faer reinterpretation.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                m,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let k = m.min(n);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<$faer_complex>(
            m,
            n,
            faer::linalg::svd::ComputeSvdVectors::No,
            faer::linalg::svd::ComputeSvdVectors::No,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            None,
            None,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd_values"))?;

        let col = s.as_ref().column_vector();
        let mut data = buffers.acquire_with_capacity::<$real>(col.nrows());
        for i in 0..col.nrows() {
            data.push(col[i].re);
        }
        tensor_from_vec_with_template(vec![k], data, placement)
    }

    fn qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let (m, n) = matrix_dims(input, "qr")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        Self::qr_core(ctx, buffers, mat, m, n, input.placement())
    }

    fn rank_revealing_qr_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
        options: crate::RankRevealingQrOptions,
    ) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<Self>> {
        let (m, n) = matrix_dims(input, "rank_revealing_qr")?;
        let k = m.min(n);
        let mat = Self::faer_mat_ref_compact(input.host_data()?, m, n);
        // SAFETY: the compile-time layout checks in impl_complex_faer_casts
        // prove Self and faer's complex scalar have identical representation.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr().cast::<$faer_complex>(),
                m,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<$faer_complex>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(mat);
        let mut coeff = Mat::zeros(block_size, k);
        let mut permutation = vec![0usize; n];
        let mut inverse_permutation = vec![0usize; n];
        // Following faer 0.24's public column-pivoted QR factor API; the
        // provider's forward permutation is the public gather convention.
        let mut mem = MemBuffer::new(
            faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, $faer_complex>(
                m,
                n,
                block_size,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        faer::linalg::qr::col_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            &mut permutation,
            &mut inverse_permutation,
            ctx.faer_parallelism(),
            MemStack::new(&mut mem),
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$faer_complex>(
                m,
                block_size,
                k,
            ),
        );
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_parallelism(),
            MemStack::new(&mut mem),
        );
        let rank = super::rank_revealing_qr::prefix_rank(
            (0..k).map(|index| {
                let value = qr[(index, index)];
                (value.re as f64).hypot(value.im as f64)
            }),
            options,
        )?;
        let permutation = permutation
            .into_iter()
            .map(|column| {
                i64::try_from(column).map_err(|_| {
                    invalid_config(
                        "rank_revealing_qr",
                        "column permutation exceeds i64 range",
                    )
                })
            })
            .collect::<tenferro_tensor::Result<Vec<_>>>()?;
        Ok(crate::RankRevealingQrResult {
            q: tensor_from_vec_with_template(
                vec![m, k],
                $vec_from_mat(buffers, q.as_ref())?,
                input.placement(),
            )?,
            r: tensor_from_vec_with_template(
                vec![k, n],
                $matrix_from_predicate(qr.as_ref(), k, n, |row, col| row <= col)?,
                input.placement(),
            )?,
            column_permutation: tensor_from_vec_with_template(
                vec![n],
                permutation,
                input.placement(),
            )?,
            rank: tensor_from_vec_with_template(vec![], vec![rank], input.placement())?,
        })
    }

    fn eigh_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        let n = square_matrix_dim(input, "eigh")?;
        let mat = Self::faer_mat_ref_compact(input.host_data()?, n, n);
        Self::eigh_core(ctx, buffers, mat, n, input.placement())
    }

    fn eigh_values_2d(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        input: &TypedTensor<Self>,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        let n = square_matrix_dim(input, "eigh_values")?;
        let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()?), n, n);
        Self::eigh_values_core(ctx, buffers, mat, n, input.placement())
    }

    fn eigh_values_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<TypedTensor<<Self as FaerLinalg>::Real>> {
        // SAFETY: `mat` is a validated host view with an in-bounds square span
        // and aligned base; the `impl_complex_faer_casts` const assertions prove
        // the scalar layouts match for this faer reinterpretation.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                n,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let mut values = Diag::zeros(n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<$faer_complex>(
            n,
            faer::linalg::evd::ComputeEigenvectors::No,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            None,
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh_values"))?;

        let col = values.as_ref().column_vector();
        let mut data = buffers.acquire_with_capacity::<$real>(col.nrows());
        for i in 0..col.nrows() {
            data.push(col[i].re);
        }
        tensor_from_vec_with_template(vec![n], data, placement)
    }

    fn faer_mat_ref_compact<'a>(data: &'a [Self], m: usize, n: usize) -> MatRef<'a, Self> {
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let faer_slice = $to_faer_slice(data);
        unsafe {
            MatRef::<Self>::from_raw_parts(
                faer_slice.as_ptr() as *const Self,
                m,
                n,
                1,
                m as isize,
            )
        }
    }

    unsafe fn faer_mat_ref_strided<'a>(
        base: *const Self,
        m: usize,
        n: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> MatRef<'a, Self> {
        // SAFETY: caller guarantees host placement, rank 2, non-negative strides, and 'a lifetime.
        unsafe { MatRef::<Self>::from_raw_parts(base, m, n, row_stride, col_stride) }
    }

    fn svd_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        full: bool,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        // Cast Self (Complex32/64) to $faer_complex (faer::c32/c64) for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                m,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let k = m.min(n);
        // Full mode returns the square unitary factors `U (m x m)` and
        // `V (n x n)`; thin mode keeps only the leading `k` vectors. The
        // singular-value count is `k` in both modes.
        let (u_cols, v_cols, vectors) = if full {
            (m, n, faer::linalg::svd::ComputeSvdVectors::Full)
        } else {
            (k, k, faer::linalg::svd::ComputeSvdVectors::Thin)
        };
        let mut u = Mat::zeros(m, u_cols);
        let mut v = Mat::zeros(n, v_cols);
        let mut s = Diag::zeros(k);
        let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<$faer_complex>(
            m,
            n,
            vectors,
            vectors,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s.as_mut(),
            Some(u.as_mut()),
            Some(v.as_mut()),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("svd"))?;

        let u = tensor_from_vec_with_template(
            vec![m, u_cols],
            $vec_from_mat(buffers, u.as_ref())?,
            placement,
        )?;
        let s = tensor_from_vec_with_template(
            vec![k],
            $vec_from_real_diag(buffers, s.as_ref()),
            placement,
        )?;
        let vt_len = checked_product("svd", "right singular vectors", &[v_cols, n])?;
        let mut vt_data = buffers.acquire_with_capacity::<Self>(vt_len);
        for j in 0..n {
            for i in 0..v_cols {
                vt_data.push(v[(j, i)].conj());
            }
        }
        let vt = tensor_from_vec_with_template(vec![v_cols, n], vt_data, placement)?;

        Ok(vec![u, s, vt])
    }

    fn qr_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        m: usize,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        // Cast Self to $faer_complex for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                m,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let k = m.min(n);
        let block_size =
            faer::linalg::qr::no_pivoting::factor::recommended_block_size::<$faer_complex>(m, n);
        let mut qr = Mat::zeros(m, n);
        qr.copy_from(mat);
        let mut coeff = Mat::zeros(block_size, k);
        let mut mem = MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<$faer_complex>(
                m,
                n,
                block_size,
                ctx.faer_parallelism(),
                Default::default(),
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            qr.as_mut(),
            coeff.as_mut(),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        );
        let mut q = Mat::identity(m, k);
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<$faer_complex>(
                m,
                block_size,
                k,
            ),
        );
        let stack = MemStack::new(&mut mem);
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            qr.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q.as_mut(),
            ctx.faer_parallelism(),
            stack,
        );
        let q = tensor_from_vec_with_template(
            vec![m, k],
            $vec_from_mat(buffers, q.as_ref())?,
            placement,
        )?;
        let r = tensor_from_vec_with_template(
            vec![k, n],
            $matrix_from_predicate(qr.as_ref(), k, n, |row, col| row <= col)?,
            placement,
        )?;

        Ok(vec![q, r])
    }

    fn eigh_core(
        ctx: &CpuExecutionContext<'_>,
        buffers: &mut BufferPool,
        mat: MatRef<'_, Self>,
        n: usize,
        placement: &tenferro_tensor::Placement,
    ) -> tenferro_tensor::Result<Vec<TypedTensor<Self>>> {
        // Cast Self to $faer_complex for faer calls.
        // SAFETY: layout identity guaranteed by impl_complex_faer_casts const asserts.
        let mat: MatRef<'_, $faer_complex> = unsafe {
            MatRef::from_raw_parts(
                mat.as_ptr() as *const $faer_complex,
                n,
                n,
                mat.row_stride(),
                mat.col_stride(),
            )
        };
        let mut values = Diag::zeros(n);
        let mut vectors = Mat::zeros(n, n);
        let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<$faer_complex>(
            n,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            ctx.faer_parallelism(),
            Default::default(),
        ));
        let stack = MemStack::new(&mut mem);
        faer::linalg::evd::self_adjoint_evd(
            mat,
            values.as_mut(),
            Some(vectors.as_mut()),
            ctx.faer_parallelism(),
            stack,
            Default::default(),
        )
        .map_err(|_| decomposition_failed("eigh"))?;

        let values = tensor_from_vec_with_template(
            vec![n],
            $vec_from_real_diag(buffers, values.as_ref()),
            placement,
        )?;
        let vectors = tensor_from_vec_with_template(
            vec![n, n],
            $vec_from_mat(buffers, vectors.as_ref())?,
            placement,
        )?;

        Ok(vec![values, vectors])
    }
        }
    };
}

impl_faer_linalg_for_complex!(
    Complex32,
    f32,
    faer::c32,
    complex32_to_faer_slice,
    complex32_to_faer_slice_mut,
    complex32_vec_from_real_diag,
    complex32_vec_from_diag,
    complex32_vec_from_mat,
    complex32_matrix_from_predicate
);
impl_faer_linalg_for_complex!(
    Complex64,
    f64,
    faer::c64,
    complex64_to_faer_slice,
    complex64_to_faer_slice_mut,
    complex64_vec_from_real_diag,
    complex64_vec_from_diag,
    complex64_vec_from_mat,
    complex64_matrix_from_predicate
);

pub(crate) fn cholesky<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch(input, "cholesky")?;
        return tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            Vec::new(),
            input.placement(),
        );
    }
    batched_single("cholesky", buffers, input, 2, |buffers, batch| {
        T::cholesky_2d(ctx, buffers, batch)
    })
}

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
    let mat = T::faer_mat_ref_compact(input, n, n);
    T::cholesky_core(ctx, buffers, mat, n)
}

pub(crate) fn lu<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "lu")?;
        let k = m.min(n);
        let parity_elements = checked_product("lu", "batch shape", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, m, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); parity_elements],
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("lu", buffers, input, 2, |buffers, batch| {
        T::lu_2d(ctx, buffers, batch)
    })
}

pub(crate) fn lu_factor<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "lu_factor")?;
        let k = m.min(n);
        let parity_len = batch_count("lu_factor", batch_shape)?;
        return Ok((
            tensor_from_vec_with_template(input.shape().to_vec(), Vec::new(), input.placement())?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); parity_len],
                input.placement(),
            )?,
        ));
    }

    let (m, n, batch_shape) = matrix_core_and_batch(input, "lu_factor")?;
    if batch_shape.is_empty() {
        return T::lu_factor_2d(ctx, buffers, input);
    }

    let k = m.min(n);
    let matrix_len = checked_product("lu_factor", "matrix shape", &[m, n])?;
    let batch_total = batch_count("lu_factor", batch_shape)?;
    let lu_len = checked_repeated_len("lu_factor", "packed LU", matrix_len, batch_total)?;
    let pivot_len = checked_repeated_len("lu_factor", "pivots", k, batch_total)?;
    let mut lu_data = buffers.acquire_with_capacity::<T>(lu_len);
    let mut pivot_data = Vec::with_capacity(pivot_len);
    let mut parity_data = buffers.acquire_with_capacity::<T>(batch_total);

    let first_range = checked_slice_range("lu_factor", 0, matrix_len)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        vec![m, n],
        &input.host_data()?[first_range],
        input.placement(),
    )?;

    for batch in 0..batch_total {
        if batch > 0 {
            let range = checked_slice_range("lu_factor", batch, matrix_len)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let (packed, pivots, parity) = T::lu_factor_2d(ctx, buffers, &batch_input)?;
        lu_data.extend_from_slice(packed.host_data()?);
        pivot_data.extend_from_slice(pivots.host_data()?);
        parity_data.extend_from_slice(parity.host_data()?);
    }

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

pub(crate) fn full_piv_lu<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch(input, "full_piv_lu")?;
        let parity_elements = checked_product("full_piv_lu", "batch shape", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::parity_one(); parity_elements],
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("full_piv_lu", buffers, input, 2, |buffers, batch| {
        T::full_piv_lu_2d(ctx, buffers, batch)
    })
}

pub(crate) fn full_piv_lu_solve<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch(a, "full_piv_lu_solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch(b, "full_piv_lu_solve")?;
        if b_rows != n {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "full_piv_lu_solve",
                vec![n],
                vec![b_rows],
            ));
        }
        if a_batch_shape != b_batch_shape {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "full_piv_lu_solve",
                a_batch_shape.to_vec(),
                b_batch_shape.to_vec(),
            ));
        }
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    batched_binary_result("full_piv_lu_solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::full_piv_lu_solve_2d(ctx, buffers, a, b, transpose_a)
    })
}

pub(crate) fn solve<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch(a, "solve")?;
        let (b_rows, _, b_batch_shape) = matrix_core_and_batch(b, "solve")?;
        if b_rows != n {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "solve",
                vec![n],
                vec![b_rows],
            ));
        }
        if a_batch_shape != b_batch_shape {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "solve",
                a_batch_shape.to_vec(),
                b_batch_shape.to_vec(),
            ));
        }
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    batched_binary_result("solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::solve_2d(ctx, buffers, a, b, transpose_a)
    })
}

/// Solve a single matrix system directly into a positive column-major output
/// view. The destination is populated only after factorization and singularity
/// checks have completed, so validation/provider failures before execution do
/// not partially modify the caller-owned buffer.
pub(crate) fn solve_into<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    a: TypedTensorView<'_, T>,
    b: TypedTensorView<'_, T>,
    out: &mut TypedTensorViewMut<'_, T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<()> {
    T::solve_2d_into_view(ctx, buffers, a, b, out, transpose_a)
}

fn same_slice<T>(data: &[T]) -> &[T] {
    data
}

fn same_slice_mut<T>(data: &mut [T]) -> &mut [T] {
    data
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
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        let (n, a_batch_shape) = square_core_and_batch(a, "triangular_solve")?;
        let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch(b, "triangular_solve")?;
        let rhs_core_dim = if left_side { b_rows } else { b_cols };
        if rhs_core_dim != n {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "triangular_solve",
                vec![n],
                vec![rhs_core_dim],
            ));
        }
        if a_batch_shape != b_batch_shape {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "triangular_solve",
                a_batch_shape.to_vec(),
                b_batch_shape.to_vec(),
            ));
        }
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b.placement());
    }
    batched_binary_result("triangular_solve", buffers, a, b, 2, 2, |buffers, a, b| {
        T::triangular_solve_2d(
            ctx,
            buffers,
            a,
            b,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )
    })
}

pub(crate) fn svd<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "svd")?;
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("svd", buffers, input, 2, |buffers, batch| {
        T::svd_2d(ctx, buffers, batch, false)
    })
}

pub(crate) fn svd_full<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "svd_full")?;
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, m, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("svd_full", buffers, input, 2, |buffers, batch| {
        T::svd_2d(ctx, buffers, batch, true)
    })
}

pub(crate) fn svd_values<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<<T as FaerLinalg>::Real>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "svd_values")?;
        let k = m.min(n);
        return tensor_from_vec_with_template(
            vector_with_batch_shape(k, batch_shape),
            Vec::new(),
            input.placement(),
        );
    }
    let mut outputs =
        batched_multi_convert_result("svd_values", buffers, input, 2, |buffers, batch| {
            T::svd_values_2d(ctx, buffers, batch).map(|values| vec![values])
        })?;
    Ok(outputs.remove(0))
}

pub(crate) fn qr<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch(input, "qr")?;
        let k = m.min(n);
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("qr", buffers, input, 2, |buffers, batch| {
        T::qr_2d(ctx, buffers, batch)
    })
}

pub(crate) fn rank_revealing_qr<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    options: crate::RankRevealingQrOptions,
) -> tenferro_tensor::Result<super::rank_revealing_qr::TypedRrqr<T>> {
    crate::rank_revealing_qr::validate_rank_revealing_qr_options("rank_revealing_qr", options)?;
    super::rank_revealing_qr::batched(buffers, input, |buffers, batch| {
        let data = batch.host_data()?;
        if data.iter().any(|&value| !value.is_finite()) {
            return Err(crate::error::into_tensor_error(
                "rank_revealing_qr",
                crate::Error::NonFinite {
                    op: "rank_revealing_qr",
                    role: "input",
                },
            ));
        }
        if data.iter().all(|&value| value.magnitude() == 0.0) {
            return super::rank_revealing_qr::zero_matrix_result(batch, T::one());
        }
        T::rank_revealing_qr_2d(ctx, buffers, batch, options)
    })
}

pub(crate) fn eigh<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch(input, "eigh")?;
        return Ok(vec![
            tensor_from_vec_with_template(
                vector_with_batch_shape(n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input.placement(),
            )?,
        ]);
    }
    batched_multi_result("eigh", buffers, input, 2, |buffers, batch| {
        T::eigh_2d(ctx, buffers, batch)
    })
}

pub(crate) fn eigh_values<T: FaerLinalg>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<<T as FaerLinalg>::Real>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch(input, "eigh_values")?;
        return tensor_from_vec_with_template(
            vector_with_batch_shape(n, batch_shape),
            Vec::new(),
            input.placement(),
        );
    }
    let mut outputs =
        batched_multi_convert_result("eigh_values", buffers, input, 2, |buffers, batch| {
            T::eigh_values_2d(ctx, buffers, batch).map(|values| vec![values])
        })?;
    Ok(outputs.remove(0))
}

/// Predicate: can this view be fed to faer as a strided MatRef?
pub(crate) fn faer_strided_ok<T: 'static>(view: &TypedTensorView<'_, T>) -> bool {
    view.backend_buffer().is_none()
        && view.shape().len() == 2
        && view.strides().iter().all(|&s| s >= 0)
}

/// Get the host base pointer at the view's offset.
fn host_base_ptr<T: 'static>(view: &TypedTensorView<'_, T>) -> tenferro_tensor::Result<*const T> {
    let storage = view.host_storage()?;
    let offset = view.offset();
    if offset < 0 {
        return Err(tenferro_tensor::Error::runtime_state(
            "faer_view",
            "view offset is negative",
        ));
    }
    let offset_usize = offset as usize;
    if !storage.is_empty() && offset_usize >= storage.len() {
        return Err(tenferro_tensor::Error::runtime_state(
            "faer_view",
            "view offset is outside host buffer",
        ));
    }
    // SAFETY: offset is validated above.
    Ok(unsafe { storage.as_ptr().offset(offset) })
}

fn output_storage_slice_mut<'view, T: 'static>(
    view: &'view mut TypedTensorViewMut<'_, T>,
    rows: usize,
    cols: usize,
    col_stride: isize,
    op: &'static str,
) -> tenferro_tensor::Result<&'view mut [T]> {
    let col_stride = usize::try_from(col_stride).map_err(|_| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "out",
            "output column stride must be non-negative",
        )
    })?;
    let span = cols
        .checked_sub(1)
        .and_then(|last_col| last_col.checked_mul(col_stride))
        .and_then(|last_offset| last_offset.checked_add(rows))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(op, "out", "output view span overflows usize")
        })?;
    let offset = usize::try_from(view.offset())
        .map_err(|_| tenferro_tensor::Error::runtime_state(op, "output view offset is negative"))?;
    let end = offset.checked_add(span).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "out",
            "output view end offset overflows usize",
        )
    })?;
    view.host_storage_mut()?
        .get_mut(offset..end)
        .ok_or_else(|| {
            tenferro_tensor::Error::runtime_state(
                op,
                "output view does not contain the requested Faer span",
            )
        })
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

fn rhs_matrix_dims_view<T: 'static>(
    view: &TypedTensorView<'_, T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    match view.shape() {
        [rows] => Ok((*rows, 1)),
        _ => matrix_dims_view(view, op),
    }
}

fn copy_rhs_view_into<T: Copy + 'static>(
    src: &TypedTensorView<'_, T>,
    dst: &mut TypedTensorViewMut<'_, T>,
    rows: usize,
    cols: usize,
    op: &'static str,
) -> tenferro_tensor::Result<()> {
    if dst.shape().len() == 1 {
        for row in 0..rows {
            let value = src.get(&[row]).ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
            })?;
            let target = dst.get_mut(&[row]).ok_or_else(|| {
                tenferro_tensor::Error::runtime_state(op, "output view is not host-addressable")
            })?;
            *target = *value;
        }
    } else {
        for col in 0..cols {
            for row in 0..rows {
                let value = src.get(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "RHS view is not host-addressable")
                })?;
                let target = dst.get_mut(&[row, col]).ok_or_else(|| {
                    tenferro_tensor::Error::runtime_state(op, "output view is not host-addressable")
                })?;
                *target = *value;
            }
        }
    }
    Ok(())
}

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

pub(crate) fn svd_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims_view(&view, "svd")?;
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::svd_core(ctx, buffers, mat, m, n, false, &placement)
}

pub(crate) fn svd_values_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<<T as FaerLinalg>::Real>> {
    let (m, n) = matrix_dims_view(&view, "svd_values")?;
    let placement = view.placement().clone();
    if m == 0 || n == 0 {
        return tensor_from_vec_with_template(vec![m.min(n)], Vec::new(), &placement);
    }
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::svd_values_core(ctx, buffers, mat, m, n, &placement)
}

pub(crate) fn qr_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims_view(&view, "qr")?;
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::qr_core(ctx, buffers, mat, m, n, &placement)
}

pub(crate) fn eigh_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims_view(&view, "eigh")?;
    if m != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "eigh",
            vec![m],
            vec![n],
        ));
    }
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::eigh_core(ctx, buffers, mat, m, &placement)
}

pub(crate) fn eigh_values_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<<T as FaerLinalg>::Real>> {
    let (m, n) = matrix_dims_view(&view, "eigh_values")?;
    if m != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "eigh_values",
            vec![m],
            vec![n],
        ));
    }
    let placement = view.placement().clone();
    if n == 0 {
        return tensor_from_vec_with_template(vec![0], Vec::new(), &placement);
    }
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, n, n, view.strides()[0], view.strides()[1]) };
    T::eigh_values_core(ctx, buffers, mat, n, &placement)
}

pub(crate) fn cholesky_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let (m, n) = matrix_dims_view(&view, "cholesky")?;
    if m != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "cholesky",
            vec![m],
            vec![n],
        ));
    }
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    tensor_from_vec_with_template(
        vec![m, m],
        T::cholesky_core(ctx, buffers, mat, m)?,
        &placement,
    )
}

pub(crate) fn lu_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims_view(&view, "lu")?;
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::lu_core(ctx, buffers, mat, m, n, &placement)
}

pub(crate) fn full_piv_lu_view<T: FaerLinalg + 'static>(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    view: TypedTensorView<'_, T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims_view(&view, "full_piv_lu")?;
    if m != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "full_piv_lu",
            vec![m],
            vec![n],
        ));
    }
    let placement = view.placement().clone();
    let base = host_base_ptr(&view)?;
    // SAFETY: `TypedTensorView` construction validates the shape/stride span and
    // offset against its host allocation; `faer_strided_ok` additionally proves
    // host placement, rank 2, and non-negative strides. `host_base_ptr` returns
    // the aligned non-null element pointer at that validated offset.
    let mat = unsafe { T::faer_mat_ref_strided(base, m, n, view.strides()[0], view.strides()[1]) };
    T::full_piv_lu_core(ctx, buffers, mat, m, &placement)
}

macro_rules! impl_eig_real_2d {
    ($name:ident, $real:ty, $complex:ty, $real_eig_to_complex_outputs:ident) => {
        fn $name(
            ctx: &CpuExecutionContext<'_>,
            buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> tenferro_tensor::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let mat = MatRef::from_column_major_slice(input.host_data()?, n, n);
            let mut u_real = Mat::zeros(n, n);
            let mut s_re = Diag::zeros(n);
            let mut s_im = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$real>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::Yes,
                ctx.faer_parallelism(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_real(
                mat,
                s_re.as_mut(),
                s_im.as_mut(),
                None,
                Some(u_real.as_mut()),
                ctx.faer_parallelism(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig"))?;
            let (u, s) = $real_eig_to_complex_outputs(
                buffers,
                u_real.as_ref(),
                s_re.as_ref(),
                s_im.as_ref(),
            )?;

            Ok(vec![
                tensor_from_vec_with_template(vec![n], s, input.placement())?,
                tensor_from_vec_with_template(vec![n, n], u, input.placement())?,
            ])
        }
    };
}

macro_rules! impl_eig_values_real_2d {
    ($name:ident, $real:ty, $complex:ty, $real_eig_to_complex_values:ident) => {
        fn $name(
            ctx: &CpuExecutionContext<'_>,
            buffers: &mut BufferPool,
            input: &TypedTensor<$real>,
        ) -> tenferro_tensor::Result<TypedTensor<$complex>> {
            let n = square_matrix_dim(input, "eig_values")?;
            let mat = MatRef::from_column_major_slice(input.host_data()?, n, n);
            let mut s_re = Diag::zeros(n);
            let mut s_im = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$real>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::No,
                ctx.faer_parallelism(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_real(
                mat,
                s_re.as_mut(),
                s_im.as_mut(),
                None,
                None,
                ctx.faer_parallelism(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig_values"))?;
            let s = $real_eig_to_complex_values(buffers, s_re.as_ref(), s_im.as_ref());

            tensor_from_vec_with_template(vec![n], s, input.placement())
        }
    };
}

macro_rules! impl_eig_complex_2d {
    (
        $name:ident,
        $complex:ty,
        $faer_complex:ty,
        $to_faer_slice:ident,
        $vec_from_diag:ident,
        $vec_from_mat:ident
    ) => {
        fn $name(
            ctx: &CpuExecutionContext<'_>,
            buffers: &mut BufferPool,
            input: &TypedTensor<$complex>,
        ) -> tenferro_tensor::Result<Vec<TypedTensor<$complex>>> {
            let n = square_matrix_dim(input, "eig")?;
            let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()?), n, n);
            let mut u = Mat::zeros(n, n);
            let mut s = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$faer_complex>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::Yes,
                ctx.faer_parallelism(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_cplx(
                mat,
                s.as_mut(),
                None,
                Some(u.as_mut()),
                ctx.faer_parallelism(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig"))?;

            Ok(vec![
                tensor_from_vec_with_template(
                    vec![n],
                    $vec_from_diag(buffers, s.as_ref()),
                    input.placement(),
                )?,
                tensor_from_vec_with_template(
                    vec![n, n],
                    $vec_from_mat(buffers, u.as_ref())?,
                    input.placement(),
                )?,
            ])
        }
    };
}

macro_rules! impl_eig_values_complex_2d {
    (
        $name:ident,
        $complex:ty,
        $faer_complex:ty,
        $to_faer_slice:ident,
        $vec_from_diag:ident
    ) => {
        fn $name(
            ctx: &CpuExecutionContext<'_>,
            buffers: &mut BufferPool,
            input: &TypedTensor<$complex>,
        ) -> tenferro_tensor::Result<TypedTensor<$complex>> {
            let n = square_matrix_dim(input, "eig_values")?;
            let mat = MatRef::from_column_major_slice($to_faer_slice(input.host_data()?), n, n);
            let mut s = Diag::zeros(n);
            let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$faer_complex>(
                n,
                faer::linalg::evd::ComputeEigenvectors::No,
                faer::linalg::evd::ComputeEigenvectors::No,
                ctx.faer_parallelism(),
                Default::default(),
            ));
            let stack = MemStack::new(&mut mem);
            faer::linalg::evd::evd_cplx(
                mat,
                s.as_mut(),
                None,
                None,
                ctx.faer_parallelism(),
                stack,
                Default::default(),
            )
            .map_err(|_| decomposition_failed("eig_values"))?;

            tensor_from_vec_with_template(
                vec![n],
                $vec_from_diag(buffers, s.as_ref()),
                input.placement(),
            )
        }
    };
}

impl_eig_real_2d!(eig_real32_2d, f32, Complex32, real32_eig_to_complex_outputs);
impl_eig_real_2d!(eig_real64_2d, f64, Complex64, real64_eig_to_complex_outputs);
impl_eig_values_real_2d!(
    eig_values_real32_2d,
    f32,
    Complex32,
    real32_eig_to_complex_values
);
impl_eig_values_real_2d!(
    eig_values_real64_2d,
    f64,
    Complex64,
    real64_eig_to_complex_values
);
impl_eig_complex_2d!(
    eig_complex32_2d,
    Complex32,
    faer::c32,
    complex32_to_faer_slice,
    complex32_vec_from_diag,
    complex32_vec_from_mat
);
impl_eig_complex_2d!(
    eig_complex64_2d,
    Complex64,
    faer::c64,
    complex64_to_faer_slice,
    complex64_vec_from_diag,
    complex64_vec_from_mat
);
impl_eig_values_complex_2d!(
    eig_values_complex32_2d,
    Complex32,
    faer::c32,
    complex32_to_faer_slice,
    complex32_vec_from_diag
);
impl_eig_values_complex_2d!(
    eig_values_complex64_2d,
    Complex64,
    faer::c64,
    complex64_to_faer_slice,
    complex64_vec_from_diag
);

pub(crate) fn eig(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) = split_shape_core_and_batch(input.shape(), 2, "eig")?;
        let n = matrix_shape[0];
        if matrix_shape[1] != n {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "eig",
                vec![n],
                vec![matrix_shape[1]],
            ));
        }
        let value_shape = vector_with_batch_shape(n, batch_shape);
        let vector_shape = matrix_with_batch_shape(n, n, batch_shape);
        return match input {
            Tensor::F32(_) | Tensor::C32(_) => Ok(vec![
                Tensor::C32(TypedTensor::from_vec_col_major(value_shape, Vec::new())?),
                Tensor::C32(TypedTensor::from_vec_col_major(vector_shape, Vec::new())?),
            ]),
            Tensor::F64(_) | Tensor::C64(_) => Ok(vec![
                Tensor::C64(TypedTensor::from_vec_col_major(value_shape, Vec::new())?),
                Tensor::C64(TypedTensor::from_vec_col_major(vector_shape, Vec::new())?),
            ]),
            _ => Err(crate::error::unsupported_dtype("eig", input.dtype())),
        };
    }

    match input {
        Tensor::F32(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_real32_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C32)
                .collect(),
            )
        }
        Tensor::F64(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_real64_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C64)
                .collect(),
            )
        }
        Tensor::C32(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_complex32_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C32)
                .collect(),
            )
        }
        Tensor::C64(t) => {
            Ok(
                batched_multi_convert_result("eig", buffers, t, 2, |buffers, batch| {
                    eig_complex64_2d(ctx, buffers, batch)
                })?
                .into_iter()
                .map(Tensor::C64)
                .collect(),
            )
        }
        _ => Err(crate::error::unsupported_dtype("eig", input.dtype())),
    }
}

pub(crate) fn eig_values(
    ctx: &CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    if has_zero_dim(input.shape()) {
        let (matrix_shape, batch_shape) =
            split_shape_core_and_batch(input.shape(), 2, "eig_values")?;
        let n = matrix_shape[0];
        if matrix_shape[1] != n {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "eig_values",
                vec![n],
                vec![matrix_shape[1]],
            ));
        }
        let value_shape = vector_with_batch_shape(n, batch_shape);
        return match input {
            Tensor::F32(_) | Tensor::C32(_) => Ok(Tensor::C32(TypedTensor::from_vec_col_major(
                value_shape,
                Vec::new(),
            )?)),
            Tensor::F64(_) | Tensor::C64(_) => Ok(Tensor::C64(TypedTensor::from_vec_col_major(
                value_shape,
                Vec::new(),
            )?)),
            _ => Err(crate::error::unsupported_dtype("eig_values", input.dtype())),
        };
    }

    match input {
        Tensor::F32(t) => {
            let mut outputs =
                batched_multi_convert_result("eig_values", buffers, t, 2, |buffers, batch| {
                    eig_values_real32_2d(ctx, buffers, batch).map(|values| vec![values])
                })?;
            Ok(Tensor::C32(outputs.remove(0)))
        }
        Tensor::F64(t) => {
            let mut outputs =
                batched_multi_convert_result("eig_values", buffers, t, 2, |buffers, batch| {
                    eig_values_real64_2d(ctx, buffers, batch).map(|values| vec![values])
                })?;
            Ok(Tensor::C64(outputs.remove(0)))
        }
        Tensor::C32(t) => {
            let mut outputs =
                batched_multi_convert_result("eig_values", buffers, t, 2, |buffers, batch| {
                    eig_values_complex32_2d(ctx, buffers, batch).map(|values| vec![values])
                })?;
            Ok(Tensor::C32(outputs.remove(0)))
        }
        Tensor::C64(t) => {
            let mut outputs =
                batched_multi_convert_result("eig_values", buffers, t, 2, |buffers, batch| {
                    eig_values_complex64_2d(ctx, buffers, batch).map(|values| vec![values])
                })?;
            Ok(Tensor::C64(outputs.remove(0)))
        }
        _ => Err(crate::error::unsupported_dtype("eig_values", input.dtype())),
    }
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

#[cfg(test)]
#[path = "faer_linalg/tests.rs"]
mod tests;
