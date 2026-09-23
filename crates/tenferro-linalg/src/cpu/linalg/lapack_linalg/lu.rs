use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, batched_multi, check_lapack_info, checked_product, dim_i32, has_zero_dim,
    leading_upper_triangle_from_lapack, matrix_core_and_batch_result, matrix_dims,
    matrix_with_batch_shape, pooled_copy, pooled_zeroed, release_scratch,
    tensor_from_vec_with_template, vector_with_batch_shape,
};

pub(crate) trait LapackLu: Clone + Copy + Default + PoolScalar {
    fn one() -> Self;
    fn negative_one() -> Self;
    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
}

impl LapackLu for f64 {
    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate `m`, `n`, and `lda`, provide a mutable
        // column-major `lda x n` matrix, `min(m, n)` pivots, and live `info`.
        unsafe {
            lapack::dgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

impl LapackLu for f32 {
    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate `m`, `n`, and `lda`, provide a mutable
        // column-major `lda x n` matrix, `min(m, n)` pivots, and live `info`.
        unsafe {
            lapack::sgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

impl LapackLu for Complex32 {
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex32::new(-1.0, 0.0)
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate `m`, `n`, and `lda`, provide a mutable
        // column-major `lda x n` matrix, `min(m, n)` pivots, and live `info`.
        unsafe {
            lapack::cgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

impl LapackLu for Complex64 {
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex64::new(-1.0, 0.0)
    }

    fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32) {
        // SAFETY: callers validate `m`, `n`, and `lda`, provide a mutable
        // column-major `lda x n` matrix, `min(m, n)` pivots, and live `info`.
        unsafe {
            lapack::zgetrf(m, n, data, lda, ipiv, info);
        }
    }
}

fn lu_2d<T: LapackLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims(input, "lu")?;
    let k = m.min(n);
    let m_i32 = dim_i32(m, "lu")?;
    let n_i32 = dim_i32(n, "lu")?;
    let mut lu = pooled_copy(buffers, input.host_data()?);
    let mut ipiv = pooled_zeroed::<i32>(buffers, k);
    let mut info = 0;
    T::getrf(m_i32, n_i32, &mut lu, m_i32, &mut ipiv, &mut info);
    check_lapack_info("lu", "getrf", info.min(0))?;

    let mut permutation: Vec<usize> = (0..m).collect();
    let mut swap_count = 0usize;
    for (idx, &pivot_one_based) in ipiv.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) => pivot,
            Err(_) => {
                return Err(tenferro_tensor::Error::Internal(
                    "LAPACK getrf returned an invalid pivot index".to_string(),
                ));
            }
        };
        if pivot >= m {
            return Err(tenferro_tensor::Error::Internal(
                "LAPACK getrf returned an out-of-bounds pivot index".to_string(),
            ));
        }
        if pivot != idx {
            permutation.swap(idx, pivot);
            swap_count += 1;
        }
    }

    let p_len = checked_product("lu", "permutation matrix", &[m, m])?;
    let mut p_data = pooled_zeroed::<T>(buffers, p_len);
    for (row, &source_row) in permutation.iter().enumerate() {
        p_data[row + source_row * m] = T::one();
    }
    let parity = if swap_count.is_multiple_of(2) {
        T::one()
    } else {
        T::negative_one()
    };

    let l_len = checked_product("lu", "lower factor", &[m, k])?;
    let mut l_data = pooled_zeroed::<T>(buffers, l_len);
    for col in 0..k {
        for row in col..m {
            l_data[row + col * m] = lu[row + col * m];
        }
        l_data[col + col * m] = T::one();
    }
    let u_data = leading_upper_triangle_from_lapack(&lu, m, k, n)?;
    release_scratch(buffers, lu);
    release_scratch(buffers, ipiv);

    Ok(vec![
        tensor_from_vec_with_template(vec![m, m], p_data, input)?,
        tensor_from_vec_with_template(vec![m, k], l_data, input)?,
        tensor_from_vec_with_template(vec![k, n], u_data, input)?,
        tensor_from_vec_with_template(vec![], vec![parity], input)?,
    ])
}

pub(crate) fn lu<T: LapackLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch_result(input, "lu")?;
        let k = m.min(n);
        let parity_elements = batch_element_count("lu", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, m, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::one(); parity_elements],
                input,
            )?,
        ]);
    }
    batched_multi("lu", buffers, input, lu_2d)
}

pub(crate) fn lu_factor<T: LapackLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    if has_zero_dim(input.shape()) {
        let (m, n, batch_shape) = matrix_core_and_batch_result(input, "lu_factor")?;
        let k = m.min(n);
        let parity_elements = batch_element_count("lu_factor", batch_shape)?;
        return Ok((
            tensor_from_vec_with_template(input.shape().to_vec(), Vec::new(), input)?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::one(); parity_elements],
                input,
            )?,
        ));
    }

    let (m, n, batch_shape) = matrix_core_and_batch_result(input, "lu_factor")?;
    let k = m.min(n);
    let batch_total = batch_element_count("lu_factor", batch_shape)?;
    let pivot_len = checked_product("lu_factor", "pivot output", &[k, batch_total])?;
    let mut lu_data = pooled_copy(buffers, input.host_data()?);
    let mut pivot_data = pooled_zeroed::<i32>(buffers, pivot_len);
    let mut parity_data = buffers.acquire_with_capacity::<T>(batch_total);
    parity_data.resize(batch_total, T::one());
    lu_factor_batched_in_place(
        "lu_factor",
        m,
        n,
        &mut lu_data,
        &mut pivot_data,
        &mut parity_data,
    )?;

    Ok((
        tensor_from_vec_with_template(input.shape().to_vec(), lu_data, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), pivot_data, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity_data, input)?,
    ))
}

/// Factor every column-major `m x n` matrix of `lu_data` in place with `getrf`.
///
/// `pivot_data` receives `min(m, n)` one-based LAPACK pivots per matrix and
/// `parity_data` one permutation parity (`+1` or `-1`) per matrix. Exactly
/// singular matrices are not an error here: `getrf` reports them through a
/// positive `info`, and the packed factors stay valid for callers that check
/// the `U` diagonal themselves.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` when a dimension exceeds the LAPACK `i32`
/// range, when the buffer lengths do not describe the same batch, or when
/// LAPACK reports an illegal argument.
pub(crate) fn lu_factor_batched_in_place<T: LapackLu>(
    op: &'static str,
    m: usize,
    n: usize,
    lu_data: &mut [T],
    pivot_data: &mut [i32],
    parity_data: &mut [T],
) -> tenferro_tensor::Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch_total = parity_data.len();
    if lu_data.len() != checked_product(op, "packed LU output", &[matrix_len, batch_total])?
        || pivot_data.len() != checked_product(op, "pivot output", &[k, batch_total])?
    {
        return Err(tenferro_tensor::Error::Internal(format!(
            "{op}: packed LU, pivot, and parity buffers describe different batches"
        )));
    }
    if batch_total == 0 || matrix_len == 0 {
        return Ok(());
    }
    let m_i32 = dim_i32(m, op)?;
    let n_i32 = dim_i32(n, op)?;
    // INVARIANT: the three buffers were checked above to hold exactly
    // `batch_total` matrices, pivot vectors, and parities, and the early
    // return guarantees `matrix_len > 0` and hence `k > 0`, so the chunk
    // iterators stay in lockstep. The serial loop is intentional: the LAPACK
    // provider owns any threading inside `getrf`, and one call per matrix
    // writes straight into the caller's output buffers without scratch.
    for ((matrix, ipiv), parity) in lu_data
        .chunks_exact_mut(matrix_len)
        .zip(pivot_data.chunks_exact_mut(k))
        .zip(parity_data.iter_mut())
    {
        let mut info = 0;
        T::getrf(m_i32, n_i32, matrix, m_i32, ipiv, &mut info);
        check_lapack_info(op, "getrf", info.min(0))?;
        let swap_count = ipiv
            .iter()
            .enumerate()
            .filter(|(idx, pivot_one_based)| **pivot_one_based != (*idx as i32 + 1))
            .count();
        *parity = if swap_count.is_multiple_of(2) {
            T::one()
        } else {
            T::negative_one()
        };
    }
    Ok(())
}
