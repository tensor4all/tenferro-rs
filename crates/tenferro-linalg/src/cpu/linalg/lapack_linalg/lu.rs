use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, batched_multi, check_lapack_info, checked_product, checked_slice_range,
    dim_i32, has_zero_dim, leading_upper_triangle_from_lapack, matrix_core_and_batch_result,
    matrix_dims, matrix_with_batch_shape, refill_tensor_from_slice,
    tensor_from_pooled_slice_with_template, tensor_from_vec_with_template, vector_with_batch_shape,
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
    _buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (m, n) = matrix_dims(input, "lu")?;
    let k = m.min(n);
    let m_i32 = dim_i32(m, "lu")?;
    let n_i32 = dim_i32(n, "lu")?;
    let mut lu = input.host_data()?.to_vec();
    let mut ipiv = vec![0_i32; k];
    let mut info = 0;
    T::getrf(m_i32, n_i32, &mut lu, m_i32, &mut ipiv, &mut info);
    check_lapack_info("lu", "getrf", info.min(0))?;

    let mut permutation: Vec<usize> = (0..m).collect();
    let mut swap_count = 0usize;
    for (idx, &pivot_one_based) in ipiv.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) => pivot,
            Err(_) => {
                return Err(tenferro_tensor::Error::backend_failure(
                    "lu",
                    "LAPACK getrf returned invalid pivot index",
                ));
            }
        };
        if pivot >= m {
            return Err(tenferro_tensor::Error::backend_failure(
                "lu",
                "LAPACK getrf returned out-of-bounds pivot index",
            ));
        }
        if pivot != idx {
            permutation.swap(idx, pivot);
            swap_count += 1;
        }
    }

    let mut p_data = vec![T::default(); m * m];
    for (row, &source_row) in permutation.iter().enumerate() {
        p_data[row + source_row * m] = T::one();
    }
    let parity = if swap_count % 2 == 0 {
        T::one()
    } else {
        T::negative_one()
    };

    let mut l_data = vec![T::default(); m * k];
    for col in 0..k {
        for row in col..m {
            l_data[row + col * m] = lu[row + col * m];
        }
        l_data[col + col * m] = T::one();
    }
    let u_data = leading_upper_triangle_from_lapack(&lu, m, k, n)?;

    Ok(vec![
        tensor_from_vec_with_template(vec![m, m], p_data, input)?,
        tensor_from_vec_with_template(vec![m, k], l_data, input)?,
        tensor_from_vec_with_template(vec![k, n], u_data, input)?,
        tensor_from_vec_with_template(vec![], vec![parity], input)?,
    ])
}

fn lu_factor_2d<T: LapackLu>(
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    let (m, n) = matrix_dims(input, "lu_factor")?;
    let k = m.min(n);
    let m_i32 = dim_i32(m, "lu_factor")?;
    let n_i32 = dim_i32(n, "lu_factor")?;
    let mut lu = input.host_data()?.to_vec();
    let mut ipiv = vec![0_i32; k];
    let mut info = 0;
    T::getrf(m_i32, n_i32, &mut lu, m_i32, &mut ipiv, &mut info);
    check_lapack_info("lu_factor", "getrf", info.min(0))?;

    let swap_count = ipiv
        .iter()
        .enumerate()
        .filter(|(idx, pivot_one_based)| **pivot_one_based != (*idx as i32 + 1))
        .count();
    let parity = if swap_count % 2 == 0 {
        T::one()
    } else {
        T::negative_one()
    };

    Ok((
        tensor_from_vec_with_template(vec![m, n], lu, input)?,
        tensor_from_vec_with_template(vec![k], ipiv, input)?,
        tensor_from_vec_with_template(vec![], vec![parity], input)?,
    ))
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
    let matrix_len = checked_product("lu_factor", "matrix shape", &[m, n])?;
    let k = m.min(n);
    let batch_total = batch_element_count("lu_factor", batch_shape)?;
    let mut lu_data = Vec::with_capacity(checked_product(
        "lu_factor",
        "packed LU output",
        &[matrix_len, batch_total],
    )?);
    let mut pivot_data = Vec::with_capacity(checked_product(
        "lu_factor",
        "pivot output",
        &[k, batch_total],
    )?);
    let mut parity_data = Vec::with_capacity(batch_total);

    let first_range = checked_slice_range("lu_factor", 0, matrix_len)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        vec![m, n],
        &input.host_data()?[first_range],
        input,
    )?;

    for batch in 0..batch_total {
        if batch > 0 {
            let range = checked_slice_range("lu_factor", batch, matrix_len)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let (packed, pivots, parity) = lu_factor_2d(&batch_input)?;
        lu_data.extend_from_slice(packed.host_data()?);
        pivot_data.extend_from_slice(pivots.host_data()?);
        parity_data.extend_from_slice(parity.host_data()?);
    }

    Ok((
        tensor_from_vec_with_template(input.shape().to_vec(), lu_data, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), pivot_data, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity_data, input)?,
    ))
}
