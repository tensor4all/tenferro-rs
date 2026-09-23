use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, check_lapack_info, checked_product, dim_i32, has_zero_dim,
    leading_upper_triangle_from_lapack, lower_triangle_from_lapack, matrix_core_and_batch_result,
    matrix_with_batch_shape, pooled_copy, pooled_zeroed, release_scratch,
    square_core_and_batch_result, square_matrix_dim, tensor_from_vec_with_template,
};

// SAFETY: declarations retain the provider's LAPACK LP64 ABI; callers validate buffers.
unsafe extern "C" {
    #[link_name = "sgetc2_"]
    fn sgetc2_ffi(
        n: *const i32,
        a: *mut f32,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "sgesc2_"]
    fn sgesc2_ffi(
        n: *const i32,
        a: *const f32,
        lda: *const i32,
        rhs: *mut f32,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f32,
    );

    #[link_name = "dgetc2_"]
    fn dgetc2_ffi(
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "dgesc2_"]
    fn dgesc2_ffi(
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        rhs: *mut f64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );

    #[link_name = "cgetc2_"]
    fn cgetc2_ffi(
        n: *const i32,
        a: *mut Complex32,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "cgesc2_"]
    fn cgesc2_ffi(
        n: *const i32,
        a: *const Complex32,
        lda: *const i32,
        rhs: *mut Complex32,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f32,
    );

    #[link_name = "zgetc2_"]
    fn zgetc2_ffi(
        n: *const i32,
        a: *mut Complex64,
        lda: *const i32,
        ipiv: *mut i32,
        jpiv: *mut i32,
        info: *mut i32,
    );

    #[link_name = "zgesc2_"]
    fn zgesc2_ffi(
        n: *const i32,
        a: *const Complex64,
        lda: *const i32,
        rhs: *mut Complex64,
        ipiv: *const i32,
        jpiv: *const i32,
        scale: *mut f64,
    );
}

pub(crate) trait LapackFullPivLu: Clone + Copy + Default + PoolScalar {
    type Scale: Copy;

    fn one() -> Self;
    fn negative_one() -> Self;
    fn scale_one() -> Self::Scale;
    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    );
    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Scale,
    );
    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Scale);
}

impl LapackFullPivLu for f32 {
    type Scale = f32;

    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn scale_one() -> Self::Scale {
        1.0
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            sgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Scale,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            sgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Scale) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

impl LapackFullPivLu for f64 {
    type Scale = f64;

    fn one() -> Self {
        1.0
    }

    fn negative_one() -> Self {
        -1.0
    }

    fn scale_one() -> Self::Scale {
        1.0
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            dgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Scale,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            dgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Scale) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

impl LapackFullPivLu for Complex32 {
    type Scale = f32;

    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex32::new(-1.0, 0.0)
    }

    fn scale_one() -> Self::Scale {
        1.0
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            cgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Scale,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            cgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Scale) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

impl LapackFullPivLu for Complex64 {
    type Scale = f64;

    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn negative_one() -> Self {
        Complex64::new(-1.0, 0.0)
    }

    fn scale_one() -> Self::Scale {
        1.0
    }

    fn getc2(
        n: i32,
        data: &mut [Self],
        lda: i32,
        ipiv: &mut [i32],
        jpiv: &mut [i32],
        info: &mut i32,
    ) {
        // SAFETY: `data` stores an `lda x n` LAPACK column-major matrix,
        // pivot arrays have length at least `n`, and all pointers are valid
        // for the duration of the FFI call.
        unsafe {
            zgetc2_ffi(
                &n,
                data.as_mut_ptr(),
                &lda,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
                info,
            );
        }
    }

    fn gesc2(
        n: i32,
        data: &[Self],
        lda: i32,
        rhs: &mut [Self],
        ipiv: &[i32],
        jpiv: &[i32],
        scale: &mut Self::Scale,
    ) {
        // SAFETY: `data` stores the factorized `lda x n` matrix, `rhs` has
        // length at least `n`, pivot arrays have length at least `n`, and
        // LAPACK only writes through `rhs` and `scale`.
        unsafe {
            zgesc2_ffi(
                &n,
                data.as_ptr(),
                &lda,
                rhs.as_mut_ptr(),
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                scale,
            );
        }
    }

    fn apply_inverse_scale(rhs: &mut [Self], scale: Self::Scale) {
        if scale != 1.0 {
            for value in rhs {
                *value /= scale;
            }
        }
    }
}

fn permutation_from_lapack_pivots(
    pivots: &[i32],
    op: &'static str,
) -> tenferro_tensor::Result<Vec<usize>> {
    let mut permutation: Vec<usize> = (0..pivots.len()).collect();
    for (idx, &pivot_one_based) in pivots.iter().enumerate() {
        let pivot = match usize::try_from(pivot_one_based - 1) {
            Ok(pivot) if pivot < pivots.len() => pivot,
            _ => {
                return Err(tenferro_tensor::Error::Internal(format!(
                    "{op}: LAPACK getc2 returned an invalid pivot index"
                )));
            }
        };
        if pivot != idx {
            permutation.swap(idx, pivot);
        }
    }
    Ok(permutation)
}

fn permutation_matrix<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    permutation: &[usize],
) -> tenferro_tensor::Result<Vec<T>> {
    let n = permutation.len();
    let len = checked_product("full_piv_lu", "permutation matrix", &[n, n])?;
    let mut data = pooled_zeroed::<T>(buffers, len);
    for (row, &source) in permutation.iter().enumerate() {
        data[row + source * n] = T::one();
    }
    Ok(data)
}

/// Run `?getc2` on one compact `n x n` matrix in place.
///
/// Reference LAPACK sets `IPIV(N) = JPIV(N) = N` on return, but some
/// providers (Apple Accelerate) leave that last entry untouched. The last
/// step of a complete-pivot elimination never swaps, and `?gesc2` only
/// applies the first `N - 1` interchanges, so writing `N` here is exact.
fn getc2_in_place<T: LapackFullPivLu>(
    op: &'static str,
    data: &mut [T],
    n_i32: i32,
    ipiv: &mut [i32],
    jpiv: &mut [i32],
) -> tenferro_tensor::Result<()> {
    let mut info = 0;
    T::getc2(n_i32, data, n_i32, ipiv, jpiv, &mut info);
    check_lapack_info(op, "getc2", info.min(0))?;
    if info > 0 {
        return Err(crate::error::into_tensor_error(
            op,
            crate::Error::Singular { op },
        ));
    }
    if let (Some(last_row), Some(last_col)) = (ipiv.last_mut(), jpiv.last_mut()) {
        *last_row = n_i32;
        *last_col = n_i32;
    }
    Ok(())
}

fn factor_getc2<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    op: &'static str,
    data: &mut [T],
    n: usize,
) -> tenferro_tensor::Result<(Vec<i32>, Vec<i32>)> {
    let n_i32 = dim_i32(n, op)?;
    let mut ipiv = pooled_zeroed::<i32>(buffers, n);
    let mut jpiv = pooled_zeroed::<i32>(buffers, n);
    getc2_in_place(op, data, n_i32, &mut ipiv, &mut jpiv)?;
    Ok((ipiv, jpiv))
}

fn full_piv_lu_2d<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let n = square_matrix_dim(input, "full_piv_lu")?;
    let mut lu = pooled_copy(buffers, input.host_data()?);
    let (ipiv, jpiv) = factor_getc2(buffers, "full_piv_lu", &mut lu, n)?;

    let row_perm = permutation_from_lapack_pivots(&ipiv, "full_piv_lu")?;
    let col_perm = permutation_from_lapack_pivots(&jpiv, "full_piv_lu")?;
    let p_data = permutation_matrix::<T>(buffers, &row_perm)?;
    let q_data = permutation_matrix::<T>(buffers, &col_perm)?;
    let mut l_data = lower_triangle_from_lapack(&lu, n, n)?;
    for index in 0..n {
        l_data[index + index * n] = T::one();
    }
    let u_data = leading_upper_triangle_from_lapack(&lu, n, n, n)?;
    let row_swap_count = ipiv
        .iter()
        .enumerate()
        .filter(|(idx, row)| **row != (*idx as i32 + 1))
        .count();
    let col_swap_count = jpiv
        .iter()
        .enumerate()
        .filter(|(idx, col)| **col != (*idx as i32 + 1))
        .count();
    let parity = if (row_swap_count + col_swap_count) % 2 == 0 {
        T::one()
    } else {
        T::negative_one()
    };

    Ok(vec![
        tensor_from_vec_with_template(vec![n, n], p_data, input)?,
        tensor_from_vec_with_template(vec![n, n], l_data, input)?,
        tensor_from_vec_with_template(vec![n, n], u_data, input)?,
        tensor_from_vec_with_template(vec![n, n], q_data, input)?,
        tensor_from_vec_with_template(vec![], vec![parity], input)?,
    ])
}

pub(crate) fn full_piv_lu<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    if has_zero_dim(input.shape()) {
        let (n, batch_shape) = square_core_and_batch_result(input, "full_piv_lu")?;
        let parity_elements = batch_element_count("full_piv_lu", batch_shape)?;
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
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
    super::helpers::batched_multi("full_piv_lu", buffers, input, full_piv_lu_2d)
}

pub(crate) fn full_piv_lu_solve<T: LapackFullPivLu>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "full_piv_lu_solve";
    let (n, a_batch_shape) = square_core_and_batch_result(a, OP)?;
    let (b_rows, b_cols, b_batch_shape) = matrix_core_and_batch_result(b, OP)?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            vec![n],
            vec![b_rows],
        ));
    }
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }

    let matrix_len = checked_product(OP, "matrix", &[n, n])?;
    let rhs_len = checked_product(OP, "rhs", &[n, b_cols])?;
    let n_i32 = dim_i32(n, OP)?;
    let mut lu = buffers.acquire_with_capacity::<T>(matrix_len);
    let mut ipiv = pooled_zeroed::<i32>(buffers, n);
    let mut jpiv = pooled_zeroed::<i32>(buffers, n);
    let mut output = pooled_copy(buffers, b.host_data()?);
    // INVARIANT: owned tensors are compact column-major, the batch shapes
    // match, and no dimension is zero, so both chunk iterators yield the same
    // number of nonempty matrix and RHS blocks, and every RHS block splits
    // into `b_cols` exact columns of length `n`. The LU and pivot scratch is
    // private and refilled per matrix. The serial loop is intentional: the
    // LAPACK provider owns threading.
    for (matrix, rhs) in a
        .host_data()?
        .chunks_exact(matrix_len)
        .zip(output.chunks_exact_mut(rhs_len))
    {
        lu.clear();
        if transpose_a {
            for col in 0..n {
                lu.extend((0..n).map(|row| matrix[col + row * n]));
            }
        } else {
            lu.extend_from_slice(matrix);
        }
        getc2_in_place(OP, &mut lu, n_i32, &mut ipiv, &mut jpiv)?;
        for column in rhs.chunks_exact_mut(n) {
            let mut scale = T::scale_one();
            T::gesc2(n_i32, &lu, n_i32, column, &ipiv, &jpiv, &mut scale);
            T::apply_inverse_scale(column, scale);
        }
    }
    release_scratch(buffers, lu);
    release_scratch(buffers, ipiv);
    release_scratch(buffers, jpiv);
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}
