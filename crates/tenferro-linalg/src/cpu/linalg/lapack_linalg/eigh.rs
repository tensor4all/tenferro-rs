use num_complex::{Complex32, Complex64};

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, check_lapack_info, checked_product, dim_i32, has_zero_dim,
    matrix_with_batch_shape, pooled_copy, pooled_zeroed, release_scratch,
    square_core_and_batch_result, tensor_from_vec_with_template, vector_with_batch_shape, work_len,
};

pub(crate) trait LapackEigh:
    Clone + Copy + Default + PoolScalar + tenferro_tensor::TensorScalar
{
    type Real: Clone + Copy + Default + PoolScalar + tenferro_tensor::TensorScalar;

    /// Diagonalize every compact `n x n` Hermitian matrix of `matrices` in
    /// place, writing `n` ascending eigenvalues per matrix into `values`.
    ///
    /// With `vectors`, each matrix is overwritten by its eigenvectors;
    /// otherwise its contents are destroyed. The workspace is queried once
    /// and reused across the batch.
    ///
    /// # Errors
    ///
    /// Returns `Error::Internal` for inconsistent buffer lengths, an invalid
    /// workspace error for a bad LAPACK query, and `NonConvergence` or an
    /// illegal-argument error from the LAPACK driver.
    fn eigh_batched(
        buffers: &mut BufferPool,
        op: &'static str,
        n: usize,
        vectors: bool,
        matrices: &mut [Self],
        values: &mut [<Self as LapackEigh>::Real],
    ) -> tenferro_tensor::Result<()>;

    /// Real eigenvalues in this scalar type, reusing the buffer when the
    /// scalar is real.
    fn values_as_scalar(
        buffers: &mut BufferPool,
        values: Vec<<Self as LapackEigh>::Real>,
    ) -> Vec<Self>;
}

/// Check that `matrices` and `values` hold the same number of `n x n`
/// matrices and length-`n` value vectors, returning the matrix length.
fn check_eigh_batches(
    op: &'static str,
    n: usize,
    matrices: usize,
    values: usize,
) -> tenferro_tensor::Result<usize> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let consistent = if matrix_len == 0 {
        values == 0
    } else {
        matrices.is_multiple_of(matrix_len)
            && values == checked_product(op, "values", &[n, matrices / matrix_len])?
    };
    if !consistent {
        return Err(tenferro_tensor::Error::Internal(format!(
            "{op}: matrix and eigenvalue buffers describe different batches"
        )));
    }
    Ok(matrix_len)
}

fn iwork_len(query: i32, op: &'static str, routine: &'static str) -> tenferro_tensor::Result<i32> {
    if query < 1 {
        return Err(crate::error::invalid_workspace(
            op,
            "LAPACK",
            routine,
            format!("invalid integer workspace size {query}"),
        ));
    }
    Ok(query)
}

fn queried_iwork_len(
    query: &[i32],
    op: &'static str,
    routine: &'static str,
) -> tenferro_tensor::Result<i32> {
    let query = query.first().copied().ok_or_else(|| {
        crate::error::invalid_workspace(
            op,
            "LAPACK",
            routine,
            "did not return an integer workspace size",
        )
    })?;
    iwork_len(query, op, routine)
}

fn iwork_capacity(
    len: i32,
    op: &'static str,
    routine: &'static str,
) -> tenferro_tensor::Result<usize> {
    usize::try_from(len).map_err(|_| {
        crate::error::invalid_workspace(
            op,
            "LAPACK",
            routine,
            format!("integer workspace size {len} does not fit usize"),
        )
    })
}

macro_rules! impl_real_eigh {
    ($scalar:ty, $syevd:path, $routine:literal) => {
        impl LapackEigh for $scalar {
            type Real = $scalar;

            fn eigh_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                n: usize,
                vectors: bool,
                matrices: &mut [Self],
                values: &mut [Self],
            ) -> tenferro_tensor::Result<()> {
                let matrix_len = check_eigh_batches(op, n, matrices.len(), values.len())?;
                if matrix_len == 0 || matrices.is_empty() {
                    return Ok(());
                }
                let jobz = if vectors { b'V' } else { b'N' };
                let n_i32 = dim_i32(n, op)?;
                let mut query = [<$scalar>::default(); 1];
                let mut iquery = [0_i32; 1];
                let mut info = 0;
                // SAFETY: the first chunk is a mutable column-major `n x n`
                // buffer and the first `n` values exist (lengths checked
                // above); `lwork = liwork = -1` writes only the query slots.
                unsafe {
                    $syevd(
                        jobz,
                        b'L',
                        n_i32,
                        &mut matrices[..matrix_len],
                        n_i32,
                        &mut values[..n],
                        &mut query,
                        -1,
                        &mut iquery,
                        -1,
                        &mut info,
                    );
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0] as f64, op, $routine)?;
                let liwork = queried_iwork_len(&iquery, op, $routine)?;
                let liwork_capacity = iwork_capacity(liwork, op, $routine)?;
                let mut work = pooled_zeroed::<$scalar>(buffers, lwork as usize);
                let mut iwork = pooled_zeroed::<i32>(buffers, liwork_capacity);
                // INVARIANT: the lengths were checked above, so the chunk
                // iterators yield the same number of `n x n` matrices and
                // length-`n` value blocks; the workspace depends only on
                // `(jobz, n)` and is reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for (matrix, vals) in matrices
                    .chunks_exact_mut(matrix_len)
                    .zip(values.chunks_exact_mut(n))
                {
                    // SAFETY: dimensions and workspace lengths come from the
                    // validated shape plus the LAPACK query; all buffers are live.
                    unsafe {
                        $syevd(
                            jobz, b'L', n_i32, matrix, n_i32, vals, &mut work, lwork, &mut iwork,
                            liwork, &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, work);
                release_scratch(buffers, iwork);
                Ok(())
            }

            fn values_as_scalar(_buffers: &mut BufferPool, values: Vec<Self>) -> Vec<Self> {
                values
            }
        }
    };
}

macro_rules! impl_complex_eigh {
    ($complex:ty, $real:ty, $heev:path, $routine:literal) => {
        impl LapackEigh for $complex {
            type Real = $real;

            fn eigh_batched(
                buffers: &mut BufferPool,
                op: &'static str,
                n: usize,
                vectors: bool,
                matrices: &mut [Self],
                values: &mut [$real],
            ) -> tenferro_tensor::Result<()> {
                let matrix_len = check_eigh_batches(op, n, matrices.len(), values.len())?;
                if matrix_len == 0 || matrices.is_empty() {
                    return Ok(());
                }
                let jobz = if vectors { b'V' } else { b'N' };
                let n_i32 = dim_i32(n, op)?;
                let rwork_len = checked_product(op, "real workspace", &[3, n])?
                    .checked_sub(2)
                    .unwrap_or(1)
                    .max(1);
                let mut rwork = pooled_zeroed::<$real>(buffers, rwork_len);
                let mut query = [<$complex>::default(); 1];
                let mut info = 0;
                // SAFETY: the first chunk and value block satisfy LAPACK's
                // Hermitian eigensolver dimensions (lengths checked above);
                // `lwork = -1` writes only `query`.
                unsafe {
                    $heev(
                        jobz,
                        b'L',
                        n_i32,
                        &mut matrices[..matrix_len],
                        n_i32,
                        &mut values[..n],
                        &mut query,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }
                check_lapack_info(op, concat!($routine, "(work query)"), info)?;
                let lwork = work_len(query[0].re as f64, op, $routine)?;
                let mut work = pooled_zeroed::<$complex>(buffers, lwork as usize);
                // INVARIANT: the lengths were checked above, so the chunk
                // iterators yield the same number of `n x n` matrices and
                // length-`n` value blocks; `work` and `rwork` depend only on
                // `(jobz, n)` and are reused across the batch. The serial
                // loop is intentional: the LAPACK provider owns threading.
                for (matrix, vals) in matrices
                    .chunks_exact_mut(matrix_len)
                    .zip(values.chunks_exact_mut(n))
                {
                    // SAFETY: every buffer matches the validated `n x n`
                    // problem and the queried workspace length.
                    unsafe {
                        $heev(
                            jobz, b'L', n_i32, matrix, n_i32, vals, &mut work, lwork, &mut rwork,
                            &mut info,
                        );
                    }
                    check_lapack_info(op, $routine, info)?;
                }
                release_scratch(buffers, rwork);
                release_scratch(buffers, work);
                Ok(())
            }

            fn values_as_scalar(buffers: &mut BufferPool, values: Vec<$real>) -> Vec<Self> {
                let mut converted = buffers.acquire_with_capacity::<$complex>(values.len());
                converted.extend(values.iter().map(|&value| <$complex>::new(value, 0.0)));
                release_scratch(buffers, values);
                converted
            }
        }
    };
}

impl_real_eigh!(f32, lapack::ssyevd, "ssyevd");
impl_real_eigh!(f64, lapack::dsyevd, "dsyevd");
impl_complex_eigh!(Complex32, f32, lapack::cheev, "cheev");
impl_complex_eigh!(Complex64, f64, lapack::zheev, "zheev");

pub(crate) fn eigh<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    let (n, batch_shape) = square_core_and_batch_result(input, "eigh")?;
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(
                vector_with_batch_shape(n, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(n, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ]);
    }
    let batch = batch_element_count("eigh", batch_shape)?;
    let values_len = checked_product("eigh", "values", &[n, batch])?;
    let mut vectors = pooled_copy(buffers, input.host_data()?);
    let mut values = pooled_zeroed::<<T as LapackEigh>::Real>(buffers, values_len);
    T::eigh_batched(buffers, "eigh", n, true, &mut vectors, &mut values)?;
    let values = T::values_as_scalar(buffers, values);
    Ok(vec![
        tensor_from_vec_with_template(vector_with_batch_shape(n, batch_shape), values, input)?,
        tensor_from_vec_with_template(input.shape().to_vec(), vectors, input)?,
    ])
}

pub(crate) fn eigh_values<T: LapackEigh>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<<T as LapackEigh>::Real>> {
    let (n, batch_shape) = square_core_and_batch_result(input, "eigh_values")?;
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(
            vector_with_batch_shape(n, batch_shape),
            Vec::new(),
            input,
        );
    }
    let batch = batch_element_count("eigh_values", batch_shape)?;
    let values_len = checked_product("eigh_values", "values", &[n, batch])?;
    let mut work_matrices = pooled_copy(buffers, input.host_data()?);
    let mut values = pooled_zeroed::<<T as LapackEigh>::Real>(buffers, values_len);
    T::eigh_batched(
        buffers,
        "eigh_values",
        n,
        false,
        &mut work_matrices,
        &mut values,
    )?;
    release_scratch(buffers, work_matrices);
    tensor_from_vec_with_template(vector_with_batch_shape(n, batch_shape), values, input)
}
