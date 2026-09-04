use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::{TensorScalar, TypedTensor};

use crate::{RankRevealingQrOptions, RankRevealingQrResult};

pub(crate) type TypedRrqr<T> = RankRevealingQrResult<TypedTensor<T>, TypedTensor<i64>>;

pub(crate) fn prefix_rank(
    diagonal_magnitudes: impl IntoIterator<Item = f64>,
    options: RankRevealingQrOptions,
) -> tenferro_tensor::Result<i64> {
    let diagonal = diagonal_magnitudes.into_iter().collect::<Vec<_>>();
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(crate::error::into_tensor_error(
            "rank_revealing_qr",
            crate::Error::NonFinite {
                op: "rank_revealing_qr",
                role: "R diagonal",
            },
        ));
    }
    let Some(&leading) = diagonal.first() else {
        return Ok(0);
    };
    // A finite rtol times a finite leading magnitude can overflow to infinity;
    // that means no diagonal clears the requested threshold, hence rank zero.
    let threshold = options.atol.max(options.rtol * leading);
    let rank = diagonal
        .iter()
        .take_while(|&&value| value > threshold)
        .count();
    i64::try_from(rank).map_err(|_| {
        tenferro_tensor::Error::invalid_argument(
            "rank_revealing_qr",
            "rank",
            "rank exceeds i64 range",
        )
    })
}

pub(crate) fn identity_permutation(n: usize) -> tenferro_tensor::Result<Vec<i64>> {
    (0..n)
        .map(|column| {
            i64::try_from(column).map_err(|_| {
                tenferro_tensor::Error::invalid_argument(
                    "rank_revealing_qr",
                    "column_permutation",
                    "column index exceeds i64 range",
                )
            })
        })
        .collect()
}

pub(crate) fn zero_matrix_result<T>(
    input: &TypedTensor<T>,
    one: T,
) -> tenferro_tensor::Result<TypedRrqr<T>>
where
    T: Copy + Default + TensorScalar,
{
    let [m, n] = input.shape() else {
        return Err(tenferro_tensor::Error::rank_mismatch(
            "rank_revealing_qr",
            2,
            input.shape().len(),
        ));
    };
    let k = (*m).min(*n);
    let q_len = checked_product(&[*m, k], "Q")?;
    let r_len = checked_product(&[k, *n], "R")?;
    let mut q = vec![T::default(); q_len];
    for diagonal in 0..k {
        q[diagonal + diagonal * *m] = one;
    }
    Ok(RankRevealingQrResult {
        q: tensor_with_template(vec![*m, k], q, input)?,
        r: tensor_with_template(vec![k, *n], vec![T::default(); r_len], input)?,
        column_permutation: tensor_with_template(vec![*n], identity_permutation(*n)?, input)?,
        rank: tensor_with_template(vec![], vec![0_i64], input)?,
    })
}

pub(crate) fn batched<T, F>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    factor: F,
) -> tenferro_tensor::Result<TypedRrqr<T>>
where
    T: Copy + Default + PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> tenferro_tensor::Result<TypedRrqr<T>>,
{
    if input.shape().len() < 2 {
        return Err(tenferro_tensor::Error::rank_mismatch(
            "rank_revealing_qr",
            2,
            input.shape().len(),
        ));
    }
    let m = input.shape()[0];
    let n = input.shape()[1];
    let k = m.min(n);
    let batch_shape = &input.shape()[2..];
    let batch_count = checked_product(batch_shape, "batch")?;
    let matrix_len = checked_product(&[m, n], "input matrix")?;

    if batch_shape.is_empty() {
        return factor(buffers, input);
    }
    if batch_count == 0 {
        return empty_batched_result(input, m, n, k, batch_shape);
    }

    let q_batch_len = checked_product(&[m, k], "Q batch")?;
    let r_batch_len = checked_product(&[k, n], "R batch")?;
    let mut q_data = Vec::with_capacity(checked_repeated(q_batch_len, batch_count, "Q")?);
    let mut r_data = Vec::with_capacity(checked_repeated(r_batch_len, batch_count, "R")?);
    let mut permutation_data = Vec::with_capacity(checked_repeated(n, batch_count, "permutation")?);
    let mut rank_data = Vec::with_capacity(batch_count);

    let input_data = input.host_data()?;
    let mut batch_input =
        tensor_with_template(vec![m, n], input_data[..matrix_len].to_vec(), input)?;
    for batch_index in 0..batch_count {
        if batch_index != 0 {
            let start = batch_index.checked_mul(matrix_len).ok_or_else(overflow)?;
            let end = start.checked_add(matrix_len).ok_or_else(overflow)?;
            batch_input
                .host_data_mut()?
                .copy_from_slice(&input_data[start..end]);
        }
        let result = factor(buffers, &batch_input)?;
        validate_core_shapes(&result, m, n, k)?;
        q_data.extend_from_slice(result.q.host_data()?);
        r_data.extend_from_slice(result.r.host_data()?);
        permutation_data.extend_from_slice(result.column_permutation.host_data()?);
        rank_data.extend_from_slice(result.rank.host_data()?);
    }

    Ok(RankRevealingQrResult {
        q: tensor_with_template(matrix_shape(m, k, batch_shape), q_data, input)?,
        r: tensor_with_template(matrix_shape(k, n, batch_shape), r_data, input)?,
        column_permutation: tensor_with_template(
            vector_shape(n, batch_shape),
            permutation_data,
            input,
        )?,
        rank: tensor_with_template(batch_shape.to_vec(), rank_data, input)?,
    })
}

fn empty_batched_result<T: Copy + Default + TensorScalar>(
    input: &TypedTensor<T>,
    m: usize,
    n: usize,
    k: usize,
    batch_shape: &[usize],
) -> tenferro_tensor::Result<TypedRrqr<T>> {
    Ok(RankRevealingQrResult {
        q: tensor_with_template(matrix_shape(m, k, batch_shape), Vec::new(), input)?,
        r: tensor_with_template(matrix_shape(k, n, batch_shape), Vec::new(), input)?,
        column_permutation: tensor_with_template(vector_shape(n, batch_shape), Vec::new(), input)?,
        rank: tensor_with_template(batch_shape.to_vec(), Vec::new(), input)?,
    })
}

fn validate_core_shapes<T>(
    result: &TypedRrqr<T>,
    m: usize,
    n: usize,
    k: usize,
) -> tenferro_tensor::Result<()> {
    for (role, actual, expected) in [
        ("Q", result.q.shape(), &[m, k][..]),
        ("R", result.r.shape(), &[k, n][..]),
        (
            "column_permutation",
            result.column_permutation.shape(),
            &[n][..],
        ),
        ("rank", result.rank.shape(), &[][..]),
    ] {
        if actual != expected {
            return Err(tenferro_tensor::Error::invalid_argument(
                "rank_revealing_qr",
                "provider output",
                format!("{role} shape {actual:?} does not match {expected:?}"),
            ));
        }
    }
    Ok(())
}

fn tensor_with_template<T: Clone + TensorScalar, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(template.placement().clone());
    Ok(tensor)
}

fn matrix_shape(rows: usize, cols: usize, batch: &[usize]) -> Vec<usize> {
    let mut shape = vec![rows, cols];
    shape.extend_from_slice(batch);
    shape
}

fn vector_shape(len: usize, batch: &[usize]) -> Vec<usize> {
    let mut shape = vec![len];
    shape.extend_from_slice(batch);
    shape
}

fn checked_product(shape: &[usize], role: &'static str) -> tenferro_tensor::Result<usize> {
    tenferro_tensor::validate::checked_shape_product("rank_revealing_qr", role, shape)
}

fn checked_repeated(
    per_batch: usize,
    batch_count: usize,
    role: &'static str,
) -> tenferro_tensor::Result<usize> {
    per_batch.checked_mul(batch_count).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "rank_revealing_qr",
            role,
            "batched output length overflows usize",
        )
    })
}

fn overflow() -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(
        "rank_revealing_qr",
        "batch offset",
        "batch offset overflows usize",
    )
}
