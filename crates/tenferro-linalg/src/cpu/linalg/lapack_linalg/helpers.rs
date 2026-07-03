use std::ops::Range;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::{Tensor, TypedTensor};

pub(crate) fn matrix_dims<T>(
    input: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<(usize, usize)> {
    if input.shape().len() != 2 {
        return Err(tenferro_tensor::Error::RankMismatch {
            op,
            expected: 2,
            actual: input.shape().len(),
        });
    }
    Ok((input.shape()[0], input.shape()[1]))
}

pub(crate) fn square_matrix_dim<T>(
    input: &TypedTensor<T>,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    let (rows, cols) = matrix_dims(input, op)?;
    if rows != cols {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op,
            lhs: vec![rows],
            rhs: vec![cols],
        });
    }
    Ok(rows)
}

pub(crate) fn tensor_from_vec_with_template<T: Clone, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut tensor = TypedTensor::from_vec_col_major(shape, data)?;
    tensor.set_placement(template.placement().clone());
    Ok(tensor)
}

pub(crate) fn tensor_from_pooled_slice_with_template<T: PoolScalar, U>(
    buffers: &mut BufferPool,
    shape: Vec<usize>,
    data: &[T],
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    let mut owned = buffers.acquire_with_capacity::<T>(data.len());
    owned.extend_from_slice(data);
    tensor_from_vec_with_template(shape, owned, template)
}

pub(crate) fn refill_tensor_from_slice<T: Copy>(
    tensor: &mut TypedTensor<T>,
    data: &[T],
) -> tenferro_tensor::Result<()> {
    tensor.host_data_mut()?.copy_from_slice(data);
    Ok(())
}

pub(crate) fn split_core_and_batch_result<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &'static str,
) -> tenferro_tensor::Result<(&'a [usize], &'a [usize])> {
    if input.shape().len() < core_rank {
        return Err(tenferro_tensor::Error::RankMismatch {
            op,
            expected: core_rank,
            actual: input.shape().len(),
        });
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
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op,
            lhs: vec![rows],
            rhs: vec![cols],
        });
    }
    Ok((rows, batch_shape))
}

pub(crate) fn batch_element_count(
    op: &'static str,
    batch_shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    if batch_shape.is_empty() {
        return Ok(1);
    }
    batch_shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                op,
                message: "batch element count overflow".into(),
            })
    })
}

pub(crate) fn checked_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                op,
                message: format!("{role} element count overflow"),
            })
    })
}

fn checked_repeated_len(
    op: &'static str,
    role: &'static str,
    per_batch: usize,
    batch_count: usize,
) -> tenferro_tensor::Result<usize> {
    per_batch
        .checked_mul(batch_count)
        .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
            op,
            message: format!("{role} repeated batch length overflow"),
        })
}

pub(crate) fn checked_slice_range(
    op: &'static str,
    batch_idx: usize,
    slice_size: usize,
) -> tenferro_tensor::Result<Range<usize>> {
    let start =
        batch_idx
            .checked_mul(slice_size)
            .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                op,
                message: "batch slice start overflow".into(),
            })?;
    let end =
        start
            .checked_add(slice_size)
            .ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
                op,
                message: "batch slice end overflow".into(),
            })?;
    Ok(start..end)
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
    i32::try_from(value).map_err(|_| tenferro_tensor::Error::InvalidConfig {
        op,
        message: format!("dimension {value} exceeds LAPACK i32 range"),
    })
}

pub(crate) fn work_len(
    query: f64,
    op: &'static str,
    routine: &'static str,
) -> tenferro_tensor::Result<i32> {
    if !(query.is_finite() && query >= 1.0) {
        return Err(tenferro_tensor::Error::backend_failure(
            op,
            format!("LAPACK {routine} returned invalid workspace size {query}"),
        ));
    }
    dim_i32(query.ceil() as usize, op)
}

pub(crate) fn check_lapack_info(
    op: &'static str,
    routine: &'static str,
    info: i32,
) -> tenferro_tensor::Result<()> {
    if info < 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op,
            message: format!("LAPACK {routine} argument {} had an illegal value", -info),
        });
    }
    if info > 0 {
        return Err(tenferro_tensor::Error::backend_failure(
            op,
            format!("LAPACK {routine} failed with info {info}"),
        ));
    }
    Ok(())
}

pub(crate) fn lower_triangle_from_lapack<T: Copy + Default>(
    data: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); rows * cols];
    for col in 0..cols {
        for row in col..rows {
            out[row + col * rows] = data[row + col * rows];
        }
    }
    out
}

pub(crate) fn leading_upper_triangle_from_lapack<T: Copy + Default>(
    data: &[T],
    source_rows: usize,
    rows: usize,
    cols: usize,
) -> Vec<T> {
    let mut out = vec![T::default(); rows * cols];
    for col in 0..cols {
        for row in 0..rows.min(col + 1) {
            out[row + col * rows] = data[row + col * source_rows];
        }
    }
    out
}

pub(crate) fn transpose_col_major_data<T: Copy>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut transposed = Vec::with_capacity(data.len());
    for j in 0..rows {
        for i in 0..cols {
            transposed.push(data[j + i * rows]);
        }
    }
    transposed
}

pub(crate) fn batched_single<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    op: F,
) -> tenferro_tensor::Result<TypedTensor<T>>
where
    T: PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> tenferro_tensor::Result<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch_result(input, 2, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = batch_element_count(op_name, batch_shape)?;
    if batch_count == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: op_name,
            message: "zero-sized batch dims must be handled by the caller".into(),
        });
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input,
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_output = op(buffers, &batch_input)?;

        if let Some(expected_shape) = &out_core_shape {
            if batch_output.shape() != expected_shape.as_slice() {
                return Err(tenferro_tensor::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape().to_vec(),
                    rhs: expected_shape.clone(),
                });
            }
        } else {
            out_data = Some(Vec::with_capacity(checked_repeated_len(
                op_name,
                "output",
                batch_output.n_elements(),
                batch_count,
            )?));
            out_core_shape = Some(batch_output.shape().to_vec());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()?),
            None => {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: "missing output buffer after first batch".into(),
                });
            }
        }
    }

    let mut out_shape = out_core_shape.ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
        op: op_name,
        message: "missing output shape".into(),
    })?;
    out_shape.extend_from_slice(batch_shape);
    let out_data = out_data.ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
        op: op_name,
        message: "missing output data".into(),
    })?;
    tensor_from_vec_with_template(out_shape, out_data, input)
}

pub(crate) fn batched_multi<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    op: F,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>
where
    T: PoolScalar,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch_result(input, 2, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = batch_element_count(op_name, batch_shape)?;
    if batch_count == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: op_name,
            message: "zero-sized batch dims must be handled by the caller".into(),
        });
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<T>> = Vec::new();

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input,
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            if batch_outputs.is_empty() {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: "missing outputs for first batch".into(),
                });
            }
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape().to_vec())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| {
                    checked_repeated_len(op_name, "output", tensor.n_elements(), batch_count)
                        .map(Vec::with_capacity)
                })
                .collect::<tenferro_tensor::Result<_>>()?;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: format!(
                        "output count mismatch across batches: got {}, expected {}",
                        batch_outputs.len(),
                        out_shapes.len()
                    ),
                });
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape() != out_shapes[idx].as_slice() {
                return Err(tenferro_tensor::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape().to_vec(),
                    rhs: out_shapes[idx].clone(),
                });
            }
            out_data[idx].extend_from_slice(batch_output.host_data()?);
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect()
}

pub(crate) fn batched_multi_convert<InT: PoolScalar, OutT: Clone, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<InT>,
    op: F,
) -> tenferro_tensor::Result<Vec<TypedTensor<OutT>>>
where
    F: Fn(&mut BufferPool, &TypedTensor<InT>) -> tenferro_tensor::Result<Vec<TypedTensor<OutT>>>,
{
    let (core_shape, batch_shape) = split_core_and_batch_result(input, 2, op_name)?;
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size = checked_product(op_name, "core shape", core_shape)?;
    let batch_count = batch_element_count(op_name, batch_shape)?;
    if batch_count == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: op_name,
            message: "zero-sized batch dims must be handled by the caller".into(),
        });
    }

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<OutT>> = Vec::new();

    let first_range = checked_slice_range(op_name, 0, slice_size)?;
    let mut batch_input = tensor_from_pooled_slice_with_template(
        buffers,
        core_shape.to_vec(),
        &input.host_data()?[first_range],
        input,
    )?;

    for batch_idx in 0..batch_count {
        if batch_idx > 0 {
            let range = checked_slice_range(op_name, batch_idx, slice_size)?;
            refill_tensor_from_slice(&mut batch_input, &input.host_data()?[range])?;
        }
        let batch_outputs = op(buffers, &batch_input)?;

        if out_shapes.is_empty() {
            if batch_outputs.is_empty() {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: "missing outputs for first batch".into(),
                });
            }
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape().to_vec())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| {
                    checked_repeated_len(op_name, "output", tensor.n_elements(), batch_count)
                        .map(Vec::with_capacity)
                })
                .collect::<tenferro_tensor::Result<_>>()?;
        } else {
            if batch_outputs.len() != out_shapes.len() {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: format!(
                        "output count mismatch across batches: got {}, expected {}",
                        batch_outputs.len(),
                        out_shapes.len()
                    ),
                });
            }
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            if batch_output.shape() != out_shapes[idx].as_slice() {
                return Err(tenferro_tensor::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape().to_vec(),
                    rhs: out_shapes[idx].clone(),
                });
            }
            out_data[idx].extend_from_slice(batch_output.host_data()?);
        }
    }

    out_shapes
        .into_iter()
        .zip(out_data)
        .map(|(mut out_shape, out_data)| {
            out_shape.extend_from_slice(batch_shape);
            tensor_from_vec_with_template(out_shape, out_data, input)
        })
        .collect()
}

pub(crate) fn batched_binary_result<T, F>(
    op_name: &'static str,
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    op: F,
) -> tenferro_tensor::Result<TypedTensor<T>>
where
    T: PoolScalar,
    F: Fn(
        &mut BufferPool,
        &TypedTensor<T>,
        &TypedTensor<T>,
    ) -> tenferro_tensor::Result<TypedTensor<T>>,
{
    let (a_core_shape, a_batch_shape) = split_core_and_batch_result(a, 2, op_name)?;
    let (b_core_shape, b_batch_shape) = split_core_and_batch_result(b, 2, op_name)?;
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: op_name,
            lhs: a_batch_shape.to_vec(),
            rhs: b_batch_shape.to_vec(),
        });
    }

    if a_batch_shape.is_empty() {
        return op(buffers, a, b);
    }

    let a_slice_size = checked_product(op_name, "lhs core shape", a_core_shape)?;
    let b_slice_size = checked_product(op_name, "rhs core shape", b_core_shape)?;
    let batch_count = batch_element_count(op_name, a_batch_shape)?;
    if batch_count == 0 {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: op_name,
            message: "zero-sized batch dims must be handled by the caller".into(),
        });
    }

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    let a_first_range = checked_slice_range(op_name, 0, a_slice_size)?;
    let b_first_range = checked_slice_range(op_name, 0, b_slice_size)?;
    let mut batch_a = tensor_from_pooled_slice_with_template(
        buffers,
        a_core_shape.to_vec(),
        &a.host_data()?[a_first_range],
        a,
    )?;
    let mut batch_b = tensor_from_pooled_slice_with_template(
        buffers,
        b_core_shape.to_vec(),
        &b.host_data()?[b_first_range],
        b,
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
                return Err(tenferro_tensor::Error::ShapeMismatch {
                    op: op_name,
                    lhs: batch_output.shape().to_vec(),
                    rhs: expected_shape.clone(),
                });
            }
        } else {
            out_data = Some(Vec::with_capacity(checked_repeated_len(
                op_name,
                "output",
                batch_output.n_elements(),
                batch_count,
            )?));
            out_core_shape = Some(batch_output.shape().to_vec());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()?),
            None => {
                return Err(tenferro_tensor::Error::InvalidConfig {
                    op: op_name,
                    message: "missing output buffer after first batch".into(),
                });
            }
        }
    }

    let mut out_shape = out_core_shape.ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
        op: op_name,
        message: "missing output shape".into(),
    })?;
    out_shape.extend_from_slice(a_batch_shape);
    let out_data = out_data.ok_or_else(|| tenferro_tensor::Error::InvalidConfig {
        op: op_name,
        message: "missing output data".into(),
    })?;
    tensor_from_vec_with_template(out_shape, out_data, b)
}

pub(crate) fn zero_dim_eig_outputs(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
    let shape = input.shape();
    if shape.len() < 2 {
        return Err(tenferro_tensor::Error::RankMismatch {
            op: "eig",
            expected: 2,
            actual: shape.len(),
        });
    }
    let n = shape[0];
    if shape[1] != n {
        return Err(tenferro_tensor::Error::ShapeMismatch {
            op: "eig",
            lhs: vec![n],
            rhs: vec![shape[1]],
        });
    }
    let batch_shape = &shape[2..];
    let value_shape = vector_with_batch_shape(n, batch_shape);
    let vector_shape = matrix_with_batch_shape(n, n, batch_shape);
    match input {
        Tensor::F32(_) | Tensor::C32(_) => Ok(vec![
            Tensor::C32(TypedTensor::from_vec_col_major(value_shape, Vec::new())?),
            Tensor::C32(TypedTensor::from_vec_col_major(vector_shape, Vec::new())?),
        ]),
        Tensor::F64(_) | Tensor::C64(_) => Ok(vec![
            Tensor::C64(TypedTensor::from_vec_col_major(value_shape, Vec::new())?),
            Tensor::C64(TypedTensor::from_vec_col_major(vector_shape, Vec::new())?),
        ]),
        _ => Err(tenferro_tensor::Error::backend_failure(
            "eig",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
    }
}
