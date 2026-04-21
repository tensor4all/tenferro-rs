use crate::buffer_pool::BufferPool;
use crate::{Tensor, TypedTensor};

pub(crate) fn matrix_dims<T>(input: &TypedTensor<T>, op: &str) -> (usize, usize) {
    assert_eq!(input.shape.len(), 2, "{op}: expected a 2D matrix");
    (input.shape[0], input.shape[1])
}

pub(crate) fn square_matrix_dim<T>(input: &TypedTensor<T>, op: &str) -> usize {
    let (rows, cols) = matrix_dims(input, op);
    assert_eq!(rows, cols, "{op}: expected a square matrix");
    rows
}

pub(crate) fn tensor_from_vec_with_template<T: Clone, U>(
    shape: Vec<usize>,
    data: Vec<T>,
    template: &TypedTensor<U>,
) -> TypedTensor<T> {
    TypedTensor {
        buffer: crate::Buffer::Host(data),
        shape,
        placement: template.placement.clone(),
    }
}

pub(crate) fn split_core_and_batch<'a, T>(
    input: &'a TypedTensor<T>,
    core_rank: usize,
    op: &str,
) -> (&'a [usize], &'a [usize]) {
    assert!(
        input.shape.len() >= core_rank,
        "{op}: expected rank >= {core_rank}"
    );
    input.shape.split_at(core_rank)
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

pub(crate) fn dim_i32(value: usize, op: &str) -> i32 {
    match i32::try_from(value) {
        Ok(value) => value,
        Err(_) => panic!("{op}: dimension exceeds LAPACK i32 range"),
    }
}

pub(crate) fn work_len(query: f64, op: &str, routine: &str) -> i32 {
    assert!(
        query.is_finite() && query >= 1.0,
        "{op}: LAPACK {routine} returned invalid workspace size {query}"
    );
    dim_i32(query.ceil() as usize, op)
}

pub(crate) fn panic_on_lapack_error(op: &str, routine: &str, info: i32) {
    if info < 0 {
        panic!(
            "{op}: LAPACK {routine} argument {} had an illegal value",
            -info
        );
    }
    if info > 0 {
        panic!("{op}: LAPACK {routine} failed with info {info}");
    }
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
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    op: F,
) -> crate::Result<TypedTensor<T>>
where
    T: Clone,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> crate::Result<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, 2, "batched_single");
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_single: zero-sized batch dims are unsupported"
    );

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_output = op(buffers, &batch_input)?;

        if let Some(expected_shape) = &out_core_shape {
            assert_eq!(
                batch_output.shape.as_slice(),
                expected_shape.as_slice(),
                "batched_single: output core shape mismatch across batches"
            );
        } else {
            out_data = Some(Vec::with_capacity(batch_output.n_elements() * batch_count));
            out_core_shape = Some(batch_output.shape.clone());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()),
            None => panic!("batched_single: missing output buffer"),
        }
    }

    let mut out_shape = match out_core_shape {
        Some(shape) => shape,
        None => panic!("batched_single: missing output shape"),
    };
    out_shape.extend_from_slice(batch_shape);
    Ok(tensor_from_vec_with_template(
        out_shape,
        match out_data {
            Some(data) => data,
            None => panic!("batched_single: missing output data"),
        },
        input,
    ))
}

pub(crate) fn batched_multi<T, F>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    op: F,
) -> Vec<TypedTensor<T>>
where
    T: Clone,
    F: Fn(&mut BufferPool, &TypedTensor<T>) -> Vec<TypedTensor<T>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, 2, "batched_multi");
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_multi: zero-sized batch dims are unsupported"
    );

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<T>> = Vec::new();

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_outputs = op(buffers, &batch_input);

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| Vec::with_capacity(tensor.n_elements() * batch_count))
                .collect();
        } else {
            assert_eq!(
                batch_outputs.len(),
                out_shapes.len(),
                "batched_multi: output count mismatch across batches"
            );
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            assert_eq!(
                batch_output.shape.as_slice(),
                out_shapes[idx].as_slice(),
                "batched_multi: output core shape mismatch across batches"
            );
            out_data[idx].extend_from_slice(batch_output.host_data());
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

pub(crate) fn batched_multi_convert<InT: Clone, OutT: Clone, F>(
    buffers: &mut BufferPool,
    input: &TypedTensor<InT>,
    op: F,
) -> Vec<TypedTensor<OutT>>
where
    F: Fn(&mut BufferPool, &TypedTensor<InT>) -> Vec<TypedTensor<OutT>>,
{
    let (core_shape, batch_shape) = split_core_and_batch(input, 2, "batched_multi");
    if batch_shape.is_empty() {
        return op(buffers, input);
    }

    let slice_size: usize = core_shape.iter().product();
    let batch_count: usize = batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_multi: zero-sized batch dims are unsupported"
    );

    let mut out_shapes: Vec<Vec<usize>> = Vec::new();
    let mut out_data: Vec<Vec<OutT>> = Vec::new();

    for batch_idx in 0..batch_count {
        let start = batch_idx * slice_size;
        let end = start + slice_size;
        let batch_input = tensor_from_vec_with_template(
            core_shape.to_vec(),
            input.host_data()[start..end].to_vec(),
            input,
        );
        let batch_outputs = op(buffers, &batch_input);

        if out_shapes.is_empty() {
            out_shapes = batch_outputs
                .iter()
                .map(|tensor| tensor.shape.clone())
                .collect();
            out_data = batch_outputs
                .iter()
                .map(|tensor| Vec::with_capacity(tensor.n_elements() * batch_count))
                .collect();
        } else {
            assert_eq!(
                batch_outputs.len(),
                out_shapes.len(),
                "batched_multi: output count mismatch across batches"
            );
        }

        for (idx, batch_output) in batch_outputs.iter().enumerate() {
            assert_eq!(
                batch_output.shape.as_slice(),
                out_shapes[idx].as_slice(),
                "batched_multi: output core shape mismatch across batches"
            );
            out_data[idx].extend_from_slice(batch_output.host_data());
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

pub(crate) fn batched_binary<T, F>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    op: F,
) -> TypedTensor<T>
where
    T: Clone,
    F: Fn(&mut BufferPool, &TypedTensor<T>, &TypedTensor<T>) -> TypedTensor<T>,
{
    let (a_core_shape, a_batch_shape) = split_core_and_batch(a, 2, "batched_binary");
    let (b_core_shape, b_batch_shape) = split_core_and_batch(b, 2, "batched_binary");
    assert_eq!(
        a_batch_shape, b_batch_shape,
        "batched_binary: batch shape mismatch"
    );

    if a_batch_shape.is_empty() {
        return op(buffers, a, b);
    }

    let a_slice_size: usize = a_core_shape.iter().product();
    let b_slice_size: usize = b_core_shape.iter().product();
    let batch_count: usize = a_batch_shape.iter().product();
    assert!(
        batch_count > 0,
        "batched_binary: zero-sized batch dims are unsupported"
    );

    let mut out_core_shape: Option<Vec<usize>> = None;
    let mut out_data: Option<Vec<T>> = None;

    for batch_idx in 0..batch_count {
        let a_start = batch_idx * a_slice_size;
        let a_end = a_start + a_slice_size;
        let b_start = batch_idx * b_slice_size;
        let b_end = b_start + b_slice_size;

        let batch_a = tensor_from_vec_with_template(
            a_core_shape.to_vec(),
            a.host_data()[a_start..a_end].to_vec(),
            a,
        );
        let batch_b = tensor_from_vec_with_template(
            b_core_shape.to_vec(),
            b.host_data()[b_start..b_end].to_vec(),
            b,
        );
        let batch_output = op(buffers, &batch_a, &batch_b);

        if let Some(expected_shape) = &out_core_shape {
            assert_eq!(
                batch_output.shape.as_slice(),
                expected_shape.as_slice(),
                "batched_binary: output core shape mismatch across batches"
            );
        } else {
            out_data = Some(Vec::with_capacity(batch_output.n_elements() * batch_count));
            out_core_shape = Some(batch_output.shape.clone());
        }

        match &mut out_data {
            Some(data) => data.extend_from_slice(batch_output.host_data()),
            None => panic!("batched_binary: missing output buffer"),
        }
    }

    let mut out_shape = match out_core_shape {
        Some(shape) => shape,
        None => panic!("batched_binary: missing output shape"),
    };
    out_shape.extend_from_slice(a_batch_shape);
    tensor_from_vec_with_template(
        out_shape,
        match out_data {
            Some(data) => data,
            None => panic!("batched_binary: missing output data"),
        },
        b,
    )
}

pub(crate) fn zero_dim_eig_outputs(input: &Tensor) -> Vec<Tensor> {
    let n = input.shape()[0];
    let batch_shape = &input.shape()[2..];
    vec![
        Tensor::C64(TypedTensor::from_vec(
            vector_with_batch_shape(n, batch_shape),
            Vec::new(),
        )),
        Tensor::C64(TypedTensor::from_vec(
            matrix_with_batch_shape(n, n, batch_shape),
            Vec::new(),
        )),
    ]
}
