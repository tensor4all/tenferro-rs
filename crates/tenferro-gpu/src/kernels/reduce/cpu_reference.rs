//! CPU reference implementations for CubeCL reduction tests.

/// Reduce an `i64` tensor by summing one axis with keepdims shape semantics.
///
/// The input slice is interpreted in column-major order. The returned storage
/// has the same rank as `input_shape`, with the reduced axis length set to one.
///
/// # Panics
///
/// Panics if `axis` is out of bounds for `input_shape` or if `input.len()` does
/// not match the product of the dimensions in `input_shape`.
///
pub(crate) fn reduce_sum_i64_keepdims(
    input: &[i64],
    input_shape: &[usize],
    axis: usize,
) -> Vec<i64> {
    // Integer `reduce_sum` is a CPU/CUDA parity contract: use explicit
    // two's-complement wrapping, never bare `+` on user data
    // (REPOSITORY_RULES.md, integer arithmetic parity rule). The CUDA
    // `reduce_sum_int` kernel uses the same `wrapping_add`.
    reduce_keepdims(input, input_shape, axis, 0_i64, |acc, value| {
        acc.wrapping_add(value)
    })
}

fn reduce_keepdims<T, F>(
    input: &[T],
    input_shape: &[usize],
    axis: usize,
    identity: T,
    combine: F,
) -> Vec<T>
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    assert!(
        axis < input_shape.len(),
        "axis {axis} is out of bounds for rank {}",
        input_shape.len()
    );
    assert_eq!(
        input.len(),
        input_shape.iter().product::<usize>(),
        "input length does not match input shape"
    );

    let reduce_len = input_shape[axis];
    let axis_stride = column_major_stride(input_shape, axis);
    let output_shape = keepdims_shape(input_shape, axis);
    let output_len = output_shape.iter().product();
    let mut output = Vec::with_capacity(output_len);

    for output_index in 0..output_len {
        let input_base = output_linear_to_input_base(output_index, input_shape, axis);
        let mut acc = identity;

        for reduce_index in 0..reduce_len {
            let input_index = input_base + reduce_index * axis_stride;
            acc = combine(acc, input[input_index]);
        }

        output.push(acc);
    }

    output
}

fn keepdims_shape(input_shape: &[usize], axis: usize) -> Vec<usize> {
    let mut output_shape = input_shape.to_vec();
    output_shape[axis] = 1;
    output_shape
}

fn column_major_stride(input_shape: &[usize], axis: usize) -> usize {
    input_shape.iter().take(axis).product()
}

fn output_linear_to_input_base(
    mut output_index: usize,
    input_shape: &[usize],
    axis: usize,
) -> usize {
    let mut input_offset = 0;
    let mut input_stride = 1;

    for (dim, dim_len) in input_shape.iter().copied().enumerate() {
        let output_dim_len = if dim == axis { 1 } else { dim_len };
        let coord = output_index % output_dim_len;
        output_index /= output_dim_len;
        input_offset += coord * input_stride;
        input_stride *= dim_len;
    }

    input_offset
}
