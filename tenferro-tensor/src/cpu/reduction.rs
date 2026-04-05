use std::ops::{Add, Mul};

use num_traits::{Float, One, Zero};
use strided_kernel::{col_major_strides, reduce_axis, StridedArray};

use crate::types::{dispatch_tensor, Tensor, TypedTensor};

use super::tensor_from_array;

pub fn reduce_sum(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_reduce_sum(t, axes))
}

pub fn reduce_prod(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_reduce_prod(t, axes))
}

pub fn reduce_max(input: &Tensor, axes: &[usize]) -> Tensor {
    if axes.is_empty() {
        return input.clone();
    }

    match input {
        Tensor::F32(tensor) => Tensor::F32(typed_reduce_max(tensor, axes)),
        Tensor::F64(tensor) => Tensor::F64(typed_reduce_max(tensor, axes)),
        Tensor::C32(_) | Tensor::C64(_) => {
            panic!("reduce_max is only implemented for real tensors")
        }
    }
}

pub fn reduce_min(input: &Tensor, axes: &[usize]) -> Tensor {
    if axes.is_empty() {
        return input.clone();
    }

    match input {
        Tensor::F32(tensor) => Tensor::F32(typed_reduce_min(tensor, axes)),
        Tensor::F64(tensor) => Tensor::F64(typed_reduce_min(tensor, axes)),
        Tensor::C32(_) | Tensor::C64(_) => {
            panic!("reduce_min is only implemented for real tensors")
        }
    }
}

fn typed_reduce<T, M, R>(
    input: &TypedTensor<T>,
    axes: &[usize],
    map_fn: M,
    reduce_fn: R,
    init: T,
    label: &str,
) -> TypedTensor<T>
where
    T: Copy + Clone,
    M: Fn(T) -> T + Copy,
    R: Fn(T, T) -> T + Copy,
{
    if axes.is_empty() {
        return input.clone();
    }

    let strides = col_major_strides(&input.shape);
    let mut current =
        StridedArray::from_parts(input.host_data().to_vec(), &input.shape, &strides, 0)
            .unwrap_or_else(|err| panic!("{label} input: {err}"));

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in sorted_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .unwrap_or_else(|err| panic!("{label}: {err}"));
    }

    tensor_from_array(current)
}

pub fn typed_reduce_sum<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    typed_reduce(input, axes, |x| x, |a, b| a + b, T::zero(), "reduce_sum")
}

pub fn typed_reduce_prod<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Copy + Clone + One + Mul<Output = T>,
{
    typed_reduce(input, axes, |x| x, |a, b| a * b, T::one(), "reduce_prod")
}

pub fn typed_reduce_max<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Float,
{
    typed_reduce(
        input,
        axes,
        |x| x,
        |a, b| a.max(b),
        T::neg_infinity(),
        "reduce_max",
    )
}

pub fn typed_reduce_min<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Float,
{
    typed_reduce(
        input,
        axes,
        |x| x,
        |a, b| a.min(b),
        T::infinity(),
        "reduce_min",
    )
}
