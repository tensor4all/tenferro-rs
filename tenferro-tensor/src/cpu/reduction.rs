use std::ops::Add;

use num_traits::Zero;
use strided_kernel::{col_major_strides, reduce_axis, StridedArray};

use crate::types::{dispatch_tensor, Tensor, TypedTensor};

use super::tensor_from_array;

pub fn reduce_sum(input: &Tensor, axes: &[usize]) -> Tensor {
    dispatch_tensor!(input, t => typed_reduce_sum(t, axes))
}

pub fn reduce_prod(_input: &Tensor, _axes: &[usize]) -> Tensor {
    todo!("reduce_prod")
}

pub fn reduce_max(_input: &Tensor, _axes: &[usize]) -> Tensor {
    todo!("reduce_max")
}

pub fn reduce_min(_input: &Tensor, _axes: &[usize]) -> Tensor {
    todo!("reduce_min")
}

pub fn typed_reduce_sum<T>(input: &TypedTensor<T>, axes: &[usize]) -> TypedTensor<T>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    if axes.is_empty() {
        return input.clone();
    }

    let strides = col_major_strides(&input.shape);
    let mut current =
        StridedArray::from_parts(input.host_data().to_vec(), &input.shape, &strides, 0)
            .expect("reduce_sum input");

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in sorted_axes {
        current =
            reduce_axis(&current.view(), axis, |x| x, |a, b| a + b, T::zero()).expect("reduce_sum");
    }

    tensor_from_array(current)
}
