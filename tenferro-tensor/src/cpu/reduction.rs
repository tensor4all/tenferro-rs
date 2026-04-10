use std::ops::{Add, Mul};

use num_traits::{Float, One, Zero};
use strided_kernel::{reduce_axis, StridedArray};

use crate::types::{Tensor, TypedTensor};

fn backend_failure(op: &'static str, err: impl ToString) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: err.to_string(),
    }
}

fn validate_axes(op: &'static str, axes: &[usize], rank: usize) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis {
                op,
                axis,
                role: "axes",
            });
        }
        seen[axis] = true;
    }
    Ok(())
}

pub fn reduce_sum(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reduce_sum(t, axes)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reduce_sum(t, axes)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_sum(t, axes)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_sum(t, axes)?)),
    }
}

pub fn reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reduce_prod(t, axes)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reduce_prod(t, axes)?)),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_prod(t, axes)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_prod(t, axes)?)),
    }
}

pub fn reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return Ok(input.clone());
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_max(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_max(tensor, axes)?)),
        Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
            op: "reduce_max",
            message: format!("unsupported dtype {:?}", input.dtype()),
        }),
    }
}

pub fn reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return Ok(input.clone());
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_min(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_min(tensor, axes)?)),
        Tensor::C32(_) | Tensor::C64(_) => Err(crate::Error::BackendFailure {
            op: "reduce_min",
            message: format!("unsupported dtype {:?}", input.dtype()),
        }),
    }
}

fn typed_reduce<T, M, R>(
    input: &TypedTensor<T>,
    axes: &[usize],
    map_fn: M,
    reduce_fn: R,
    init: T,
    label: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone,
    M: Fn(T) -> T + Copy,
    R: Fn(T, T) -> T + Copy,
{
    validate_axes(label, axes, input.shape.len())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let output_shape: Vec<usize> = input
        .shape
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, &dim)| dim)
        .collect();

    let mut current = StridedArray::from_parts(
        input.host_data().to_vec(),
        &input.shape,
        &input.strides,
        input.offset,
    )
    .map_err(|err| backend_failure(label, err))?;

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in sorted_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .map_err(|err| backend_failure(label, err))?;
    }

    Ok(TypedTensor::from_vec(output_shape, current.into_data()))
}

pub fn typed_reduce_sum<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T>,
{
    typed_reduce(input, axes, |x| x, |a, b| a + b, T::zero(), "reduce_sum")
}

pub fn typed_reduce_prod<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + One + Mul<Output = T>,
{
    typed_reduce(input, axes, |x| x, |a, b| a * b, T::one(), "reduce_prod")
}

pub fn typed_reduce_max<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
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

pub fn typed_reduce_min<T>(input: &TypedTensor<T>, axes: &[usize]) -> crate::Result<TypedTensor<T>>
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
