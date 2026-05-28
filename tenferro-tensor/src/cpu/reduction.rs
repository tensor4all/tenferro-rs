use std::ops::{Add, Mul};

use num_traits::{Float, One, Zero};
use strided_kernel::reduce_axis;

use super::{typed_view, typed_view_from_view};
use crate::types::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

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
        Tensor::I32(t) => Ok(Tensor::I32(typed_reduce_sum(t, axes)?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_reduce_sum(t, axes)?)),
        Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "reduce_sum",
            "unsupported dtype Bool",
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_sum(t, axes)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_sum(t, axes)?)),
    }
}

pub(crate) fn reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(input) => reduce_sum(input, axes),
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            f32::zero(),
            "reduce_sum",
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            f64::zero(),
            "reduce_sum",
        )?)),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::I32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            i32::zero(),
            "reduce_sum",
        )?)),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::I64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            i64::zero(),
            "reduce_sum",
        )?)),
        TensorRead::View(TensorView::Bool(_)) => Err(crate::Error::backend_failure(
            "reduce_sum",
            "unsupported dtype Bool",
        )),
        TensorRead::View(TensorView::C32(t)) => Ok(Tensor::C32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            num_complex::Complex32::zero(),
            "reduce_sum",
        )?)),
        TensorRead::View(TensorView::C64(t)) => Ok(Tensor::C64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a + b,
            num_complex::Complex64::zero(),
            "reduce_sum",
        )?)),
    }
}

pub fn reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_reduce_prod(t, axes)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_reduce_prod(t, axes)?)),
        Tensor::I32(t) => Ok(Tensor::I32(typed_reduce_prod(t, axes)?)),
        Tensor::I64(t) => Ok(Tensor::I64(typed_reduce_prod(t, axes)?)),
        Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "reduce_prod",
            "unsupported dtype Bool",
        )),
        Tensor::C32(t) => Ok(Tensor::C32(typed_reduce_prod(t, axes)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_reduce_prod(t, axes)?)),
    }
}

pub(crate) fn reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
    match input {
        TensorRead::Tensor(input) => reduce_prod(input, axes),
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            f32::one(),
            "reduce_prod",
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            f64::one(),
            "reduce_prod",
        )?)),
        TensorRead::View(TensorView::I32(t)) => Ok(Tensor::I32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            i32::one(),
            "reduce_prod",
        )?)),
        TensorRead::View(TensorView::I64(t)) => Ok(Tensor::I64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            i64::one(),
            "reduce_prod",
        )?)),
        TensorRead::View(TensorView::Bool(_)) => Err(crate::Error::backend_failure(
            "reduce_prod",
            "unsupported dtype Bool",
        )),
        TensorRead::View(TensorView::C32(t)) => Ok(Tensor::C32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            num_complex::Complex32::one(),
            "reduce_prod",
        )?)),
        TensorRead::View(TensorView::C64(t)) => Ok(Tensor::C64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a * b,
            num_complex::Complex64::one(),
            "reduce_prod",
        )?)),
    }
}

pub fn reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return Ok(input.clone());
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_max(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_max(tensor, axes)?)),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
            Err(crate::Error::backend_failure(
                "reduce_max",
                format!("unsupported dtype {:?}", input.dtype()),
            ))
        }
    }
}

pub(crate) fn reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return match input {
            TensorRead::Tensor(input) => Ok(input.clone()),
            TensorRead::View(input) => view_to_contiguous_tensor(input),
        };
    }

    match input {
        TensorRead::Tensor(input) => reduce_max(input, axes),
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a.max(b),
            f32::neg_infinity(),
            "reduce_max",
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a.max(b),
            f64::neg_infinity(),
            "reduce_max",
        )?)),
        view => Err(crate::Error::backend_failure(
            "reduce_max",
            format!("unsupported dtype {:?}", view.dtype()),
        )),
    }
}

pub fn reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return Ok(input.clone());
    }

    match input {
        Tensor::F32(tensor) => Ok(Tensor::F32(typed_reduce_min(tensor, axes)?)),
        Tensor::F64(tensor) => Ok(Tensor::F64(typed_reduce_min(tensor, axes)?)),
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) | Tensor::C32(_) | Tensor::C64(_) => {
            Err(crate::Error::backend_failure(
                "reduce_min",
                format!("unsupported dtype {:?}", input.dtype()),
            ))
        }
    }
}

pub(crate) fn reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
    if axes.is_empty() {
        return match input {
            TensorRead::Tensor(input) => Ok(input.clone()),
            TensorRead::View(input) => view_to_contiguous_tensor(input),
        };
    }

    match input {
        TensorRead::Tensor(input) => reduce_min(input, axes),
        TensorRead::View(TensorView::F32(t)) => Ok(Tensor::F32(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a.min(b),
            f32::infinity(),
            "reduce_min",
        )?)),
        TensorRead::View(TensorView::F64(t)) => Ok(Tensor::F64(typed_reduce_view(
            &t,
            axes,
            |x| x,
            |a, b| a.min(b),
            f64::infinity(),
            "reduce_min",
        )?)),
        view => Err(crate::Error::backend_failure(
            "reduce_min",
            format!("unsupported dtype {:?}", view.dtype()),
        )),
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
    validate_axes(label, axes, input.shape().len())?;
    if axes.is_empty() {
        return Ok(input.clone());
    }

    let output_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, &dim)| dim)
        .collect();

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    let Some((&first_axis, remaining_axes)) = sorted_axes.split_first() else {
        return Ok(input.clone());
    };

    let input_view = typed_view(label, input)?;
    let mut current = reduce_axis(&input_view, first_axis, map_fn, reduce_fn, init)
        .map_err(|err| crate::Error::backend_failure(label, err))?;

    for &axis in remaining_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .map_err(|err| crate::Error::backend_failure(label, err))?;
    }

    Ok(TypedTensor::from_vec_col_major(
        output_shape,
        current.into_data(),
    ))
}

pub(crate) fn typed_reduce_view<T, M, R, TR>(
    input: &TypedTensorView<'_, T, TR>,
    axes: &[usize],
    map_fn: M,
    reduce_fn: R,
    init: T,
    label: &'static str,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + 'static,
    M: Fn(T) -> T + Copy,
    R: Fn(T, T) -> T + Copy,
    TR: TensorRank,
{
    validate_axes(label, axes, input.shape().len())?;
    if axes.is_empty() {
        return view_to_dyn_contiguous(input);
    }

    let output_shape: Vec<usize> = input
        .shape()
        .iter()
        .enumerate()
        .filter(|(axis, _)| !axes.contains(axis))
        .map(|(_, &dim)| dim)
        .collect();

    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
    let Some((&first_axis, remaining_axes)) = sorted_axes.split_first() else {
        return view_to_dyn_contiguous(input);
    };

    let input_view = typed_view_from_view(label, input)?;
    let mut current = reduce_axis(&input_view, first_axis, map_fn, reduce_fn, init)
        .map_err(|err| crate::Error::backend_failure(label, err))?;

    for &axis in remaining_axes {
        current = reduce_axis(&current.view(), axis, map_fn, reduce_fn, init)
            .map_err(|err| crate::Error::backend_failure(label, err))?;
    }

    Ok(TypedTensor::from_vec_col_major(
        output_shape,
        current.into_data(),
    ))
}

fn view_to_dyn_contiguous<T, R>(input: &TypedTensorView<'_, T, R>) -> crate::Result<TypedTensor<T>>
where
    T: Clone + 'static,
    R: TensorRank,
{
    let compact = input.to_contiguous()?;
    let (shape, data) = compact.try_into_vec_col_major()?;
    Ok(TypedTensor::from_vec_col_major(shape, data))
}

fn view_to_contiguous_tensor(input: TensorView<'_>) -> crate::Result<Tensor> {
    match input {
        TensorView::F32(t) => Ok(Tensor::F32(view_to_dyn_contiguous(&t)?)),
        TensorView::F64(t) => Ok(Tensor::F64(view_to_dyn_contiguous(&t)?)),
        TensorView::I32(t) => Ok(Tensor::I32(view_to_dyn_contiguous(&t)?)),
        TensorView::I64(t) => Ok(Tensor::I64(view_to_dyn_contiguous(&t)?)),
        TensorView::Bool(t) => Ok(Tensor::Bool(view_to_dyn_contiguous(&t)?)),
        TensorView::C32(t) => Ok(Tensor::C32(view_to_dyn_contiguous(&t)?)),
        TensorView::C64(t) => Ok(Tensor::C64(view_to_dyn_contiguous(&t)?)),
    }
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
