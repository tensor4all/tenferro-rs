use std::cmp::Ordering;

use num_traits::{Float, NumCast};
use strided_kernel::{map_into, reduce, reduce_axis};
use strided_view::{StridedArray, StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::cpu::common::{execute_unary_map, unflatten_index_into, CpuScalarValue};
use crate::for_each_index;

fn scalar_from_usize<S: CpuScalarValue>(value: usize) -> Result<S> {
    let Some(real) = <S::Real as NumCast>::from(value) else {
        return Err(Error::InvalidArgument(format!(
            "cannot represent reduction size {value} in scalar type {}",
            std::any::type_name::<S>()
        )));
    };
    Ok(S::from_real(real))
}

fn build_reduction_input_index(
    out_idx: &[usize],
    red_idx: &[usize],
    reduced_axes: &[usize],
    in_idx: &mut [usize],
) {
    let mut out_pos = 0usize;
    let mut red_pos = 0usize;
    for (axis, slot) in in_idx.iter_mut().enumerate() {
        if red_pos < reduced_axes.len() && reduced_axes[red_pos] == axis {
            *slot = red_idx[red_pos];
            red_pos += 1;
        } else {
            *slot = out_idx[out_pos];
            out_pos += 1;
        }
    }
}

fn apply_reduction_array<S: CpuScalarValue>(
    alpha: S,
    reduced: &StridedArray<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
) -> Result<()> {
    let reduced_view = reduced.view();
    if beta == S::zero() {
        let alpha_value = alpha;
        map_into(output, &reduced_view, move |x| alpha_value * x)
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        return Ok(());
    }

    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let value = alpha * reduced_view.get(idx);
        output.set(idx, value + beta * output.get(idx));
    });
    Ok(())
}

fn reduce_axes_same_type<S, F>(
    input: &StridedView<S>,
    reduced_axes: &[usize],
    init: S,
    reduce_fn: F,
) -> Result<Option<StridedArray<S>>>
where
    S: CpuScalarValue,
    F: Fn(S, S) -> S + Copy + Sync,
{
    if reduced_axes.is_empty() {
        return Ok(None);
    }

    let mut axes = reduced_axes.to_vec();
    axes.sort_unstable_by(|a, b| b.cmp(a));
    let mut current = reduce_axis(input, axes[0], |x| x, reduce_fn, init)
        .map_err(|err| Error::DeviceError(err.to_string()))?;
    for &axis in axes.iter().skip(1) {
        current = reduce_axis(&current.view(), axis, |x| x, reduce_fn, init)
            .map_err(|err| Error::DeviceError(err.to_string()))?;
    }
    Ok(Some(current))
}

pub(crate) fn execute_sum_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    if reduced_axes.is_empty() {
        return execute_unary_map(alpha, input, beta, output, |x| x);
    }

    if reduced_axes.len() == input.ndim() {
        let sum = reduce(input, |x| x, |a, b| a + b, S::zero())
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(&[])
        };
        output.set(&[], alpha * sum + old);
        return Ok(());
    }

    let Some(reduced) = reduce_axes_same_type(input, reduced_axes, S::zero(), |a, b| a + b)? else {
        return Err(Error::InvalidArgument(
            "sum reduction requires at least one reduced axis".into(),
        ));
    };
    apply_reduction_array(alpha, &reduced, beta, output)
}

pub(crate) fn execute_prod_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    if reduced_axes.is_empty() {
        return execute_unary_map(alpha, input, beta, output, |x| x);
    }

    if reduced_axes.len() == input.ndim() {
        let prod = reduce(input, |x| x, |a, b| a * b, S::one())
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(&[])
        };
        output.set(&[], alpha * prod + old);
        return Ok(());
    }

    let Some(reduced) = reduce_axes_same_type(input, reduced_axes, S::one(), |a, b| a * b)? else {
        return Err(Error::InvalidArgument(
            "prod reduction requires at least one reduced axis".into(),
        ));
    };
    apply_reduction_array(alpha, &reduced, beta, output)
}

pub(crate) fn execute_mean_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    if reduced_axes.is_empty() {
        return execute_unary_map(alpha, input, beta, output, |x| x);
    }

    let scale = scalar_from_usize::<S>(
        reduced_axes
            .iter()
            .map(|&axis| input.dims()[axis])
            .product(),
    )?;
    let mean_scale = S::one() / scale;

    if reduced_axes.len() == input.ndim() {
        let sum = reduce(input, |x| x, |a, b| a + b, S::zero())
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(&[])
        };
        output.set(&[], alpha * mean_scale * sum + old);
        return Ok(());
    }

    let Some(reduced) = reduce_axes_same_type(input, reduced_axes, S::zero(), |a, b| a + b)? else {
        return Err(Error::InvalidArgument(
            "mean reduction requires at least one reduced axis".into(),
        ));
    };
    apply_reduction_array(alpha * mean_scale, &reduced, beta, output)
}

pub(crate) fn execute_variance_reduction<S>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()>
where
    S: CpuScalarValue + Float,
{
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| in_dims[axis]).collect();
    let reduced_total: usize = reduced_dims.iter().product();
    if reduced_total == 0 {
        return Err(Error::InvalidArgument(
            "variance reduction requires a non-empty reduction domain".into(),
        ));
    }

    let scale = scalar_from_usize::<S>(reduced_total)?;
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = S::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            sum = sum + input.get(&in_idx);
        }
        let mean = sum / scale;

        let mut sq_sum = S::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            let delta = input.get(&in_idx) - mean;
            sq_sum = sq_sum + delta * delta;
        }
        let variance = sq_sum / scale;
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * variance + old);
    });

    Ok(())
}

pub(crate) fn execute_std_reduction<S>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()>
where
    S: CpuScalarValue + Float,
{
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| in_dims[axis]).collect();
    let reduced_total: usize = reduced_dims.iter().product();
    if reduced_total == 0 {
        return Err(Error::InvalidArgument(
            "std reduction requires a non-empty reduction domain".into(),
        ));
    }

    let scale = scalar_from_usize::<S>(reduced_total)?;
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = S::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            sum = sum + input.get(&in_idx);
        }
        let mean = sum / scale;

        let mut sq_sum = S::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            let delta = input.get(&in_idx) - mean;
            sq_sum = sq_sum + delta * delta;
        }
        let std = Float::sqrt(sq_sum / scale);
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * std + old);
    });

    Ok(())
}

fn prefer_extrema_candidate<S: Scalar + Float>(
    candidate: S,
    current: S,
    want_max: bool,
) -> Result<bool> {
    match candidate.partial_cmp(&current) {
        Some(Ordering::Greater) => Ok(want_max),
        Some(Ordering::Less) => Ok(!want_max),
        Some(Ordering::Equal) => Ok(false),
        None => Err(Error::InvalidArgument(
            "extrema reduction encountered unordered values".into(),
        )),
    }
}

pub(crate) fn execute_extrema_reduction<S>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
    want_max: bool,
) -> Result<()>
where
    S: Scalar + Float,
{
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| in_dims[axis]).collect();
    let reduced_total: usize = reduced_dims.iter().product();
    if reduced_total == 0 {
        return Err(Error::InvalidArgument(
            "extrema reduction requires a non-empty reduction domain".into(),
        ));
    }

    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];
    let mut error = None;

    for_each_index(&out_dims, |out_idx| {
        if error.is_some() {
            return;
        }

        let mut best = None;
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            let candidate = input.get(&in_idx);
            match best {
                None => best = Some(candidate),
                Some(current) => match prefer_extrema_candidate(candidate, current, want_max) {
                    Ok(true) => best = Some(candidate),
                    Ok(false) => best = Some(current),
                    Err(err) => {
                        error = Some(err);
                        return;
                    }
                },
            }
        }

        let Some(best) = best else {
            error = Some(Error::InvalidArgument(
                "extrema reduction requires a non-empty reduction domain".into(),
            ));
            return;
        };
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * best + old);
    });

    if let Some(err) = error {
        return Err(err);
    }
    Ok(())
}
