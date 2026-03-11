use std::any::TypeId;
use std::cmp::Ordering;
use std::ops::{Add, Div, Mul};

use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::{Float, NumCast, Zero};
use strided_kernel::{map_into, zip_map2_into};
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::{for_each_index, validate_rank, validate_shape_count, validate_shape_eq};

pub(crate) trait CpuScalarValue:
    Scalar
    + ComplexFloat
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + Zero
{
    fn from_real(real: Self::Real) -> Self;
}

pub(crate) trait ComplexCpuScalarValue: CpuScalarValue {
    fn pow_complex(self, exponent: Self) -> Self;
}

impl CpuScalarValue for f32 {
    fn from_real(real: Self::Real) -> Self {
        real
    }
}

impl CpuScalarValue for f64 {
    fn from_real(real: Self::Real) -> Self {
        real
    }
}

impl CpuScalarValue for Complex32 {
    fn from_real(real: Self::Real) -> Self {
        Complex32::new(real, 0.0)
    }
}

impl ComplexCpuScalarValue for Complex32 {
    fn pow_complex(self, exponent: Self) -> Self {
        self.powc(exponent)
    }
}

impl CpuScalarValue for Complex64 {
    fn from_real(real: Self::Real) -> Self {
        Complex64::new(real, 0.0)
    }
}

impl ComplexCpuScalarValue for Complex64 {
    fn pow_complex(self, exponent: Self) -> Self {
        self.powc(exponent)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReductionPlanSpec {
    pub(crate) reduced_axes: Vec<usize>,
    pub(crate) reduced_total: usize,
}

pub(crate) fn validate_pointwise_shapes(
    shapes: &[&[usize]],
    arity: usize,
    op_name: &str,
) -> Result<()> {
    validate_shape_count(shapes, arity + 1, op_name)?;
    let output_shape = shapes[arity];
    for (idx, shape) in shapes[..arity].iter().enumerate() {
        validate_shape_eq(shape, output_shape, &format!("{op_name} input {idx}"))?;
    }
    Ok(())
}

pub(crate) fn plan_reduction(
    modes_a: &[u32],
    modes_c: &[u32],
    shapes: &[&[usize]],
    op_name: &str,
) -> Result<ReductionPlanSpec> {
    validate_shape_count(shapes, 2, op_name)?;
    validate_rank(shapes[0], modes_a.len(), &format!("{op_name} input"))?;
    validate_rank(shapes[1], modes_c.len(), &format!("{op_name} output"))?;

    let mut expected_output = Vec::with_capacity(modes_c.len());
    for &mode in modes_c {
        let Some(axis) = modes_a.iter().position(|&candidate| candidate == mode) else {
            return Err(Error::InvalidArgument(format!(
                "{op_name}: output mode {mode} not found in input modes {modes_a:?}"
            )));
        };
        expected_output.push(shapes[0][axis]);
    }
    validate_shape_eq(shapes[1], &expected_output, &format!("{op_name} output"))?;

    let reduced_axes: Vec<usize> = modes_a
        .iter()
        .enumerate()
        .filter(|(_, mode)| !modes_c.contains(mode))
        .map(|(idx, _)| idx)
        .collect();
    let reduced_total = reduced_axes
        .iter()
        .map(|&axis| shapes[0][axis])
        .product::<usize>();

    Ok(ReductionPlanSpec {
        reduced_axes,
        reduced_total,
    })
}

pub(crate) fn execute_unary_map<S, F>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    f: F,
) -> Result<()>
where
    S: CpuScalarValue,
    F: Fn(S) -> S + Copy,
{
    if beta == S::zero() {
        let alpha_value = alpha;
        map_into(output, input, move |x| alpha_value * f(x))
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        return Ok(());
    }

    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let value = alpha * f(input.get(idx));
        output.set(idx, value + beta * output.get(idx));
    });
    Ok(())
}

pub(crate) fn execute_binary_map<S, F>(
    alpha: S,
    lhs: &StridedView<S>,
    rhs: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    f: F,
) -> Result<()>
where
    S: CpuScalarValue,
    F: Fn(S, S) -> S + Copy,
{
    if beta == S::zero() {
        let alpha_value = alpha;
        zip_map2_into(output, lhs, rhs, move |x, y| alpha_value * f(x, y))
            .map_err(|err| Error::DeviceError(err.to_string()))?;
        return Ok(());
    }

    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let value = alpha * f(lhs.get(idx), rhs.get(idx));
        output.set(idx, value + beta * output.get(idx));
    });
    Ok(())
}

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

fn unflatten_index_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    for (axis, &dim) in dims.iter().enumerate() {
        out[axis] = flat % dim;
        flat /= dim;
    }
}

pub(crate) fn execute_sum_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| in_dims[axis]).collect();
    let reduced_total: usize = reduced_dims.iter().product();

    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut sum = S::zero();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            sum = sum + input.get(&in_idx);
        }
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * sum + old);
    });

    Ok(())
}

pub(crate) fn execute_prod_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    let in_dims = input.dims().to_vec();
    let out_dims = output.dims().to_vec();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| in_dims[axis]).collect();
    let reduced_total: usize = reduced_dims.iter().product();

    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; in_dims.len()];

    for_each_index(&out_dims, |out_idx| {
        let mut prod = S::one();
        for red_flat in 0..reduced_total {
            unflatten_index_into(red_flat, &reduced_dims, &mut red_idx);
            build_reduction_input_index(out_idx, &red_idx, reduced_axes, &mut in_idx);
            prod = prod * input.get(&in_idx);
        }
        let old = if beta == S::zero() {
            S::zero()
        } else {
            beta * output.get(out_idx)
        };
        output.set(out_idx, alpha * prod + old);
    });

    Ok(())
}

pub(crate) fn execute_mean_reduction<S: CpuScalarValue>(
    alpha: S,
    input: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    reduced_axes: &[usize],
) -> Result<()> {
    let scale = scalar_from_usize::<S>(
        reduced_axes
            .iter()
            .map(|&axis| input.dims()[axis])
            .product(),
    )?;
    let mean_scale = S::one() / scale;
    execute_sum_reduction(alpha * mean_scale, input, beta, output, reduced_axes)
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

pub(crate) fn is_supported_scalar_type<T: Scalar + 'static>() -> bool {
    let tid = TypeId::of::<T>();
    tid == TypeId::of::<f32>()
        || tid == TypeId::of::<f64>()
        || tid == TypeId::of::<Complex32>()
        || tid == TypeId::of::<Complex64>()
}

pub(crate) fn is_supported_ordered_real_type<T: Scalar + 'static>() -> bool {
    let tid = TypeId::of::<T>();
    tid == TypeId::of::<f32>() || tid == TypeId::of::<f64>()
}
