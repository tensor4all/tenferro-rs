use std::ops::{Add, Div, Mul};

use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::Zero;
use strided_kernel::{map_into, zip_map2_into, zip_map3_into};
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::infra::typed_dispatch::{dispatch_real_scalar_type, dispatch_standard_scalar_type};
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

pub(crate) fn execute_ternary_map<S, F>(
    alpha: S,
    cond: &StridedView<S>,
    on_true: &StridedView<S>,
    on_false: &StridedView<S>,
    beta: S,
    output: &mut StridedViewMut<S>,
    f: F,
) -> Result<()>
where
    S: CpuScalarValue,
    F: Fn(S, S, S) -> S + Copy,
{
    if beta == S::zero() {
        let alpha_value = alpha;
        zip_map3_into(output, cond, on_true, on_false, move |c, t, f_value| {
            alpha_value * f(c, t, f_value)
        })
        .map_err(|err| Error::DeviceError(err.to_string()))?;
        return Ok(());
    }

    let dims = output.dims().to_vec();
    for_each_index(&dims, |idx| {
        let value = alpha * f(cond.get(idx), on_true.get(idx), on_false.get(idx));
        output.set(idx, value + beta * output.get(idx));
    });
    Ok(())
}

/// Unflatten a linear index into a pre-allocated buffer (column-major).
pub(super) fn unflatten_index_into(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    debug_assert!(
        flat < dims.iter().product::<usize>(),
        "flat index {flat} out of range for dims {dims:?}"
    );
    for d in 0..dims.len() {
        out[d] = flat % dims[d];
        flat /= dims[d];
    }
}

pub(crate) fn is_supported_scalar_type<T: Scalar + 'static>() -> bool {
    dispatch_standard_scalar_type!(T, _Concrete, { return true });
    false
}

pub(crate) fn is_supported_ordered_real_type<T: Scalar + 'static>() -> bool {
    dispatch_real_scalar_type!(T, _Concrete, { return true });
    false
}
