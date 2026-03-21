use num_complex::{Complex32, Complex64};
use num_traits::NumCast;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};

use crate::infra::typed_dispatch::{
    cast_scalar_value, dispatch_complex_scalar_type, dispatch_real_scalar_type,
};
use crate::{validate_rank, validate_shape_count, validate_shape_eq};

pub(super) struct ReductionPlanSpec {
    pub(super) reduced_total: usize,
}

pub(super) fn validate_pointwise_shapes(
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

pub(super) fn plan_reduction_shapes(
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

    let reduced_total = modes_a
        .iter()
        .enumerate()
        .filter(|(_, mode)| !modes_c.contains(mode))
        .map(|(axis, _)| shapes[0][axis])
        .product::<usize>();
    Ok(ReductionPlanSpec { reduced_total })
}

pub(super) fn supports_real_scalar_type<T: Scalar + 'static>() -> bool {
    dispatch_real_scalar_type!(T, _Concrete, { return true });
    false
}

pub(super) fn supports_complex_scalar_type<T: Scalar + 'static>() -> bool {
    dispatch_complex_scalar_type!(T, _Concrete, { return true });
    false
}

pub(super) fn scale_standard_alpha<T: Scalar + 'static>(alpha: T, divisor: usize) -> Result<T> {
    dispatch_real_scalar_type!(T, Concrete, {
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let Some(scale) = <Concrete as NumCast>::from(divisor) else {
            return Err(Error::DeviceError(format!(
                "Failed to cast divisor {divisor} for CUDA reduction scaling"
            )));
        };
        return Ok(cast_scalar_value!(alpha / scale, Concrete, T));
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let scale = if std::mem::size_of::<Concrete>() == std::mem::size_of::<Complex32>() {
            <f32 as NumCast>::from(divisor)
                .map(|value| cast_scalar_value!(Complex32::new(value, 0.0), Complex32, Concrete))
        } else {
            <f64 as NumCast>::from(divisor)
                .map(|value| cast_scalar_value!(Complex64::new(value, 0.0), Complex64, Concrete))
        };
        let Some(scale) = scale else {
            return Err(Error::DeviceError(format!(
                "Failed to cast divisor {divisor} for CUDA complex reduction scaling"
            )));
        };
        return Ok(cast_scalar_value!(alpha / scale, Concrete, T));
    });

    Err(Error::InvalidArgument(format!(
        "CUDA reduction scaling is not supported for {}",
        std::any::type_name::<T>()
    )))
}

pub(super) fn scale_real_alpha<T: Scalar + 'static>(alpha: T, divisor: usize) -> Result<T> {
    if !supports_real_scalar_type::<T>() {
        return Err(Error::InvalidArgument(format!(
            "CUDA real reduction scaling is not supported for {}",
            std::any::type_name::<T>()
        )));
    }
    scale_standard_alpha(alpha, divisor)
}
