use num_complex::ComplexFloat;
use num_traits::Float;
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cpu::common::{
    execute_binary_map, execute_unary_map, is_supported_ordered_real_type,
    is_supported_scalar_type, plan_reduction, validate_pointwise_shapes, ComplexCpuScalarValue,
    CpuScalarValue, ReductionPlanSpec,
};
use crate::cpu::family_reduction::{execute_std_reduction, execute_variance_reduction};
use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::infra::typed_dispatch::{
    cast_scalar_value, cast_strided_view, cast_strided_view_mut, dispatch_complex_scalar_type,
    dispatch_real_scalar_type, dispatch_standard_scalar_type,
};
use crate::{
    validate_execute_inputs, AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticReductionOp,
    AnalyticUnaryOp, CpuBackend, CpuContext, TensorAnalyticPrims,
};

/// CPU execution plan for the analytic protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuAnalyticPlan;
/// let _ = std::mem::size_of::<CpuAnalyticPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuAnalyticPlan {
    PointwiseUnary {
        op: AnalyticUnaryOp,
    },
    PointwiseBinary {
        op: AnalyticBinaryOp,
    },
    Reduction {
        reduced_axes: Vec<usize>,
        op: AnalyticReductionOp,
    },
}

fn supports_analytic_unary<S: Scalar + 'static>(op: AnalyticUnaryOp) -> bool {
    is_supported_scalar_type::<S>()
        && matches!(
            op,
            AnalyticUnaryOp::Sqrt
                | AnalyticUnaryOp::Rsqrt
                | AnalyticUnaryOp::Exp
                | AnalyticUnaryOp::Expm1
                | AnalyticUnaryOp::Log
                | AnalyticUnaryOp::Log1p
                | AnalyticUnaryOp::Sin
                | AnalyticUnaryOp::Cos
                | AnalyticUnaryOp::Tan
                | AnalyticUnaryOp::Tanh
                | AnalyticUnaryOp::Asin
                | AnalyticUnaryOp::Acos
                | AnalyticUnaryOp::Atan
                | AnalyticUnaryOp::Sinh
                | AnalyticUnaryOp::Cosh
                | AnalyticUnaryOp::Asinh
                | AnalyticUnaryOp::Acosh
                | AnalyticUnaryOp::Atanh
        )
}

fn supports_analytic_binary<S: Scalar + 'static>(op: AnalyticBinaryOp) -> bool {
    match op {
        AnalyticBinaryOp::Pow | AnalyticBinaryOp::Xlogy => is_supported_scalar_type::<S>(),
        AnalyticBinaryOp::Atan2 | AnalyticBinaryOp::Hypot => is_supported_ordered_real_type::<S>(),
    }
}

fn supports_analytic_reduction<S: Scalar + 'static>(op: AnalyticReductionOp) -> bool {
    match op {
        AnalyticReductionOp::Var | AnalyticReductionOp::Std => {
            is_supported_ordered_real_type::<S>()
        }
    }
}

fn execute_analytic_unary_typed<S: CpuScalarValue>(
    alpha: S,
    input: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticUnaryOp,
) -> Result<()> {
    match op {
        AnalyticUnaryOp::Sqrt => execute_unary_map(alpha, input, beta, output, |x| x.sqrt()),
        AnalyticUnaryOp::Rsqrt => {
            execute_unary_map(alpha, input, beta, output, |x| S::one() / x.sqrt())
        }
        AnalyticUnaryOp::Exp => execute_unary_map(alpha, input, beta, output, |x| x.exp()),
        AnalyticUnaryOp::Expm1 => {
            execute_unary_map(alpha, input, beta, output, |x| x.exp() - S::one())
        }
        AnalyticUnaryOp::Log => execute_unary_map(alpha, input, beta, output, |x| x.ln()),
        AnalyticUnaryOp::Log1p => {
            execute_unary_map(alpha, input, beta, output, |x| (x + S::one()).ln())
        }
        AnalyticUnaryOp::Sin => execute_unary_map(alpha, input, beta, output, |x| x.sin()),
        AnalyticUnaryOp::Cos => execute_unary_map(alpha, input, beta, output, |x| x.cos()),
        AnalyticUnaryOp::Tan => execute_unary_map(alpha, input, beta, output, |x| x.tan()),
        AnalyticUnaryOp::Tanh => execute_unary_map(alpha, input, beta, output, |x| x.tanh()),
        AnalyticUnaryOp::Asin => execute_unary_map(alpha, input, beta, output, |x| x.asin()),
        AnalyticUnaryOp::Acos => execute_unary_map(alpha, input, beta, output, |x| x.acos()),
        AnalyticUnaryOp::Atan => execute_unary_map(alpha, input, beta, output, |x| x.atan()),
        AnalyticUnaryOp::Sinh => execute_unary_map(alpha, input, beta, output, |x| x.sinh()),
        AnalyticUnaryOp::Cosh => execute_unary_map(alpha, input, beta, output, |x| x.cosh()),
        AnalyticUnaryOp::Asinh => execute_unary_map(alpha, input, beta, output, |x| x.asinh()),
        AnalyticUnaryOp::Acosh => execute_unary_map(alpha, input, beta, output, |x| x.acosh()),
        AnalyticUnaryOp::Atanh => execute_unary_map(alpha, input, beta, output, |x| x.atanh()),
    }
}

fn execute_analytic_binary_real<S: Float + CpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    match op {
        AnalyticBinaryOp::Pow => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| Float::powf(x, y))
        }
        AnalyticBinaryOp::Atan2 => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.atan2(y))
        }
        AnalyticBinaryOp::Hypot => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.hypot(y))
        }
        AnalyticBinaryOp::Xlogy => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| {
            if x == S::zero() {
                S::zero()
            } else {
                x * Float::ln(y)
            }
        }),
    }
}

fn execute_analytic_binary_complex<S: ComplexCpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    match op {
        AnalyticBinaryOp::Pow => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.pow_complex(y))
        }
        AnalyticBinaryOp::Xlogy => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| {
            if x == S::zero() {
                S::zero()
            } else {
                x * ComplexFloat::ln(y)
            }
        }),
        _ => Err(Error::InvalidArgument(format!(
            "analytic binary operation {op:?} requires ordered real scalars"
        ))),
    }
}

fn execute_analytic_unary<T: Scalar + 'static>(
    alpha: T,
    input: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: AnalyticUnaryOp,
) -> Result<()> {
    dispatch_standard_scalar_type!(T, Concrete, {
        let input = cast_strided_view!(input, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_analytic_unary_typed(alpha, input, beta, output, op);
    });

    Err(Error::InvalidArgument(format!(
        "analytic unary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

fn execute_analytic_binary<T: Scalar + 'static>(
    alpha: T,
    lhs: &strided_view::StridedView<T>,
    rhs: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    dispatch_real_scalar_type!(T, Concrete, {
        let lhs = cast_strided_view!(lhs, T, Concrete);
        let rhs = cast_strided_view!(rhs, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_analytic_binary_real(alpha, lhs, rhs, beta, output, op);
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let lhs = cast_strided_view!(lhs, T, Concrete);
        let rhs = cast_strided_view!(rhs, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_analytic_binary_complex(alpha, lhs, rhs, beta, output, op);
    });

    Err(Error::InvalidArgument(format!(
        "analytic binary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

fn execute_analytic_reduction_real<S: Float + CpuScalarValue>(
    alpha: S,
    input: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    reduced_axes: &[usize],
    op: AnalyticReductionOp,
) -> Result<()> {
    match op {
        AnalyticReductionOp::Var => {
            execute_variance_reduction(alpha, input, beta, output, reduced_axes)
        }
        AnalyticReductionOp::Std => execute_std_reduction(alpha, input, beta, output, reduced_axes),
    }
}

fn execute_analytic_reduction<T: Scalar + 'static>(
    alpha: T,
    input: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    reduced_axes: &[usize],
    op: AnalyticReductionOp,
) -> Result<()> {
    dispatch_real_scalar_type!(T, Concrete, {
        let input = cast_strided_view!(input, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_analytic_reduction_real(alpha, input, beta, output, reduced_axes, op);
    });

    Err(Error::InvalidArgument(format!(
        "analytic reduction {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

impl<S: Scalar + 'static> TensorAnalyticPrims<Standard<S>> for CpuBackend {
    type Plan = CpuAnalyticPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        match desc {
            AnalyticPrimsDescriptor::PointwiseUnary { op } => {
                validate_pointwise_shapes(shapes, 1, "AnalyticPointwiseUnary")?;
                if !supports_analytic_unary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "analytic unary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuAnalyticPlan::PointwiseUnary { op: *op })
            }
            AnalyticPrimsDescriptor::PointwiseBinary { op } => {
                validate_pointwise_shapes(shapes, 2, "AnalyticPointwiseBinary")?;
                if !supports_analytic_binary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "analytic binary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuAnalyticPlan::PointwiseBinary { op: *op })
            }
            AnalyticPrimsDescriptor::Reduction {
                modes_a,
                modes_c,
                op,
            } => {
                let ReductionPlanSpec { reduced_axes, .. } =
                    plan_reduction(modes_a, modes_c, shapes, "AnalyticReduction")?;
                if !supports_analytic_reduction::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "analytic reduction {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuAnalyticPlan::Reduction {
                    reduced_axes,
                    op: *op,
                })
            }
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        let views: Vec<_> = inputs
            .iter()
            .map(|tensor| tensor_to_view(tensor))
            .collect::<Result<_>>()?;
        let view_refs: Vec<_> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;

        match plan {
            CpuAnalyticPlan::PointwiseUnary { op } => {
                validate_execute_inputs(inputs, 1, "AnalyticPointwiseUnary")?;
                execute_analytic_unary(alpha, view_refs[0], beta, &mut out_view, *op)
            }
            CpuAnalyticPlan::PointwiseBinary { op } => {
                validate_execute_inputs(inputs, 2, "AnalyticPointwiseBinary")?;
                execute_analytic_binary(alpha, view_refs[0], view_refs[1], beta, &mut out_view, *op)
            }
            CpuAnalyticPlan::Reduction { reduced_axes, op } => {
                validate_execute_inputs(inputs, 1, "AnalyticReduction")?;
                execute_analytic_reduction(
                    alpha,
                    view_refs[0],
                    beta,
                    &mut out_view,
                    reduced_axes,
                    *op,
                )
            }
        }
    }

    fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool {
        match desc {
            AnalyticPrimsDescriptor::PointwiseUnary { op } => supports_analytic_unary::<S>(op),
            AnalyticPrimsDescriptor::PointwiseBinary { op } => supports_analytic_binary::<S>(op),
            AnalyticPrimsDescriptor::Reduction { op, .. } => supports_analytic_reduction::<S>(op),
        }
    }
}
