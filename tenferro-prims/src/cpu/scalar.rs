use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cpu::common::{
    execute_binary_map, execute_unary_map, is_supported_ordered_real_type,
    is_supported_scalar_type, plan_reduction, validate_pointwise_shapes, CpuScalarValue,
};
use crate::cpu::family_reduction::{
    execute_extrema_reduction, execute_mean_reduction, execute_prod_reduction,
    execute_sum_reduction,
};
use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::infra::typed_dispatch::{
    cast_scalar_value, cast_strided_view, cast_strided_view_mut, dispatch_complex_scalar_type,
    dispatch_real_scalar_type, dispatch_standard_scalar_type,
};
use crate::{
    validate_execute_inputs, CpuBackend, CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor,
    ScalarReductionOp, ScalarUnaryOp, TensorScalarPrims,
};

/// CPU execution plan for the scalar protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuScalarPlan;
/// let _ = std::mem::size_of::<CpuScalarPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuScalarPlan {
    PointwiseUnary {
        op: ScalarUnaryOp,
    },
    PointwiseBinary {
        op: ScalarBinaryOp,
    },
    Reduction {
        reduced_axes: Vec<usize>,
        op: ScalarReductionOp,
    },
}

fn supports_scalar_unary<S: Scalar + 'static>(op: ScalarUnaryOp) -> bool {
    is_supported_scalar_type::<S>()
        && matches!(
            op,
            ScalarUnaryOp::Neg
                | ScalarUnaryOp::Conj
                | ScalarUnaryOp::Abs
                | ScalarUnaryOp::Reciprocal
                | ScalarUnaryOp::Real
                | ScalarUnaryOp::Imag
                | ScalarUnaryOp::Square
        )
}

fn supports_scalar_binary<S: Scalar + 'static>(op: ScalarBinaryOp) -> bool {
    match op {
        ScalarBinaryOp::Add | ScalarBinaryOp::Sub | ScalarBinaryOp::Mul | ScalarBinaryOp::Div => {
            is_supported_scalar_type::<S>()
        }
        ScalarBinaryOp::Maximum
        | ScalarBinaryOp::Minimum
        | ScalarBinaryOp::ClampMin
        | ScalarBinaryOp::ClampMax => is_supported_ordered_real_type::<S>(),
    }
}

fn supports_scalar_reduction<S: Scalar + 'static>(op: ScalarReductionOp) -> bool {
    match op {
        ScalarReductionOp::Sum | ScalarReductionOp::Prod | ScalarReductionOp::Mean => {
            is_supported_scalar_type::<S>()
        }
        ScalarReductionOp::Max | ScalarReductionOp::Min => is_supported_ordered_real_type::<S>(),
    }
}

fn execute_scalar_unary_typed<S: CpuScalarValue>(
    alpha: S,
    input: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: ScalarUnaryOp,
) -> Result<()> {
    match op {
        ScalarUnaryOp::Neg => execute_unary_map(alpha, input, beta, output, |x| -x),
        ScalarUnaryOp::Conj => execute_unary_map(alpha, input, beta, output, |x| x.conj()),
        ScalarUnaryOp::Abs => {
            execute_unary_map(alpha, input, beta, output, |x| S::from_real(x.abs()))
        }
        ScalarUnaryOp::Reciprocal => execute_unary_map(alpha, input, beta, output, |x| x.recip()),
        ScalarUnaryOp::Real => {
            execute_unary_map(alpha, input, beta, output, |x| S::from_real(x.re()))
        }
        ScalarUnaryOp::Imag => {
            execute_unary_map(alpha, input, beta, output, |x| S::from_real(x.im()))
        }
        ScalarUnaryOp::Square => execute_unary_map(alpha, input, beta, output, |x| x * x),
    }
}

fn execute_scalar_binary_real<S: num_traits::Float + CpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: ScalarBinaryOp,
) -> Result<()> {
    match op {
        ScalarBinaryOp::Add => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x + y),
        ScalarBinaryOp::Sub => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x - y),
        ScalarBinaryOp::Mul => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x * y),
        ScalarBinaryOp::Div => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x / y),
        ScalarBinaryOp::Maximum => {
            execute_binary_map(
                alpha,
                lhs,
                rhs,
                beta,
                output,
                |x, y| if x >= y { x } else { y },
            )
        }
        ScalarBinaryOp::Minimum => {
            execute_binary_map(
                alpha,
                lhs,
                rhs,
                beta,
                output,
                |x, y| if x <= y { x } else { y },
            )
        }
        ScalarBinaryOp::ClampMin => {
            execute_binary_map(
                alpha,
                lhs,
                rhs,
                beta,
                output,
                |x, y| if x >= y { x } else { y },
            )
        }
        ScalarBinaryOp::ClampMax => {
            execute_binary_map(
                alpha,
                lhs,
                rhs,
                beta,
                output,
                |x, y| if x <= y { x } else { y },
            )
        }
    }
}

fn execute_scalar_binary_complex<S: CpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: ScalarBinaryOp,
) -> Result<()> {
    match op {
        ScalarBinaryOp::Add => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x + y),
        ScalarBinaryOp::Sub => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x - y),
        ScalarBinaryOp::Mul => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x * y),
        ScalarBinaryOp::Div => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x / y),
        _ => Err(Error::InvalidArgument(format!(
            "scalar binary operation {op:?} requires ordered real scalars"
        ))),
    }
}

fn execute_scalar_unary<T: Scalar + 'static>(
    alpha: T,
    input: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: ScalarUnaryOp,
) -> Result<()> {
    dispatch_standard_scalar_type!(T, Concrete, {
        let input = cast_strided_view!(input, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_scalar_unary_typed::<Concrete>(alpha, input, beta, output, op);
    });

    Err(Error::InvalidArgument(format!(
        "scalar unary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

fn execute_scalar_binary<T: Scalar + 'static>(
    alpha: T,
    lhs: &strided_view::StridedView<T>,
    rhs: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: ScalarBinaryOp,
) -> Result<()> {
    dispatch_real_scalar_type!(T, Concrete, {
        let lhs = cast_strided_view!(lhs, T, Concrete);
        let rhs = cast_strided_view!(rhs, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_scalar_binary_real::<Concrete>(alpha, lhs, rhs, beta, output, op);
    });
    dispatch_complex_scalar_type!(T, Concrete, {
        let lhs = cast_strided_view!(lhs, T, Concrete);
        let rhs = cast_strided_view!(rhs, T, Concrete);
        let output = cast_strided_view_mut!(output, T, Concrete);
        let alpha = cast_scalar_value!(alpha, T, Concrete);
        let beta = cast_scalar_value!(beta, T, Concrete);
        return execute_scalar_binary_complex::<Concrete>(alpha, lhs, rhs, beta, output, op);
    });

    Err(Error::InvalidArgument(format!(
        "scalar binary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

fn execute_scalar_reduction<T: Scalar + 'static>(
    alpha: T,
    input: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    reduced_axes: &[usize],
    op: ScalarReductionOp,
) -> Result<()> {
    match op {
        ScalarReductionOp::Sum => {
            dispatch_standard_scalar_type!(T, Concrete, {
                let input = cast_strided_view!(input, T, Concrete);
                let output = cast_strided_view_mut!(output, T, Concrete);
                let alpha = cast_scalar_value!(alpha, T, Concrete);
                let beta = cast_scalar_value!(beta, T, Concrete);
                return execute_sum_reduction::<Concrete>(alpha, input, beta, output, reduced_axes);
            });
        }
        ScalarReductionOp::Prod => {
            dispatch_standard_scalar_type!(T, Concrete, {
                let input = cast_strided_view!(input, T, Concrete);
                let output = cast_strided_view_mut!(output, T, Concrete);
                let alpha = cast_scalar_value!(alpha, T, Concrete);
                let beta = cast_scalar_value!(beta, T, Concrete);
                return execute_prod_reduction::<Concrete>(
                    alpha,
                    input,
                    beta,
                    output,
                    reduced_axes,
                );
            });
        }
        ScalarReductionOp::Mean => {
            dispatch_standard_scalar_type!(T, Concrete, {
                let input = cast_strided_view!(input, T, Concrete);
                let output = cast_strided_view_mut!(output, T, Concrete);
                let alpha = cast_scalar_value!(alpha, T, Concrete);
                let beta = cast_scalar_value!(beta, T, Concrete);
                return execute_mean_reduction::<Concrete>(
                    alpha,
                    input,
                    beta,
                    output,
                    reduced_axes,
                );
            });
        }
        ScalarReductionOp::Max => {
            dispatch_real_scalar_type!(T, Concrete, {
                let input = cast_strided_view!(input, T, Concrete);
                let output = cast_strided_view_mut!(output, T, Concrete);
                let alpha = cast_scalar_value!(alpha, T, Concrete);
                let beta = cast_scalar_value!(beta, T, Concrete);
                return execute_extrema_reduction(alpha, input, beta, output, reduced_axes, true);
            });
        }
        ScalarReductionOp::Min => {
            dispatch_real_scalar_type!(T, Concrete, {
                let input = cast_strided_view!(input, T, Concrete);
                let output = cast_strided_view_mut!(output, T, Concrete);
                let alpha = cast_scalar_value!(alpha, T, Concrete);
                let beta = cast_scalar_value!(beta, T, Concrete);
                return execute_extrema_reduction(alpha, input, beta, output, reduced_axes, false);
            });
        }
    }

    Err(Error::InvalidArgument(format!(
        "scalar reduction {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

impl<S: Scalar + 'static> TensorScalarPrims<Standard<S>> for CpuBackend {
    type Plan = CpuScalarPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ScalarPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        match desc {
            ScalarPrimsDescriptor::PointwiseUnary { op } => {
                validate_pointwise_shapes(shapes, 1, "ScalarPointwiseUnary")?;
                if !supports_scalar_unary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "scalar unary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuScalarPlan::PointwiseUnary { op: *op })
            }
            ScalarPrimsDescriptor::PointwiseBinary { op } => {
                validate_pointwise_shapes(shapes, 2, "ScalarPointwiseBinary")?;
                if !supports_scalar_binary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "scalar binary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuScalarPlan::PointwiseBinary { op: *op })
            }
            ScalarPrimsDescriptor::Reduction {
                modes_a,
                modes_c,
                op,
            } => {
                if !supports_scalar_reduction::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "scalar reduction {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                let spec = plan_reduction(modes_a, modes_c, shapes, "ScalarReduction")?;
                let _ = spec.reduced_total;
                Ok(CpuScalarPlan::Reduction {
                    reduced_axes: spec.reduced_axes,
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
            CpuScalarPlan::PointwiseUnary { op } => {
                validate_execute_inputs(inputs, 1, "ScalarPointwiseUnary")?;
                execute_scalar_unary(alpha, view_refs[0], beta, &mut out_view, *op)
            }
            CpuScalarPlan::PointwiseBinary { op } => {
                validate_execute_inputs(inputs, 2, "ScalarPointwiseBinary")?;
                execute_scalar_binary(alpha, view_refs[0], view_refs[1], beta, &mut out_view, *op)
            }
            CpuScalarPlan::Reduction { reduced_axes, op } => {
                validate_execute_inputs(inputs, 1, "ScalarReduction")?;
                execute_scalar_reduction(
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

    fn has_scalar_support(desc: ScalarPrimsDescriptor) -> bool {
        match desc {
            ScalarPrimsDescriptor::PointwiseUnary { op } => supports_scalar_unary::<S>(op),
            ScalarPrimsDescriptor::PointwiseBinary { op } => supports_scalar_binary::<S>(op),
            ScalarPrimsDescriptor::Reduction { op, .. } => supports_scalar_reduction::<S>(op),
        }
    }
}
