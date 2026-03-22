use num_complex::ComplexFloat;
use num_traits::{Float, One, Zero};
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::cpu::common::{plan_reduction, CpuScalarValue};
use crate::cpu::family_reduction::{
    execute_extrema_reduction, execute_mean_reduction, execute_prod_reduction,
    execute_sum_reduction,
};
use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::{
    validate_execute_inputs, validate_shape_count, validate_shape_eq, ComplexRealPrimsDescriptor,
    ComplexRealUnaryOp, CpuBackend, CpuContext, ScalarReductionOp, TensorComplexRealPrims,
};

/// CPU execution plan for the complex-to-real unary protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuComplexRealPlan;
/// let _ = std::mem::size_of::<CpuComplexRealPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuComplexRealPlan {
    PointwiseUnary {
        op: ComplexRealUnaryOp,
    },
    Reduction {
        unary_op: ComplexRealUnaryOp,
        reduction_op: ScalarReductionOp,
        reduced_axes: Vec<usize>,
    },
}

fn supports_complex_real_unary(op: ComplexRealUnaryOp) -> bool {
    matches!(
        op,
        ComplexRealUnaryOp::Abs | ComplexRealUnaryOp::Real | ComplexRealUnaryOp::Imag
    )
}

fn execute_complex_real_unary_typed<Input>(
    alpha: Input::Real,
    input: &strided_view::StridedView<Input>,
    beta: Input::Real,
    output: &mut strided_view::StridedViewMut<Input::Real>,
    op: ComplexRealUnaryOp,
) -> Result<()>
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Float,
{
    match op {
        ComplexRealUnaryOp::Abs => {
            let dims = output.dims().to_vec();
            crate::for_each_index(&dims, |idx| {
                let mapped = input.get(idx).abs();
                let value = alpha * mapped;
                if beta == Input::Real::zero() {
                    output.set(idx, value);
                } else {
                    output.set(idx, value + beta * output.get(idx));
                }
            });
            Ok(())
        }
        ComplexRealUnaryOp::Real => {
            let dims = output.dims().to_vec();
            crate::for_each_index(&dims, |idx| {
                let mapped = input.get(idx).re();
                let value = alpha * mapped;
                if beta == Input::Real::zero() {
                    output.set(idx, value);
                } else {
                    output.set(idx, value + beta * output.get(idx));
                }
            });
            Ok(())
        }
        ComplexRealUnaryOp::Imag => {
            let dims = output.dims().to_vec();
            crate::for_each_index(&dims, |idx| {
                let mapped = input.get(idx).im();
                let value = alpha * mapped;
                if beta == Input::Real::zero() {
                    output.set(idx, value);
                } else {
                    output.set(idx, value + beta * output.get(idx));
                }
            });
            Ok(())
        }
    }
}

fn plan_complex_real_unary<Input>(
    desc: &ComplexRealPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CpuComplexRealPlan>
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Float,
{
    validate_shape_count(shapes, 2, "CpuComplexRealPointwiseUnary")?;
    validate_shape_eq(shapes[0], shapes[1], "CpuComplexRealPointwiseUnary")?;
    match desc {
        ComplexRealPrimsDescriptor::PointwiseUnary { op } => {
            if !supports_complex_real_unary(*op) {
                return Err(Error::InvalidArgument(format!(
                    "complex-real unary operation {op:?} is not supported on CpuBackend for {}",
                    std::any::type_name::<Input>()
                )));
            }
            Ok(CpuComplexRealPlan::PointwiseUnary { op: *op })
        }
        ComplexRealPrimsDescriptor::Reduction { .. } => Err(Error::InvalidArgument(
            "expected complex-real unary descriptor".into(),
        )),
    }
}

fn plan_complex_real_reduction<Input>(
    desc: &ComplexRealPrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CpuComplexRealPlan>
where
    Input: ComplexFloat + Scalar,
    Input::Real: Scalar + Float,
{
    match desc {
        ComplexRealPrimsDescriptor::Reduction {
            modes_a,
            modes_c,
            unary_op,
            reduction_op,
        } => {
            if !supports_complex_real_unary(*unary_op) {
                return Err(Error::InvalidArgument(format!(
                    "complex-real unary operation {unary_op:?} is not supported on CpuBackend for {}",
                    std::any::type_name::<Input>()
                )));
            }
            let spec = plan_reduction(modes_a, modes_c, shapes, "CpuComplexRealReduction")?;
            Ok(CpuComplexRealPlan::Reduction {
                unary_op: *unary_op,
                reduction_op: *reduction_op,
                reduced_axes: spec.reduced_axes,
            })
        }
        _ => Err(Error::InvalidArgument(
            "expected complex-real reduction descriptor".into(),
        )),
    }
}

fn execute_complex_real_unary<Input>(
    plan: &CpuComplexRealPlan,
    alpha: Input::Real,
    inputs: &[&Tensor<Input>],
    beta: Input::Real,
    output: &mut Tensor<Input::Real>,
) -> Result<()>
where
    Input: ComplexFloat + Scalar + 'static,
    Input::Real: CpuScalarValue + Float,
{
    validate_execute_inputs(inputs, 1, "CpuComplexRealPointwiseUnary")?;
    let input = tensor_to_view(inputs[0])?;
    let mut output = tensor_to_view_mut(output)?;

    match plan {
        CpuComplexRealPlan::PointwiseUnary { op } => {
            execute_complex_real_unary_typed::<Input>(alpha, &input, beta, &mut output, *op)
        }
        CpuComplexRealPlan::Reduction {
            unary_op,
            reduction_op,
            reduced_axes,
        } => {
            let input_space = inputs[0].logical_memory_space();
            let mut temp = Tensor::<Input::Real>::zeros(
                inputs[0].dims(),
                input_space,
                MemoryOrder::ColumnMajor,
            );
            {
                let mut temp_view = tensor_to_view_mut(&mut temp)?;
                execute_complex_real_unary_typed::<Input>(
                    Input::Real::one(),
                    &input,
                    Input::Real::zero(),
                    &mut temp_view,
                    *unary_op,
                )?;
            }

            let temp_view = tensor_to_view(&temp)?;
            match reduction_op {
                ScalarReductionOp::Sum => {
                    execute_sum_reduction(alpha, &temp_view, beta, &mut output, reduced_axes)
                }
                ScalarReductionOp::Prod => {
                    execute_prod_reduction(alpha, &temp_view, beta, &mut output, reduced_axes)
                }
                ScalarReductionOp::Mean => {
                    execute_mean_reduction(alpha, &temp_view, beta, &mut output, reduced_axes)
                }
                ScalarReductionOp::Max => execute_extrema_reduction(
                    alpha,
                    &temp_view,
                    beta,
                    &mut output,
                    reduced_axes,
                    true,
                ),
                ScalarReductionOp::Min => execute_extrema_reduction(
                    alpha,
                    &temp_view,
                    beta,
                    &mut output,
                    reduced_axes,
                    false,
                ),
            }
        }
    }
}

impl<Input> TensorComplexRealPrims<Input> for CpuBackend
where
    Input: ComplexFloat + Scalar + 'static,
    Input::Real: CpuScalarValue + Float,
{
    type Real = Input::Real;
    type Plan = CpuComplexRealPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexRealPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        match desc {
            ComplexRealPrimsDescriptor::PointwiseUnary { .. } => {
                plan_complex_real_unary::<Input>(desc, shapes)
            }
            ComplexRealPrimsDescriptor::Reduction { .. } => {
                plan_complex_real_reduction::<Input>(desc, shapes)
            }
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Input::Real,
        inputs: &[&Tensor<Input>],
        beta: Input::Real,
        output: &mut Tensor<Self::Real>,
    ) -> Result<()> {
        execute_complex_real_unary::<Input>(plan, alpha, inputs, beta, output)
    }

    fn has_complex_real_support(desc: ComplexRealPrimsDescriptor) -> bool {
        matches!(
            desc,
            ComplexRealPrimsDescriptor::PointwiseUnary {
                op: ComplexRealUnaryOp::Abs | ComplexRealUnaryOp::Real | ComplexRealUnaryOp::Imag
            } | ComplexRealPrimsDescriptor::Reduction {
                unary_op: ComplexRealUnaryOp::Abs
                    | ComplexRealUnaryOp::Real
                    | ComplexRealUnaryOp::Imag,
                reduction_op: ScalarReductionOp::Sum
                    | ScalarReductionOp::Prod
                    | ScalarReductionOp::Mean
                    | ScalarReductionOp::Max
                    | ScalarReductionOp::Min,
                ..
            }
        )
    }
}
