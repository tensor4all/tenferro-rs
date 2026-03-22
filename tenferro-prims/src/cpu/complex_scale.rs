use num_complex::ComplexFloat;
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::{
    validate_shape_count, validate_shape_eq, ComplexScalePrimsDescriptor, CpuBackend, CpuContext,
    TensorComplexScalePrims,
};

/// CPU execution plan for the complex-by-real pointwise protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuComplexScalePlan;
/// let _ = std::mem::size_of::<CpuComplexScalePlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuComplexScalePlan {
    PointwiseMul,
}

fn plan_complex_scale(
    desc: &ComplexScalePrimsDescriptor,
    shapes: &[&[usize]],
) -> Result<CpuComplexScalePlan> {
    validate_shape_count(shapes, 3, "CpuComplexScalePointwiseMul")?;
    validate_shape_eq(shapes[0], shapes[1], "CpuComplexScalePointwiseMul lhs/rhs")?;
    validate_shape_eq(
        shapes[0],
        shapes[2],
        "CpuComplexScalePointwiseMul lhs/output",
    )?;
    match desc {
        ComplexScalePrimsDescriptor::PointwiseMul => Ok(CpuComplexScalePlan::PointwiseMul),
    }
}

fn execute_complex_scale_typed<Input>(
    alpha: Input,
    lhs: &strided_view::StridedView<Input>,
    rhs: &strided_view::StridedView<Input::Real>,
    beta: Input,
    output: &mut strided_view::StridedViewMut<Input>,
) -> Result<()>
where
    Input: ComplexFloat
        + Scalar
        + std::ops::Add<Output = Input>
        + std::ops::Mul<Input::Real, Output = Input>
        + std::ops::Mul<Output = Input>,
    Input::Real: Scalar + Send + Sync,
{
    let dims = output.dims().to_vec();
    crate::for_each_index(&dims, |idx| {
        output.set(
            idx,
            alpha * (lhs.get(idx) * rhs.get(idx)) + beta * output.get(idx),
        );
    });
    Ok(())
}

fn execute_complex_scale<Input>(
    plan: &CpuComplexScalePlan,
    alpha: Input,
    lhs: &Tensor<Input>,
    rhs: &Tensor<Input::Real>,
    beta: Input,
    output: &mut Tensor<Input>,
) -> Result<()>
where
    Input: ComplexFloat
        + Scalar
        + 'static
        + std::ops::Add<Output = Input>
        + std::ops::Mul<Input::Real, Output = Input>
        + std::ops::Mul<Output = Input>,
    Input::Real: Scalar + Send + Sync,
{
    validate_shape_eq(lhs.dims(), rhs.dims(), "CpuComplexScalePointwiseMul rhs")?;
    validate_shape_eq(
        lhs.dims(),
        output.dims(),
        "CpuComplexScalePointwiseMul output",
    )?;

    let lhs = tensor_to_view(lhs)?;
    let rhs = tensor_to_view(rhs)?;
    let mut output = tensor_to_view_mut(output)?;

    match plan {
        CpuComplexScalePlan::PointwiseMul => {
            execute_complex_scale_typed::<Input>(alpha, &lhs, &rhs, beta, &mut output)
        }
    }
}

impl<Input> TensorComplexScalePrims<Input> for CpuBackend
where
    Input: ComplexFloat
        + Scalar
        + 'static
        + std::ops::Add<Output = Input>
        + std::ops::Mul<Input::Real, Output = Input>
        + std::ops::Mul<Output = Input>,
    Input::Real: Scalar + Send + Sync,
{
    type Plan = CpuComplexScalePlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &ComplexScalePrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        plan_complex_scale(desc, shapes)
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: Input,
        lhs: &Tensor<Input>,
        rhs: &Tensor<Input::Real>,
        beta: Input,
        output: &mut Tensor<Input>,
    ) -> Result<()> {
        execute_complex_scale::<Input>(plan, alpha, lhs, rhs, beta, output)
    }

    fn has_complex_scale_support(desc: ComplexScalePrimsDescriptor) -> bool {
        matches!(desc, ComplexScalePrimsDescriptor::PointwiseMul)
    }
}
