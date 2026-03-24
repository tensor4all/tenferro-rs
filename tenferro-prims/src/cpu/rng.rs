use tenferro_algebra::Standard;
use tenferro_device::{Error, Generator, Result};
use tenferro_tensor::Tensor;

use crate::cpu::tensor_to_view_mut;
use crate::{
    validate_shape_count, validate_shape_eq, CpuBackend, CpuContext, RngPrimsDescriptor,
    TensorRngPrims,
};

/// CPU execution plan for the RNG family.
///
/// The plan stores the descriptor together with the output shape so execution
/// can revalidate the destination tensor before mutating it.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuRngPlan;
/// let _plan: CpuRngPlan = (tenferro_prims::RngPrimsDescriptor::Uniform, vec![2, 2]);
/// ```
pub type CpuRngPlan = (RngPrimsDescriptor, Vec<usize>);

fn fill_tensor_with_f64_samples<F>(output: &mut Tensor<f64>, mut sample: F) -> Result<()>
where
    F: FnMut() -> Result<f64>,
{
    let dims = output.dims().to_vec();
    let mut view = tensor_to_view_mut(output)?;
    let mut error = None;
    crate::for_each_index(&dims, |idx| {
        if error.is_some() {
            return;
        }
        match sample() {
            Ok(value) => view.set(idx, value),
            Err(err) => error = Some(err),
        }
    });

    match error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn fill_tensor_with_i32_samples<F>(output: &mut Tensor<i32>, mut sample: F) -> Result<()>
where
    F: FnMut() -> Result<i32>,
{
    let dims = output.dims().to_vec();
    let mut view = tensor_to_view_mut(output)?;
    let mut error = None;
    crate::for_each_index(&dims, |idx| {
        if error.is_some() {
            return;
        }
        match sample() {
            Ok(value) => view.set(idx, value),
            Err(err) => error = Some(err),
        }
    });

    match error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

fn validate_rng_plan(
    desc: &RngPrimsDescriptor,
    shapes: &[&[usize]],
    op_name: &str,
) -> Result<Vec<usize>> {
    validate_shape_count(shapes, 1, op_name)?;
    let output_shape = shapes[0].to_vec();
    match desc {
        RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => Ok(output_shape),
        RngPrimsDescriptor::Integer { low, high } => {
            if low >= high {
                return Err(Error::InvalidArgument(format!(
                    "{op_name} requires low < high (got low={low}, high={high})"
                )));
            }
            Ok(output_shape)
        }
    }
}

impl TensorRngPrims<Standard<f64>> for CpuBackend {
    type Plan = CpuRngPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        let output_shape = validate_rng_plan(desc, shapes, "CpuRng")?;
        match desc {
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Ok((desc.clone(), output_shape))
            }
            RngPrimsDescriptor::Integer { .. } => Err(Error::InvalidArgument(
                "integer RNG planning is only supported for Tensor<i32>".into(),
            )),
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        generator: &mut Generator,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        validate_shape_eq(output.dims(), &plan.1, "CpuRng output")?;
        match &plan.0 {
            RngPrimsDescriptor::Uniform => {
                fill_tensor_with_f64_samples(output, || Ok(generator.sample_uniform_f64()))
            }
            RngPrimsDescriptor::Normal => {
                fill_tensor_with_f64_samples(output, || Ok(generator.sample_standard_normal_f64()))
            }
            RngPrimsDescriptor::Integer { .. } => Err(Error::InvalidArgument(
                "integer RNG execution is only supported for Tensor<i32>".into(),
            )),
        }
    }

    fn has_rng_support(desc: RngPrimsDescriptor) -> bool {
        matches!(
            desc,
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal
        )
    }
}

impl TensorRngPrims<Standard<i32>> for CpuBackend {
    type Plan = CpuRngPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &RngPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        let output_shape = validate_rng_plan(desc, shapes, "CpuRng")?;
        match desc {
            RngPrimsDescriptor::Integer { low, high } => {
                if low >= high {
                    return Err(Error::InvalidArgument(format!(
                        "CpuRng requires low < high (got low={low}, high={high})"
                    )));
                }
                Ok((desc.clone(), output_shape))
            }
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Err(Error::InvalidArgument(
                    "floating-point RNG planning is only supported for Tensor<f64>".into(),
                ))
            }
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        generator: &mut Generator,
        output: &mut Tensor<i32>,
    ) -> Result<()> {
        validate_shape_eq(output.dims(), &plan.1, "CpuRng output")?;
        match &plan.0 {
            RngPrimsDescriptor::Integer { low, high } => {
                fill_tensor_with_i32_samples(output, || generator.sample_integer_i32(*low, *high))
            }
            RngPrimsDescriptor::Uniform | RngPrimsDescriptor::Normal => {
                Err(Error::InvalidArgument(
                    "floating-point RNG execution is only supported for Tensor<f64>".into(),
                ))
            }
        }
    }

    fn has_rng_support(desc: RngPrimsDescriptor) -> bool {
        matches!(desc, RngPrimsDescriptor::Integer { .. })
    }
}
