use num_traits::NumCast;
use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::shape_helpers::broadcast_tensor_to_shape;
use crate::{
    cast_metadata_value, for_each_index_result, supports_metadata_cast,
    validate_metadata_cast_shapes, validate_pointwise_cast_bridge_inputs,
    validate_where_bridge_inputs, CpuBackend, CpuContext, MetadataCastPrimsDescriptor,
    MetadataTensorRef, TensorMetadataCastPrims,
};

fn execute_pointwise_cast<S>(
    input: MetadataTensorRef<'_>,
    alpha: S,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: Scalar + NumCast + 'static,
{
    match input {
        MetadataTensorRef::I32(tensor) => {
            let tensor = broadcast_tensor_to_shape(tensor, output.dims(), "MetadataCast input")?;
            let input = tensor_to_view(&tensor)?;
            let mut output_view = tensor_to_view_mut(output)?;
            let dims = output_view.dims().to_vec();
            for_each_index_result(&dims, |idx| {
                let casted = cast_metadata_value::<S, i32>(input.get(idx), "metadata i32 value")?;
                output_view.set(idx, alpha * casted + beta * output_view.get(idx));
                Ok(())
            })
        }
        MetadataTensorRef::Bool(tensor) => {
            let tensor = broadcast_tensor_to_shape(tensor, output.dims(), "MetadataCast input")?;
            let input = tensor_to_view(&tensor)?;
            let mut output_view = tensor_to_view_mut(output)?;
            let dims = output_view.dims().to_vec();
            for_each_index_result(&dims, |idx| {
                let casted = cast_metadata_value::<S, u8>(
                    if input.get(idx) != 0 { 1 } else { 0 },
                    "metadata bool value",
                )?;
                output_view.set(idx, alpha * casted + beta * output_view.get(idx));
                Ok(())
            })
        }
    }
}

fn execute_where<S>(
    cond: MetadataTensorRef<'_>,
    on_true: &Tensor<S>,
    on_false: &Tensor<S>,
    alpha: S,
    beta: S,
    output: &mut Tensor<S>,
) -> Result<()>
where
    S: Scalar + NumCast + 'static,
{
    let on_true = broadcast_tensor_to_shape(on_true, output.dims(), "MetadataCastWhere true")?;
    let on_false = broadcast_tensor_to_shape(on_false, output.dims(), "MetadataCastWhere false")?;
    let on_true = tensor_to_view(&on_true)?;
    let on_false = tensor_to_view(&on_false)?;
    match cond {
        MetadataTensorRef::Bool(cond) => {
            let cond = broadcast_tensor_to_shape(cond, output.dims(), "MetadataCastWhere cond")?;
            let cond = tensor_to_view(&cond)?;
            let mut output_view = tensor_to_view_mut(output)?;
            let dims = output_view.dims().to_vec();
            for_each_index_result(&dims, |idx| {
                let selected = if cond.get(idx) != 0 {
                    on_true.get(idx)
                } else {
                    on_false.get(idx)
                };
                output_view.set(idx, alpha * selected + beta * output_view.get(idx));
                Ok(())
            })
        }
        MetadataTensorRef::I32(_) => Err(Error::InvalidArgument(
            "MetadataCastWhere expects bool condition metadata".into(),
        )),
    }
}

impl<S> TensorMetadataCastPrims<S> for CpuBackend
where
    S: Scalar + NumCast + 'static,
{
    type Plan = MetadataCastPrimsDescriptor;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &MetadataCastPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        validate_metadata_cast_shapes(desc, shapes, "MetadataCast")?;
        if !supports_metadata_cast(desc) {
            return Err(Error::InvalidArgument(format!(
                "metadata cast descriptor {desc:?} is not supported on CpuBackend for {}",
                std::any::type_name::<S>()
            )));
        }
        Ok(desc.clone())
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[crate::MetadataScalarTensorRef<'_, S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        match plan {
            MetadataCastPrimsDescriptor::PointwiseCast { .. } => {
                let input = validate_pointwise_cast_bridge_inputs(inputs)?;
                execute_pointwise_cast(input, alpha, beta, output)
            }
            MetadataCastPrimsDescriptor::Where { .. } => {
                let (cond, on_true, on_false) = validate_where_bridge_inputs(inputs)?;
                execute_where(cond, on_true, on_false, alpha, beta, output)
            }
        }
    }

    fn has_metadata_cast_support(desc: MetadataCastPrimsDescriptor) -> bool {
        supports_metadata_cast(&desc)
    }
}
