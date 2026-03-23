use num_traits::NumCast;
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::{
    blend_cast_into_host_output, cast_metadata_tensor_to_host_scalar_tensor, scalar_where_desc,
    supports_metadata_cast, validate_metadata_cast_shapes, validate_pointwise_cast_bridge_inputs,
    validate_where_bridge_inputs, CpuBackend, CpuContext, MetadataCastPrimsDescriptor,
    TensorMetadataCastPrims, TensorScalarPrims,
};

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
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[crate::MetadataScalarTensorRef<'_, S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        match plan {
            MetadataCastPrimsDescriptor::PointwiseCast { .. } => {
                let input = validate_pointwise_cast_bridge_inputs(inputs)?;
                let casted = cast_metadata_tensor_to_host_scalar_tensor::<S>(
                    input,
                    output.dims(),
                    output.strides(),
                    output.offset(),
                )?;
                blend_cast_into_host_output(output, &casted, alpha, beta)
            }
            MetadataCastPrimsDescriptor::Where { .. } => {
                let (cond, on_true, on_false) = validate_where_bridge_inputs(inputs)?;
                let cond_scalar = cast_metadata_tensor_to_host_scalar_tensor::<S>(
                    cond,
                    output.dims(),
                    output.strides(),
                    output.offset(),
                )?;
                let scalar_desc = scalar_where_desc();
                let scalar_plan = <CpuBackend as TensorScalarPrims<Standard<S>>>::plan(
                    ctx,
                    &scalar_desc,
                    &[
                        cond_scalar.dims(),
                        on_true.dims(),
                        on_false.dims(),
                        output.dims(),
                    ],
                )?;
                <CpuBackend as TensorScalarPrims<Standard<S>>>::execute(
                    ctx,
                    &scalar_plan,
                    alpha,
                    &[&cond_scalar, on_true, on_false],
                    beta,
                    output,
                )
            }
        }
    }

    fn has_metadata_cast_support(desc: MetadataCastPrimsDescriptor) -> bool {
        supports_metadata_cast(&desc)
    }
}
