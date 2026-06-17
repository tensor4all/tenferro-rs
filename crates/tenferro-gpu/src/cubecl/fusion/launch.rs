use std::marker::PhantomData;

use cubecl::prelude::{CubeDim, CubeElement, CubeKernel, CubePrimitive, KernelId, KernelLauncher};

use super::classify::ClassifiedFusion;
use super::codegen::build_kernel_definition;
use crate::backend::ElementwiseFusionPlan;
use crate::cubecl::dispatch::{
    alloc_output, cube_count_for_len, cube_dim_1d, ensure_resident_on_runtime,
};
use crate::cubecl::runtime::CubeclRuntime;
use crate::types::TypedTensor;

pub(crate) fn launch<T>(
    runtime: &CubeclRuntime,
    classified: ClassifiedFusion<'_, T>,
) -> crate::Result<Vec<TypedTensor<T>>>
where
    T: CubeElement + CubePrimitive + Clone,
{
    let mut outputs = Vec::with_capacity(classified.plan.outputs().len());
    for _ in classified.plan.outputs() {
        outputs.push(alloc_output::<T>(runtime, &classified.output_shape)?);
    }
    let mut input_args = Vec::with_capacity(classified.inputs.len());
    for input in &classified.inputs {
        ensure_resident_on_runtime(runtime, input, "fused_elementwise")?;
        let arg = crate::cubecl::dispatch::typed_tensor_array_arg(input, "fused_elementwise")?;
        input_args.push(arg);
    }
    let mut output_args = Vec::with_capacity(outputs.len());
    for output in &outputs {
        let arg = crate::cubecl::dispatch::typed_tensor_array_arg(output, "fused_elementwise")?;
        output_args.push(arg);
    }
    if classified.n_elements == 0 {
        return Ok(outputs);
    }

    let settings = cubecl::prelude::KernelSettings::default().address_type(classified.address_type);
    let mut launcher = KernelLauncher::new(settings);
    let item = launcher.with_scope(|scope| T::as_type(scope));

    for arg in input_args {
        launcher.register_array(arg, item.clone());
    }
    for arg in output_args {
        launcher.register_array(arg, item.clone());
    }

    let kernel = FusedElementwiseKernel::<T> {
        plan: classified.plan.clone(),
        address_type: classified.address_type,
        cube_dim: cube_dim_1d(),
        _marker: PhantomData,
    };
    unsafe {
        // SAFETY: `ClassifiedFusion` proves all inputs share one dense output
        // shape, and `typed_tensor_array_arg` validates every raw array length
        // against its tensor shape. The generated kernel reads `out.len()`,
        // launches over `classified.n_elements`, and guards the body with
        // `ABSOLUTE_POS < out.len()`.
        launcher.launch_unchecked(
            cube_count_for_len(classified.n_elements)?,
            kernel,
            runtime.client(),
        );
    }
    Ok(outputs)
}

#[derive(Clone)]
struct FusedElementwiseKernel<T> {
    plan: ElementwiseFusionPlan,
    address_type: cubecl::prelude::AddressType,
    cube_dim: CubeDim,
    _marker: PhantomData<T>,
}

impl<T> cubecl::prelude::KernelMetadata for FusedElementwiseKernel<T>
where
    T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        "tenferro_fused_elementwise"
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.plan.clone())
            .cube_dim(self.cube_dim)
            .address_type(self.address_type)
    }

    fn address_type(&self) -> cubecl::prelude::StorageType {
        self.address_type.unsigned_type()
    }
}

impl<T> CubeKernel for FusedElementwiseKernel<T>
where
    T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
{
    fn define(&self) -> cubecl::prelude::KernelDefinition {
        build_kernel_definition::<T>(&self.plan, self.address_type, self.cube_dim)
    }
}
