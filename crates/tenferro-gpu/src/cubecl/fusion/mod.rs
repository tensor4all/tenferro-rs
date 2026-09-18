mod classify;
mod codegen;
mod launch;

use classify::classify;

use crate::backend::ElementwiseFusionPlan;
use crate::cubecl::CudaBackend;
use crate::Tensor;

pub(crate) fn execute_elementwise_fusion(
    backend: &CudaBackend,
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Option<Vec<Tensor>>> {
    match plan.dtype() {
        // An externally defined scalar has no fused kernel, so the fused path
        // reports that it cannot handle the plan rather than guessing.
        crate::DType::External(_) => Ok(None),
        crate::DType::F32 => {
            let Some(classified) = classify::<f32>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(
                outputs.into_iter().map(Tensor::from_typed::<f32>).collect(),
            ))
        }
        crate::DType::F64 => {
            let Some(classified) = classify::<f64>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(
                outputs.into_iter().map(Tensor::from_typed::<f64>).collect(),
            ))
        }
        crate::DType::C32 => {
            let Some(classified) = classify::<num_complex::Complex32>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<tenferro_tensor::Complex32>)
                    .collect(),
            ))
        }
        crate::DType::C64 => {
            let Some(classified) = classify::<num_complex::Complex64>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(
                outputs
                    .into_iter()
                    .map(Tensor::from_typed::<tenferro_tensor::Complex64>)
                    .collect(),
            ))
        }
        crate::DType::I32 | crate::DType::I64 | crate::DType::Bool => Ok(None),
    }
}
