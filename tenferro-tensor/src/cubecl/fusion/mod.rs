mod classify;
mod codegen;
mod launch;

use classify::classify;

use crate::backend::ElementwiseFusionPlan;
use crate::cubecl::CubeclBackend;
use crate::Tensor;

pub(crate) fn execute_elementwise_fusion(
    backend: &CubeclBackend,
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Option<Vec<Tensor>>> {
    match plan.dtype {
        crate::DType::F32 => {
            let Some(classified) = classify::<f32>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(outputs.into_iter().map(Tensor::F32).collect()))
        }
        crate::DType::F64 => {
            let Some(classified) = classify::<f64>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(outputs.into_iter().map(Tensor::F64).collect()))
        }
        crate::DType::C32 => {
            let Some(classified) = classify::<num_complex::Complex32>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(outputs.into_iter().map(Tensor::C32).collect()))
        }
        crate::DType::C64 => {
            let Some(classified) = classify::<num_complex::Complex64>(inputs, plan)? else {
                return Ok(None);
            };
            let outputs = launch::launch(backend.runtime(), classified)?;
            Ok(Some(outputs.into_iter().map(Tensor::C64).collect()))
        }
    }
}
