use std::collections::HashMap;

use chainrules_core::NodeId;
use tenferro_algebra::Scalar;
use tenferro_internal_error::{Error, Result};
use tenferro_internal_frontend_core::{DynTensorTyped, StructuredTensor};
use tenferro_tensor::Tensor;

use crate::dyn_ad_tensor::DynAdTensorTyped;
use crate::{AdMode, AdTensor};

pub fn pullback<S>(
    output: &AdTensor<S>,
    cotangent: &Tensor<S>,
) -> Result<HashMap<NodeId, Tensor<S>>>
where
    S: Scalar + DynTensorTyped + DynAdTensorTyped + Clone,
{
    if output.mode() != AdMode::Reverse {
        return Err(Error::InvalidAdTensor {
            message: "pullback requires a reverse-mode output tensor".to_string(),
        });
    }

    let (_node, tape) = output
        .reverse_handle()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output tensor must carry a tape handle".to_string(),
        })?;

    let tracked = output.as_tracked().ok_or_else(|| Error::InvalidAdTensor {
        message: "reverse-mode output tensor must expose a tracked value".to_string(),
    })?;

    let structured_cotangent = StructuredTensor::from(cotangent.clone());
    let dyn_cotangent = S::into_dyn(structured_cotangent);

    let grads = tape
        .pullback_with_seed(&tracked, dyn_cotangent)
        .map_err(Error::Autodiff)?;

    let mut result = HashMap::new();
    for (grad_node, dyn_grad) in grads.entries() {
        if let Some(tensor) = S::structured_ref(dyn_grad) {
            result.insert(*grad_node, tensor.payload().clone());
        }
    }
    Ok(result)
}
