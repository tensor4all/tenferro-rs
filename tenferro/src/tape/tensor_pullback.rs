use std::collections::HashMap;

use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::core::{DynTensorTyped, NodeId};
use crate::{AdTensor, Error, Result};

pub(crate) fn pullback<T: Scalar + DynTensorTyped + 'static>(
    output: &AdTensor<T>,
    cotangent: &Tensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    let tape = output
        .reverse_tape()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "pullback requires reverse-mode output tensor".to_string(),
        })?;
    let node = output
        .reverse_node_id()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "pullback requires reverse-mode output tensor".to_string(),
        })?;
    let tracked = tape
        .tracked_existing(
            node,
            T::into_dyn(output.structured_primal().clone()),
            output.structured_tangent().cloned().map(T::into_dyn),
        )
        .map_err(Error::from)?;
    let cotangent = output
        .structured_primal()
        .with_payload_like(cotangent.clone())?;
    let grads = tape
        .pullback_with_seed(&tracked, T::into_dyn(cotangent))
        .map_err(Error::from)?;

    let mut out = HashMap::new();
    for (input_node, grad) in grads.entries() {
        let grad = grad
            .typed_ref::<T>()
            .ok_or_else(|| Error::InvalidAdTensor {
                message: "pullback gradient dtype did not match requested tensor type".to_string(),
            })?;
        out.insert(*input_node, grad.payload().clone());
    }
    Ok(out)
}
