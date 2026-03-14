use std::collections::HashMap;

use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::{AdTensor, Error, NodeId, Result};

pub(crate) fn pullback<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &Tensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    let tape = output
        .reverse_tape()
        .cloned()
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
            output.structured_primal().clone(),
            output.structured_tangent().cloned(),
        )
        .map_err(Error::from)?;
    let cotangent = output
        .structured_primal()
        .with_payload_like(cotangent.clone())?;
    let grads = tape
        .pullback_with_seed(&tracked, cotangent)
        .map_err(Error::from)?;

    let mut out = HashMap::new();
    for (input_node, grad) in grads.entries() {
        out.insert(*input_node, grad.payload().clone());
    }
    Ok(out)
}
