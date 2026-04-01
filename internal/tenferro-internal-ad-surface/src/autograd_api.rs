use chainrules_core::Differentiable as _;

use crate::{Error, Result, Tensor};

#[derive(Debug, Clone, Default)]
pub struct BackwardOptions {
    pub retain_graph: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GradOptions {
    pub retain_graph: bool,
}

fn invalid_argument(message: impl Into<String>) -> Error {
    chainrules_core::AutodiffError::InvalidArgument(message.into()).into()
}

fn default_seed(output: &Tensor) -> crate::DynTensor {
    output.primal().seed_cotangent()
}

fn accumulate_optional_grad(slot: &mut Option<crate::DynTensor>, grad: Option<crate::DynTensor>) {
    match (slot.take(), grad) {
        (None, None) => *slot = None,
        (Some(existing), None) => *slot = Some(existing),
        (None, Some(new_grad)) => *slot = Some(new_grad),
        (Some(existing), Some(new_grad)) => {
            *slot = Some(
                <crate::DynTensor as chainrules_core::Differentiable>::accumulate_tangent(
                    existing, &new_grad,
                ),
            );
        }
    }
}

pub fn grad(
    outputs: &[&Tensor],
    inputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    options: GradOptions,
) -> Result<Vec<Option<Tensor>>> {
    if let Some(grad_outputs) = grad_outputs {
        if grad_outputs.len() != outputs.len() {
            return Err(invalid_argument(format!(
                "grad_outputs length mismatch: expected {}, found {}",
                outputs.len(),
                grad_outputs.len()
            )));
        }
    }

    let mut accum = vec![None; inputs.len()];
    let wrt = inputs.iter().map(|input| input.value()).collect::<Vec<_>>();

    for (index, output) in outputs.iter().enumerate() {
        let seed = grad_outputs
            .map(|grads| grads[index].primal().clone())
            .unwrap_or_else(|| default_seed(output));
        let grads = output.value().grad_wrt_with_seed(seed, &wrt)?;
        for (slot, grad) in accum.iter_mut().zip(grads) {
            accumulate_optional_grad(slot, grad);
        }
    }

    let _ = options.retain_graph;
    Ok(accum
        .into_iter()
        .map(|grad| grad.map(Tensor::from))
        .collect())
}

pub fn backward(
    outputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    options: BackwardOptions,
) -> Result<()> {
    if let Some(grad_outputs) = grad_outputs {
        if grad_outputs.len() != outputs.len() {
            return Err(invalid_argument(format!(
                "grad_outputs length mismatch: expected {}, found {}",
                outputs.len(),
                grad_outputs.len()
            )));
        }
    }

    for (index, output) in outputs.iter().enumerate() {
        let seed = grad_outputs
            .map(|grads| grads[index].primal().clone())
            .unwrap_or_else(|| default_seed(output));
        output.value().backward_with_seed(seed)?;
    }

    let _ = options.retain_graph;
    Ok(())
}
