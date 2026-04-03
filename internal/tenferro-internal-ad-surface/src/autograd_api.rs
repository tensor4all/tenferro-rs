use std::collections::HashMap;

use chainrules_core::Differentiable as _;

use crate::core::{DynTensor, NodeId};
use crate::{Error, Result, Tensor};

/// Options for reverse-mode gradient accumulation.
///
/// # Examples
///
/// ```ignore
/// use tenferro::BackwardOptions;
///
/// let opts = BackwardOptions::default();
/// assert!(!opts.retain_graph);
/// assert!(!opts.create_graph);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BackwardOptions {
    pub retain_graph: bool,
}

#[derive(Debug, Clone, Default)]
pub struct GradOptions {
    pub retain_graph: bool,
}

fn reverse_tape(output: &Tensor) -> Option<tidu::expert::Tape<crate::DynTensor>> {
    output.reverse_tape()
}

fn mixed_reverse_graph_error(
    expected: Option<tidu::expert::Tape<crate::DynTensor>>,
    found: Option<tidu::expert::Tape<crate::DynTensor>>,
) -> Error {
    Error::MixedReverseTape {
        expected: expected.map_or(0, |tape| tape.id() as u64),
        found: found.map_or(0, |tape| tape.id() as u64),
    }
}

fn invalid_argument(message: impl Into<String>) -> Error {
    chainrules_core::AutodiffError::InvalidArgument(message.into()).into()
}

fn default_seed(output: &Tensor) -> Result<Tensor> {
    Ok(Tensor::from(output.primal_snapshot().seed_cotangent()))
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
    let mut shared_tape: Option<tidu::expert::Tape<crate::DynTensor>> = None;
    let mut reference_output: Option<&Tensor> = None;

    for (index, output) in outputs.iter().enumerate() {
        let seed = match grad_outputs {
            Some(grad_outputs) => &grad_outputs[index],
            None => &default_seed(output)?,
        };

        if let Some(reference) = reference_output {
            if !reference.shares_reverse_graph(output) {
                return Err(mixed_reverse_graph_error(
                    reverse_tape(reference),
                    reverse_tape(output),
                ));
            }
        } else {
            reference_output = Some(output);
        }

        if let Some(tape) = reverse_tape(output) {
            if let Some(expected) = &shared_tape {
                if !expected.same_tape(&tape) {
                    return Err(mixed_reverse_graph_error(
                        Some(expected.clone()),
                        Some(tape),
                    ));
                }
            } else {
                shared_tape = Some(tape);
            }
        }

        let grads = output.pullback_wrt(seed, inputs)?;
        for (slot, grad) in accum.iter_mut().zip(grads.into_iter()) {
            accumulate_optional_grad(slot, grad);
        }
    }

    let _ = options.retain_graph;
    Ok(accum
        .into_iter()
        .map(|grad| grad.map(Tensor::from))
        .collect())
}

/// Accumulates gradients into the requested leaf `inputs`.
///
/// This is the eager reverse-mode entrypoint corresponding to
/// `torch.autograd.backward(...)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
/// x.set_requires_grad(true).unwrap();
/// let out = x.exp().unwrap().sum().unwrap();
/// backward(&[&out], None, BackwardOptions::default()).unwrap();
/// assert!(x.grad().unwrap().is_some());
/// ```
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
        let seed = match grad_outputs {
            Some(grads) => grads[index].primal_snapshot(),
            None => default_seed(output)?.primal_snapshot(),
        };
        output.value().backward_with_seed(seed)?;
    }

    let _ = options.retain_graph;
    Ok(())
}

/// Options for HVP computation.
///
/// # Examples
///
/// ```ignore
/// use tenferro::HvpOptions;
///
/// let opts = HvpOptions::default();
/// assert!(!opts.retain_graph);
/// ```
#[derive(Debug, Clone, Default)]
pub struct HvpOptions {
    /// If true, do not free the computation graph after HVP.
    pub retain_graph: bool,
}

/// Result of a Hessian-vector product computation.
///
/// # Examples
///
/// ```ignore
/// let result = hvp(&output, &[&x], &[&v], HvpOptions::default())?;
/// let grad = result.gradients[0].as_ref();
/// let hvp_val = result.hvps[0].as_ref();
/// ```
#[derive(Debug)]
pub struct HvpResult {
    /// Gradients of the output with respect to each input.
    pub gradients: Vec<Option<Tensor>>,
    /// Hessian-vector products for each input.
    pub hvps: Vec<Option<Tensor>>,
}

fn dyn_primal_from_snapshot(snapshot: DynTensor) -> Tensor {
    Tensor::from(snapshot)
}

fn reverse_handle(value: &Tensor) -> Option<(NodeId, tidu::expert::Tape<DynTensor>)> {
    value.reverse_handle()
}

fn as_tracked_dyn(output: &Tensor) -> Option<tidu::expert::TrackedValue<DynTensor>> {
    output.as_tracked()
}

fn input_to_dyn_tensor(input: &Tensor) -> DynTensor {
    input.primal_snapshot()
}

/// Computes gradient and Hessian-vector product for a scalar output.
///
/// `output` must be a scalar (rank-0) `tenferro::Tensor` on a reverse-mode tape.
/// `inputs` are the `tenferro::Tensor`s to differentiate with respect to.
/// `v` provides the tangent direction for each input (same shapes as inputs).
///
/// Returns both gradients and HVPs for each input.
///
/// # Examples
///
/// ```ignore
/// // f(x) = einsum("i,i->", &[&x, &x])  (= sum(x^2))
/// // H = 2I, Hv = 2v
/// let result = hvp(&output, &[&x], &[&v], HvpOptions::default())?;
/// ```
pub fn hvp(
    output: &Tensor,
    inputs: &[&Tensor],
    v: &[&Tensor],
    options: HvpOptions,
) -> Result<HvpResult> {
    let num_elements: usize = output.dims().iter().product::<usize>().max(1);
    if !output.dims().is_empty() || num_elements != 1 {
        return Err(Error::Autodiff(
            chainrules_core::AutodiffError::NonScalarLoss { num_elements },
        ));
    }

    if v.len() != inputs.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "hvp: v length {} does not match inputs length {}",
                v.len(),
                inputs.len()
            ),
        });
    }

    for (i, (input, vi)) in inputs.iter().zip(v.iter()).enumerate() {
        if input.dims() != vi.dims() {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "hvp: shape mismatch at index {}: input dims {:?} vs v dims {:?}",
                    i,
                    input.dims(),
                    vi.dims()
                ),
            });
        }
    }

    let tape = match reverse_tape(output) {
        Some(tape) => tape,
        None => {
            return Err(Error::Autodiff(
                chainrules_core::AutodiffError::HvpNotSupported,
            ))
        }
    };

    // Build leaf_tangents HashMap: NodeId -> DynTensor
    let mut leaf_tangents: HashMap<NodeId, DynTensor> = HashMap::new();
    for (input, vi) in inputs.iter().zip(v.iter()) {
        if let Some((node, input_tape)) = reverse_handle(input) {
            if !input_tape.same_tape(&tape) {
                return Err(Error::MixedReverseTape {
                    expected: tape.id() as u64,
                    found: input_tape.id() as u64,
                });
            }
            leaf_tangents.insert(node, input_to_dyn_tensor(vi));
        }
    }

    let tracked = as_tracked_dyn(output).ok_or(Error::InvalidAdTensor {
        message: "hvp requires reverse-mode output tensor".to_string(),
    })?;

    let tidu_result = tape.hvp(&tracked, &leaf_tangents).map_err(Error::from)?;

    // Project results to requested inputs
    let mut gradients = Vec::with_capacity(inputs.len());
    let mut hvps = Vec::with_capacity(inputs.len());

    for input in inputs {
        match reverse_handle(input) {
            Some((node, _)) => {
                gradients.push(
                    tidu_result
                        .gradients
                        .get(node)
                        .cloned()
                        .map(dyn_primal_from_snapshot),
                );
                hvps.push(
                    tidu_result
                        .hvp
                        .get(node)
                        .cloned()
                        .map(dyn_primal_from_snapshot),
                );
            }
            None => {
                gradients.push(None);
                hvps.push(None);
            }
        }
    }

    if !options.retain_graph {
        tape.free_graph();
    }

    Ok(HvpResult { gradients, hvps })
}
