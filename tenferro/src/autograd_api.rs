use chainrules_core::Differentiable;

use crate::{Error, Result, Tensor};

/// Options for reverse-mode gradient accumulation.
///
/// # Examples
///
/// ```rust
/// use tenferro::BackwardOptions;
///
/// let opts = BackwardOptions::default();
/// assert!(!opts.retain_graph);
/// assert!(!opts.create_graph);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BackwardOptions {
    pub retain_graph: bool,
    pub create_graph: bool,
}

/// Options for functional reverse-mode gradient queries.
///
/// # Examples
///
/// ```rust
/// use tenferro::GradOptions;
///
/// let opts = GradOptions::default();
/// assert!(!opts.retain_graph);
/// assert!(!opts.create_graph);
/// ```
#[derive(Debug, Clone, Default)]
pub struct GradOptions {
    pub retain_graph: bool,
    pub create_graph: bool,
}

fn reverse_tape(output: &Tensor) -> Option<tidu::Tape<crate::DynTensor>> {
    match output {
        Tensor::F32(value) => value.reverse_tape(),
        Tensor::F64(value) => value.reverse_tape(),
        Tensor::C32(value) => value.reverse_tape(),
        Tensor::C64(value) => value.reverse_tape(),
    }
}

fn default_seed(output: &Tensor) -> Result<Tensor> {
    match output {
        Tensor::F32(value) => Ok(Tensor::F32(crate::AdTensor::new_primal(
            value.structured_primal().seed_cotangent(),
        ))),
        Tensor::F64(value) => Ok(Tensor::F64(crate::AdTensor::new_primal(
            value.structured_primal().seed_cotangent(),
        ))),
        Tensor::C32(value) => Ok(Tensor::C32(crate::AdTensor::new_primal(
            value.structured_primal().seed_cotangent(),
        ))),
        Tensor::C64(value) => Ok(Tensor::C64(crate::AdTensor::new_primal(
            value.structured_primal().seed_cotangent(),
        ))),
    }
}

fn accumulate_optional_grad(slot: &mut Option<Tensor>, grad: Option<Tensor>) -> Result<()> {
    match (slot.take(), grad) {
        (None, None) => {
            *slot = None;
            Ok(())
        }
        (Some(existing), None) => {
            *slot = Some(existing);
            Ok(())
        }
        (None, Some(new_grad)) => {
            *slot = Some(new_grad);
            Ok(())
        }
        (Some(existing), Some(new_grad)) => {
            *slot = Some(existing.add(&new_grad)?);
            Ok(())
        }
    }
}

/// Computes gradients of `outputs` with respect to the requested `inputs`.
///
/// This mirrors `torch.autograd.grad(...)` more closely than leaf-local
/// `Tensor::grad()`.
///
/// # Examples
///
/// ```rust
/// use tenferro::{grad, set_default_runtime, GradOptions, RuntimeContext, Tensor};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
/// let mut y = Tensor::from_slice(&[3.0_f64, 4.0], &[2]).unwrap();
/// x.set_requires_grad(true).unwrap();
/// y.set_requires_grad(true).unwrap();
///
/// let out = x.add(&y).unwrap().sum().unwrap();
/// let grads = grad(&[&out], &[&x, &y], None, GradOptions::default()).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn grad(
    outputs: &[&Tensor],
    inputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    options: GradOptions,
) -> Result<Vec<Option<Tensor>>> {
    if options.create_graph {
        return Err(Error::UnsupportedAdOp {
            op: "grad(create_graph)",
        });
    }
    if let Some(grad_outputs) = grad_outputs {
        if grad_outputs.len() != outputs.len() {
            return Err(Error::InvalidAdTensor {
                message: format!(
                    "grad_outputs length mismatch: expected {}, found {}",
                    outputs.len(),
                    grad_outputs.len()
                ),
            });
        }
    }

    let mut accum = vec![None; inputs.len()];
    let mut shared_tape: Option<tidu::Tape<crate::DynTensor>> = None;

    for (index, output) in outputs.iter().enumerate() {
        let seed = match grad_outputs {
            Some(grad_outputs) => &grad_outputs[index],
            None => &default_seed(output)?,
        };

        if let Some(tape) = reverse_tape(output) {
            if let Some(expected) = &shared_tape {
                if !expected.same_tape(&tape) {
                    return Err(Error::MixedReverseTape {
                        expected: expected.id() as u64,
                        found: tape.id() as u64,
                    });
                }
            } else {
                shared_tape = Some(tape);
            }
        }

        let grads = output.pullback_wrt(seed, inputs)?;
        for (slot, grad) in accum.iter_mut().zip(grads.into_iter()) {
            accumulate_optional_grad(slot, grad)?;
        }
    }

    if !options.retain_graph {
        if let Some(tape) = shared_tape {
            tape.free_graph();
        }
    }

    Ok(accum)
}

/// Accumulates gradients into the requested leaf `inputs`.
///
/// This is the eager reverse-mode entrypoint corresponding to
/// `torch.autograd.backward(...)`.
///
/// # Examples
///
/// ```rust
/// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
/// use tenferro_prims::CpuContext;
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
/// x.set_requires_grad(true).unwrap();
/// let out = x.exp().unwrap().sum().unwrap();
/// backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
/// assert!(x.grad().is_some());
/// ```
pub fn backward(
    outputs: &[&Tensor],
    grad_outputs: Option<&[Tensor]>,
    inputs: &[&Tensor],
    options: BackwardOptions,
) -> Result<()> {
    if options.create_graph {
        return Err(Error::UnsupportedAdOp {
            op: "backward(create_graph)",
        });
    }
    let grads = grad(
        outputs,
        inputs,
        grad_outputs,
        GradOptions {
            retain_graph: options.retain_graph,
            create_graph: false,
        },
    )?;
    for (input, grad) in inputs.iter().zip(grads.into_iter()) {
        if let Some(grad) = grad {
            input.accumulate_grad(&grad)?;
        }
    }
    Ok(())
}
