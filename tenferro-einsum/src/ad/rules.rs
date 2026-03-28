use std::collections::HashSet;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::{AdResult, Differentiable, DualValue};

use tenferro_device::LogicalMemorySpace;

use crate::api::{einsum, einsum_with_subscripts, einsum_with_subscripts_into};

/// Creates an `n x n` identity tensor using only `Scalar` (zero/one) bounds.
fn make_delta<T: Scalar>(n: usize, _space: LogicalMemorySpace) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Tensor::from_slice(&data, &[n, n], MemoryOrder::ColumnMajor)
}
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::execute::execute_nested;
use crate::syntax::nested::NestedEinsum;
use crate::syntax::subscripts::Subscripts;

/// Dual einsum (forward-mode JVP propagation).
///
/// # Examples
///
/// ```ignore
/// use tidu::DualValue;
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::dual_einsum;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let a = DualValue::with_tangent(a_primal, a_tangent).unwrap();
/// let b = DualValue::new(b_primal);
/// let out = dual_einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b]).unwrap();
/// let _ = out.tangent();
/// ```
pub fn dual_einsum<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    operands: &[&DualValue<Tensor<Alg::Scalar>>],
) -> AdResult<DualValue<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    Tensor<Alg::Scalar>: Differentiable<Tangent = Tensor<Alg::Scalar>>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let primals: Vec<&Tensor<Alg::Scalar>> = operands.iter().map(|op| op.primal()).collect();
    let output = einsum::<Alg, Backend>(ctx, subscripts, &primals, None)
        .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;

    let tangents: Vec<Option<&Tensor<Alg::Scalar>>> =
        operands.iter().map(|op| op.tangent()).collect();
    if tangents.iter().all(|t| t.is_none()) {
        return Ok(DualValue::new(output));
    }

    let tangent = einsum_frule::<Alg, Backend>(ctx, subscripts, &primals, &tangents)
        .map_err(|e| tidu::AutodiffError::InvalidArgument(format!("{e}")))?;
    DualValue::with_tangent(output, tangent)
}

/// Reverse-mode rule (rrule) for einsum without building a global tape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_rrule;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let grads = einsum_rrule::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &dc)
///     .unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn einsum_rrule<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    operands: &[&Tensor<Alg::Scalar>],
    cotangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<Tensor<Alg::Scalar>>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = operands.len();
    let mut grads = Vec::with_capacity(n);

    // Build a size dictionary mapping labels to dimensions.
    let shapes: Vec<&[usize]> = operands.iter().map(|op| op.dims()).collect();
    let size_dict = crate::execution::util::build_size_dict(&subs, &shapes, None)?;

    for k in 0..n {
        let mut rev_inputs_subs = vec![subs.output.clone()];
        // Wirtinger VJP: conjugate non-cotangent operands so that the
        // reverse einsum computes  grad_k = einsum(cot, conj(other_ops)...).
        // For real scalars conj() is a no-op (zero-cost flag toggle on Tensor).
        let mut conj_store: Vec<Tensor<Alg::Scalar>> = Vec::new();
        for (i, &op) in operands.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                conj_store.push(op.conj());
            }
        }

        let rev_output = subs.inputs[k].clone();

        // Detect output-only labels: labels that appear in the reverse output
        // but not in any reverse input. This happens when the forward einsum
        // contracted labels that we now need to reconstruct (e.g., trace "ii->").
        let all_input_labels: HashSet<u32> = rev_inputs_subs
            .iter()
            .flat_map(|labels| labels.iter().copied())
            .collect();
        let has_output_only = rev_output.iter().any(|l| !all_input_labels.contains(l));

        let grad = if has_output_only {
            // For subscripts with output-only labels (e.g. trace "ii->"),
            // the reverse einsum "->ii" is invalid. Build the gradient
            // directly: cotangent broadcast-multiplied with the appropriate
            // identity structure.
            //
            // Step 1: build a delta tensor whose subscripts cover ALL
            //         output-only labels with fresh paired indices.
            // Step 2: einsum the cotangent with any remaining operands and
            //         the delta, producing a tensor with the right unique
            //         output labels.
            // Step 3: use "i->ii" (diagonal embedding) via forward einsum
            //         if the output has repeated labels.
            //
            // For the common single-operand trace "ii->":
            //   unique output labels = {i}, reverse needs shape [n, n]
            //   grad = cotangent * eye(n)
            let unique_output: Vec<u32> = {
                let mut seen = HashSet::new();
                rev_output
                    .iter()
                    .filter(|l| seen.insert(**l))
                    .copied()
                    .collect()
            };
            // Build a delta tensor for each missing label
            let mut delta_subs: Vec<u32> = Vec::new();
            let mut delta_tensors: Vec<Tensor<Alg::Scalar>> = Vec::new();
            for &label in &unique_output {
                if !all_input_labels.contains(&label) {
                    let dim = *size_dict.get(&label).ok_or_else(|| {
                        tenferro_device::Error::InvalidArgument(format!(
                            "einsum rrule: missing dimension for label {}",
                            label
                        ))
                    })?;
                    let space = operands[0].logical_memory_space();
                    delta_tensors.push(make_delta::<Alg::Scalar>(dim, space)?);
                    delta_subs.push(label);
                }
            }
            // Build subscripts: cotangent + conj_ops + deltas -> unique_output
            let mut fwd_inputs = rev_inputs_subs.clone();
            for &label in &delta_subs {
                // delta with subscript [label, label_fresh] where label_fresh
                // is a new label not used anywhere.
                let max_label = subs
                    .inputs
                    .iter()
                    .flat_map(|v| v.iter())
                    .chain(subs.output.iter())
                    .copied()
                    .max()
                    .unwrap_or(0);
                let fresh = max_label + 1 + (fwd_inputs.len() as u32);
                fwd_inputs.push(vec![label, fresh]);
            }
            let fwd_subs = Subscripts {
                inputs: fwd_inputs,
                output: unique_output.clone(),
            };
            let mut fwd_ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
            for c in &conj_store {
                fwd_ops.push(c);
            }
            for dt in &delta_tensors {
                fwd_ops.push(dt);
            }
            // This gives a tensor with unique_output labels
            let base = einsum_with_subscripts::<Alg, Backend>(ctx, &fwd_subs, &fwd_ops, None)?;

            // If unique_output != rev_output, need diagonal embedding
            if unique_output == rev_output {
                base
            } else {
                // Diagonal embedding: e.g. "i->ii"
                let embed_subs = Subscripts {
                    inputs: vec![unique_output],
                    output: rev_output.clone(),
                };
                einsum_with_subscripts::<Alg, Backend>(ctx, &embed_subs, &[&base], None)?
            }
        } else {
            let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
            for c in &conj_store {
                rev_operands.push(c);
            }
            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: rev_output,
            };
            einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?
        };
        grads.push(grad);
    }

    Ok(grads)
}

/// Forward-mode rule (frule) for einsum without building a global tape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_frule;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let tangent =
///     einsum_frule::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None])
///         .unwrap();
/// ```
pub fn einsum_frule<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let subs = Subscripts::parse(subscripts)?;
    let nested = if subscripts.contains('(') {
        Some(NestedEinsum::parse(subscripts)?)
    } else {
        None
    };
    einsum_frule_impl::<Alg, Backend>(ctx, &subs, nested.as_ref(), primals, tangents)
}

/// Internal frule implementation with pre-parsed subscripts.
pub(crate) fn einsum_frule_impl<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subs: &Subscripts,
    nested: Option<&NestedEinsum>,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let n = primals.len();
    let mut result: Option<Tensor<Alg::Scalar>> = None;

    for k in 0..n {
        if let Some(tangent_k) = tangents[k] {
            let mut ops: Vec<&Tensor<Alg::Scalar>> = primals.to_vec();
            ops[k] = tangent_k;

            match &mut result {
                None => {
                    let term = if let Some(nested) = nested {
                        execute_nested::<Alg, Backend>(ctx, nested, &ops, None)?
                    } else {
                        einsum_with_subscripts::<Alg, Backend>(ctx, subs, &ops, None)?
                    };
                    result = Some(term);
                }
                Some(existing) => {
                    let one = <Alg::Scalar as num_traits::One>::one();
                    if let Some(nested) = nested {
                        // Nested einsum does not support _into; materialize then
                        // accumulate via an identity contraction (the nested tree
                        // may have fewer roots than `subs.inputs`).
                        let term = execute_nested::<Alg, Backend>(ctx, nested, &ops, None)?;
                        let out_labels: &[u32] = &subs.output;
                        let identity_subs = Subscripts::new(&[out_labels], out_labels);
                        einsum_with_subscripts_into::<Alg, Backend>(
                            ctx,
                            &identity_subs,
                            &[&term],
                            one,
                            one,
                            existing,
                            None,
                        )?;
                    } else {
                        einsum_with_subscripts_into::<Alg, Backend>(
                            ctx, subs, &ops, one, one, existing, None,
                        )?;
                    }
                }
            }
        }
    }

    match result {
        Some(r) => Ok(r),
        None => {
            let primal_out = if let Some(nested) = nested {
                execute_nested::<Alg, Backend>(ctx, nested, primals, None)?
            } else {
                einsum_with_subscripts::<Alg, Backend>(ctx, subs, primals, None)?
            };
            Tensor::<Alg::Scalar>::zeros(
                primal_out.dims(),
                primal_out.logical_memory_space(),
                MemoryOrder::ColumnMajor,
            )
        }
    }
}

/// Local HVP rule for einsum without building a global tape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_algebra::Standard;
/// use tenferro_einsum::einsum_hvp;
/// use tenferro_prims::{CpuBackend, CpuContext};
///
/// let mut ctx = CpuContext::new(1);
/// let hvps = einsum_hvp::<Standard<f64>, CpuBackend>(
///     &mut ctx,
///     "ij,jk->ik",
///     &[&a, &b],
///     &[Some(&da), None],
///     &dc,
///     &ddc,
/// ).unwrap();
/// assert_eq!(hvps.len(), 2);
/// ```
pub fn einsum_hvp<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    subscripts: &str,
    primals: &[&Tensor<Alg::Scalar>],
    tangents: &[Option<&Tensor<Alg::Scalar>>],
    cotangent: &Tensor<Alg::Scalar>,
    cotangent_tangent: &Tensor<Alg::Scalar>,
) -> Result<Vec<(Tensor<Alg::Scalar>, Tensor<Alg::Scalar>)>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let subs = Subscripts::parse(subscripts)?;
    let n = primals.len();
    let mut results = Vec::with_capacity(n);

    let shapes: Vec<&[usize]> = primals.iter().map(|op| op.dims()).collect();
    let size_dict = crate::execution::util::build_size_dict(&subs, &shapes, None)?;

    for k in 0..n {
        let mut rev_inputs_subs = vec![subs.output.clone()];
        // Wirtinger VJP: conjugate non-cotangent operands (same as einsum_rrule).
        let mut conj_store: Vec<Tensor<Alg::Scalar>> = Vec::new();
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                conj_store.push(op.conj());
            }
        }

        let rev_output = subs.inputs[k].clone();

        // Inject delta tensors for output-only labels (same as einsum_rrule).
        let all_input_labels: HashSet<u32> = rev_inputs_subs
            .iter()
            .flat_map(|labels| labels.iter().copied())
            .collect();
        let mut unique_missing = Vec::new();
        {
            let mut seen = HashSet::new();
            for &label in &rev_output {
                if !all_input_labels.contains(&label) && seen.insert(label) {
                    unique_missing.push(label);
                }
            }
        }
        let mut delta_tensors: Vec<Tensor<Alg::Scalar>> = Vec::new();
        for &label in &unique_missing {
            let dim = *size_dict.get(&label).ok_or_else(|| {
                tenferro_device::Error::InvalidArgument(format!(
                    "einsum hvp: missing dimension for label {}",
                    label
                ))
            })?;
            let space = primals[0].logical_memory_space();
            let eye = make_delta::<Alg::Scalar>(dim, space)?;
            rev_inputs_subs.push(vec![label, label]);
            delta_tensors.push(eye);
        }

        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: rev_output,
        };

        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
        for c in &conj_store {
            rev_operands.push(c);
        }
        for dt in &delta_tensors {
            rev_operands.push(dt);
        }
        let grad_k = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;

        let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent_tangent];
        for c in &conj_store {
            ops.push(c);
        }
        for dt in &delta_tensors {
            ops.push(dt);
        }
        let mut hvp_k = Some(einsum_with_subscripts::<Alg, Backend>(
            ctx, &rev_subs, &ops, None,
        )?);

        for (j, tangent_j_opt) in tangents.iter().enumerate().take(n) {
            if j == k {
                continue;
            }
            if let Some(tangent_j) = *tangent_j_opt {
                let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
                // conj_store has one entry per i != k, in order.
                let mut ci = 0;
                for (i, _) in primals.iter().enumerate() {
                    if i != k {
                        ops.push(if i == j { tangent_j } else { &conj_store[ci] });
                        ci += 1;
                    }
                }
                for dt in &delta_tensors {
                    ops.push(dt);
                }
                match &mut hvp_k {
                    None => {
                        hvp_k = Some(einsum_with_subscripts::<Alg, Backend>(
                            ctx, &rev_subs, &ops, None,
                        )?);
                    }
                    Some(existing) => {
                        let one = <Alg::Scalar as num_traits::One>::one();
                        einsum_with_subscripts_into::<Alg, Backend>(
                            ctx, &rev_subs, &ops, one, one, existing, None,
                        )?;
                    }
                }
            }
        }

        let hvp_k = match hvp_k {
            Some(t) => t,
            None => {
                let space = primals[k].logical_memory_space();
                Tensor::zeros(primals[k].dims(), space, MemoryOrder::ColumnMajor)?
            }
        };

        results.push((grad_k, hvp_k));
    }

    Ok(results)
}
