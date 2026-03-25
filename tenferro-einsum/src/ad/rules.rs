use std::collections::HashSet;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::{AdResult, Differentiable, DualValue};

use tenferro_device::LogicalMemorySpace;

use crate::api::{einsum, einsum_with_subscripts};

/// Creates an `n x n` identity tensor using only `Scalar` (zero/one) bounds.
fn make_delta<T: Scalar>(n: usize, space: LogicalMemorySpace) -> Result<Tensor<T>> {
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        data[i * n + i] = T::one();
    }
    Ok(Tensor::from_slice(
        &data,
        &[n, n],
        MemoryOrder::ColumnMajor,
    )?)
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
{
    let subs = Subscripts::parse(subscripts)?;
    let n = operands.len();
    let mut grads = Vec::with_capacity(n);

    // Build a size dictionary mapping labels to dimensions.
    let shapes: Vec<&[usize]> = operands.iter().map(|op| op.dims()).collect();
    let size_dict = crate::execution::util::build_size_dict(&subs, &shapes, None)?;

    for k in 0..n {
        let mut rev_inputs_subs = vec![subs.output.clone()];
        let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];

        for (i, &op) in operands.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                rev_operands.push(op);
            }
        }

        let rev_output = subs.inputs[k].clone();

        // Detect output-only labels: labels that appear in the reverse output
        // but not in any reverse input. This happens when the forward einsum
        // contracted labels that we now need to reconstruct (e.g., trace "ii->").
        // For each such label, inject an identity (delta) tensor so the reverse
        // einsum can produce the correct gradient.
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
                    "einsum rrule: missing dimension for label {}",
                    label
                ))
            })?;
            let space = operands[0].logical_memory_space();
            let eye = make_delta::<Alg::Scalar>(dim, space)?;
            rev_inputs_subs.push(vec![label, label]);
            delta_tensors.push(eye);
        }
        for dt in &delta_tensors {
            rev_operands.push(dt);
        }

        let rev_subs = Subscripts {
            inputs: rev_inputs_subs,
            output: rev_output,
        };

        let grad = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;
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
{
    let n = primals.len();
    let mut result: Option<Tensor<Alg::Scalar>> = None;

    for k in 0..n {
        if let Some(tangent_k) = tangents[k] {
            let mut ops: Vec<&Tensor<Alg::Scalar>> = primals.to_vec();
            ops[k] = tangent_k;

            let term = if let Some(nested) = nested {
                execute_nested::<Alg, Backend>(ctx, nested, &ops, None)?
            } else {
                einsum_with_subscripts::<Alg, Backend>(ctx, subs, &ops, None)?
            };

            result = Some(match result {
                None => term,
                Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
            });
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
{
    let subs = Subscripts::parse(subscripts)?;
    let n = primals.len();
    let shapes: Vec<&[usize]> = primals.iter().map(|op| op.dims()).collect();
    let size_dict = crate::execution::util::build_size_dict(&subs, &shapes, None)?;
    let mut results = Vec::with_capacity(n);

    for k in 0..n {
        let mut rev_inputs_subs = vec![subs.output.clone()];
        for (i, _) in primals.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
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
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                rev_operands.push(op);
            }
        }
        for dt in &delta_tensors {
            rev_operands.push(dt);
        }
        let grad_k = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;

        let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent_tangent];
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                ops.push(op);
            }
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
                for (i, &op) in primals.iter().enumerate() {
                    if i != k {
                        ops.push(if i == j { tangent_j } else { op });
                    }
                }
                for dt in &delta_tensors {
                    ops.push(dt);
                }
                let term = einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &ops, None)?;
                hvp_k = Some(match hvp_k {
                    None => term,
                    Some(existing) => Tensor::<Alg::Scalar>::accumulate_tangent(existing, &term),
                });
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
