use std::collections::HashSet;

use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::api::{einsum_with_subscripts, einsum_with_subscripts_into};
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::syntax::subscripts::Subscripts;

use super::rules::make_delta;

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
        let mut conj_store: Vec<Tensor<Alg::Scalar>> = Vec::new();
        for (i, &op) in primals.iter().enumerate() {
            if i != k {
                rev_inputs_subs.push(subs.inputs[i].clone());
                conj_store.push(op.conj());
            }
        }

        let rev_output = subs.inputs[k].clone();

        let all_input_labels: HashSet<u32> = rev_inputs_subs
            .iter()
            .flat_map(|labels| labels.iter().copied())
            .collect();
        let has_output_only = rev_output.iter().any(|l| !all_input_labels.contains(l));

        let (grad_k, hvp_k) = if has_output_only {
            let unique_output: Vec<u32> = {
                let mut seen = HashSet::new();
                rev_output
                    .iter()
                    .filter(|l| seen.insert(**l))
                    .copied()
                    .collect()
            };
            let mut delta_tensors: Vec<Tensor<Alg::Scalar>> = Vec::new();
            let mut delta_labels: Vec<u32> = Vec::new();
            for &label in &unique_output {
                if !all_input_labels.contains(&label) {
                    let dim = *size_dict.get(&label).ok_or_else(|| {
                        tenferro_device::Error::InvalidArgument(format!(
                            "einsum hvp: missing dimension for label {}",
                            label
                        ))
                    })?;
                    let space = primals[0].logical_memory_space();
                    delta_tensors.push(make_delta::<Alg::Scalar>(dim, space)?);
                    delta_labels.push(label);
                }
            }
            let mut fwd_inputs = rev_inputs_subs.clone();
            for &label in &delta_labels {
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

            let needs_embed = unique_output != rev_output;
            let embed_subs = if needs_embed {
                Some(Subscripts {
                    inputs: vec![unique_output.clone()],
                    output: rev_output.clone(),
                })
            } else {
                None
            };

            let compute_with = |ctx: &mut BackendContext<Alg, Backend>,
                                seed: &Tensor<Alg::Scalar>,
                                extra_conj: Option<(usize, &Tensor<Alg::Scalar>)>|
             -> Result<Tensor<Alg::Scalar>> {
                let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![seed];
                if let Some((j, tangent_j)) = extra_conj {
                    let mut ci = 0;
                    for (i, _) in primals.iter().enumerate() {
                        if i != k {
                            ops.push(if i == j { tangent_j } else { &conj_store[ci] });
                            ci += 1;
                        }
                    }
                } else {
                    for c in &conj_store {
                        ops.push(c);
                    }
                }
                for dt in &delta_tensors {
                    ops.push(dt);
                }
                let base = einsum_with_subscripts::<Alg, Backend>(ctx, &fwd_subs, &ops, None)?;
                if let Some(ref es) = embed_subs {
                    einsum_with_subscripts::<Alg, Backend>(ctx, es, &[&base], None)
                } else {
                    Ok(base)
                }
            };

            let grad_k = compute_with(ctx, cotangent, None)?;

            let mut hvp_k = compute_with(ctx, cotangent_tangent, None)?;

            for (j, tangent_j_opt) in tangents.iter().enumerate().take(n) {
                if j == k {
                    continue;
                }
                if let Some(tangent_j) = *tangent_j_opt {
                    let cross_term = compute_with(ctx, cotangent, Some((j, tangent_j)))?;
                    let one = <Alg::Scalar as num_traits::One>::one();
                    let acc_subs = Subscripts {
                        inputs: vec![rev_output.clone()],
                        output: rev_output.clone(),
                    };
                    einsum_with_subscripts_into::<Alg, Backend>(
                        ctx,
                        &acc_subs,
                        &[&cross_term],
                        one,
                        one,
                        &mut hvp_k,
                        None,
                    )?;
                }
            }

            (grad_k, hvp_k)
        } else {
            let rev_subs = Subscripts {
                inputs: rev_inputs_subs,
                output: rev_output.clone(),
            };

            let mut rev_operands: Vec<&Tensor<Alg::Scalar>> = vec![cotangent];
            for c in &conj_store {
                rev_operands.push(c);
            }
            let grad_k =
                einsum_with_subscripts::<Alg, Backend>(ctx, &rev_subs, &rev_operands, None)?;

            let mut ops: Vec<&Tensor<Alg::Scalar>> = vec![cotangent_tangent];
            for c in &conj_store {
                ops.push(c);
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
                    let mut ci = 0;
                    for (i, _) in primals.iter().enumerate() {
                        if i != k {
                            ops.push(if i == j { tangent_j } else { &conj_store[ci] });
                            ci += 1;
                        }
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

            let hvp_k = hvp_k.unwrap_or_else(|| {
                let space = primals[k].logical_memory_space();
                Tensor::zeros(primals[k].dims(), space, MemoryOrder::ColumnMajor).unwrap()
            });

            (grad_k, hvp_k)
        };

        results.push((grad_k, hvp_k));
    }

    Ok(results)
}
