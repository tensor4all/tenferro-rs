use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring};
use tenferro_device::Result;
use tenferro_prims::TensorTempPoolContext;
use tenferro_tensor::{MemoryOrder, Tensor};
use tidu::{AdResult, Differentiable, DualValue};

use crate::ad::delta::{prepare_reverse_context, ReverseContext};
use crate::api::{einsum, einsum_with_subscripts, einsum_with_subscripts_into};
use crate::execution::backend::{BackendContext, EinsumBackend};
use crate::execution::execute::execute_nested;
use crate::syntax::nested::NestedEinsum;
use crate::syntax::subscripts::Subscripts;

fn eval_with_embed<Alg, Backend>(
    ctx: &mut BackendContext<Alg, Backend>,
    rctx: &ReverseContext<Alg::Scalar>,
    leading: &Tensor<Alg::Scalar>,
) -> Result<Tensor<Alg::Scalar>>
where
    Alg: Semiring,
    Alg::Scalar: Scalar + Conjugate + HasAlgebra<Algebra = Alg>,
    Backend: EinsumBackend<Alg>,
    BackendContext<Alg, Backend>: TensorTempPoolContext,
{
    let ops = rctx.assemble_rev_operands(leading);
    let base = einsum_with_subscripts::<Alg, Backend>(ctx, &rctx.dctx.base_subs, &ops, None)?;
    if let Some(ref es) = rctx.dctx.embed_subs {
        einsum_with_subscripts::<Alg, Backend>(ctx, es, &[&base], None)
    } else {
        Ok(base)
    }
}

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
        let rctx = prepare_reverse_context::<Alg::Scalar>(&subs, operands, k, &size_dict)?;
        let grad = eval_with_embed::<Alg, Backend>(ctx, &rctx, cotangent)?;
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
        let rctx = prepare_reverse_context::<Alg::Scalar>(&subs, primals, k, &size_dict)?;

        let grad_k = eval_with_embed::<Alg, Backend>(ctx, &rctx, cotangent)?;

        let mut hvp_base = Some(einsum_with_subscripts::<Alg, Backend>(
            ctx,
            &rctx.dctx.base_subs,
            &rctx.assemble_rev_operands(cotangent_tangent),
            None,
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
                        ops.push(if i == j {
                            tangent_j
                        } else {
                            &rctx.conj_store[ci]
                        });
                        ci += 1;
                    }
                }
                for dt in &rctx.dctx.delta_tensors {
                    ops.push(dt);
                }
                match &mut hvp_base {
                    None => {
                        hvp_base = Some(einsum_with_subscripts::<Alg, Backend>(
                            ctx,
                            &rctx.dctx.base_subs,
                            &ops,
                            None,
                        )?);
                    }
                    Some(existing) => {
                        let one = <Alg::Scalar as num_traits::One>::one();
                        einsum_with_subscripts_into::<Alg, Backend>(
                            ctx,
                            &rctx.dctx.base_subs,
                            &ops,
                            one,
                            one,
                            existing,
                            None,
                        )?;
                    }
                }
            }
        }

        let hvp_k = match hvp_base {
            Some(t) => {
                if let Some(ref es) = rctx.dctx.embed_subs {
                    einsum_with_subscripts::<Alg, Backend>(ctx, es, &[&t], None)?
                } else {
                    t
                }
            }
            None => {
                let space = primals[k].logical_memory_space();
                Tensor::zeros(primals[k].dims(), space, MemoryOrder::ColumnMajor)?
            }
        };

        results.push((grad_k, hvp_k));
    }

    Ok(results)
}
