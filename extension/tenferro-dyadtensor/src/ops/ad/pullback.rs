use super::*;
use crate::runtime::contracts::LinalgRuntimeValue;
use crate::runtime::dispatch::{dispatch_einsum_runtime, with_linalg_runtime};

/// Reverse pullback from a reverse-mode output tensor.
///
/// The returned map is keyed by reverse `NodeId` and includes the seed
/// cotangent for the output node itself.
pub fn pullback<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    if output.reverse_tape().is_none() {
        return Err(Error::InvalidAdTensor {
            message: "ad::pullback requires reverse-mode output tensor".to_string(),
        });
    }

    let cotangent_payload = normalize_cotangent_payload(output, cotangent, "ad::pullback")?;
    tape::pullback(output, &cotangent_payload)
}

/// Reverse pullback projected to requested `wrt` tensors.
///
/// Returns `None` for non-reverse tensors or disconnected reverse tensors.
pub fn pullback_wrt<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    wrt: &[&AdTensor<T>],
) -> Result<Vec<Option<StructuredTensor<T>>>> {
    let tape = output
        .reverse_tape()
        .cloned()
        .ok_or_else(|| Error::InvalidAdTensor {
            message: "ad::pullback_wrt requires reverse-mode output tensor".to_string(),
        })?;

    let all_grads = pullback(output, cotangent)?;
    let mut out = Vec::with_capacity(wrt.len());

    for wrt_tensor in wrt {
        match (wrt_tensor.reverse_node_id(), wrt_tensor.reverse_tape()) {
            (Some(node), Some(t)) => {
                if !t.same_tape(&tape) {
                    return Err(Error::MixedReverseTape {
                        expected: tape.id() as u64,
                        found: t.id() as u64,
                    });
                }
                let grad = all_grads
                    .get(&node)
                    .map(|payload| {
                        StructuredTensor::new(
                            wrt_tensor.dims().to_vec(),
                            wrt_tensor.axis_classes().to_vec(),
                            payload.clone(),
                        )
                    })
                    .transpose()?;
                out.push(grad);
            }
            _ => out.push(None),
        }
    }

    Ok(out)
}

/// Local reverse-mode rule (VJP) for einsum.
///
/// Stateless helper for interop/manual AD paths. Inputs are AD tensors, but
/// derivatives are computed from their primal payloads.
pub fn einsum_rrule<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
    cotangent: &AdTensor<T>,
) -> Result<Vec<Tensor<T>>>
where
    T: EinsumRuntimeValue,
{
    let primals: Vec<&Tensor<T>> = operands.iter().map(|op| op.primal()).collect();
    dispatch_einsum_runtime!(T, "einsum_rrule", |ctx, Backend| {
        tf_einsum::einsum_rrule::<Standard<T>, Backend>(
            ctx,
            subscripts,
            &primals,
            cotangent.primal(),
        )
        .map_err(Error::from)
    })
}

/// Local forward-mode rule (JVP) for einsum.
///
/// `tangents` must have the same length as `primals`.
pub fn einsum_frule<'a, T>(
    subscripts: &'a str,
    primals: &'a [&'a AdTensor<T>],
    tangents: &'a [Option<&'a AdTensor<T>>],
) -> Result<Tensor<T>>
where
    T: EinsumRuntimeValue,
{
    if primals.len() != tangents.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_frule requires tangents.len() == primals.len(), got {} vs {}",
                tangents.len(),
                primals.len()
            ),
        });
    }

    let primal_refs: Vec<&Tensor<T>> = primals.iter().map(|op| op.primal()).collect();
    let tangent_refs: Vec<Option<&Tensor<T>>> = tangents
        .iter()
        .map(|opt| opt.as_ref().map(|t| t.primal()))
        .collect();

    dispatch_einsum_runtime!(T, "einsum_frule", |ctx, Backend| {
        tf_einsum::einsum_frule::<Standard<T>, Backend>(
            ctx,
            subscripts,
            &primal_refs,
            &tangent_refs,
        )
        .map_err(Error::from)
    })
}

/// Local Hessian-vector product helper for einsum.
///
/// Returns one `(grad_k, hvp_k)` pair per input operand.
///
/// `tangents` must have the same length as `primals`.
pub fn einsum_hvp<'a, T>(
    subscripts: &'a str,
    primals: &'a [&'a AdTensor<T>],
    tangents: &'a [Option<&'a AdTensor<T>>],
    cotangent: &AdTensor<T>,
    cotangent_tangent: &AdTensor<T>,
) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
where
    T: EinsumRuntimeValue,
{
    if primals.len() != tangents.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_hvp requires tangents.len() == primals.len(), got {} vs {}",
                tangents.len(),
                primals.len()
            ),
        });
    }

    let primal_refs: Vec<&Tensor<T>> = primals.iter().map(|op| op.primal()).collect();
    let tangent_refs: Vec<Option<&Tensor<T>>> = tangents
        .iter()
        .map(|opt| opt.as_ref().map(|t| t.primal()))
        .collect();

    dispatch_einsum_runtime!(T, "einsum_hvp", |ctx, Backend| {
        tf_einsum::einsum_hvp::<Standard<T>, Backend>(
            ctx,
            subscripts,
            &primal_refs,
            &tangent_refs,
            cotangent.primal(),
            cotangent_tangent.primal(),
        )
        .map_err(Error::from)
    })
}

/// Local reverse-mode rule (VJP) for triangular solve.
///
/// This is the stateless wrapper for `tenferro_linalg::solve_triangular_rrule`.
pub fn solve_triangular_rrule<T: Scalar>(
    a: &AdTensor<T>,
    b: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    upper: bool,
) -> Result<SolveGrad<T>>
where
    T: LinalgRuntimeValue,
{
    with_linalg_runtime::<T, _>(
        "solve_triangular_rrule",
        tenferro_linalg::backend::LinalgCapabilityOp::SolveTriangular,
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(
                ctx,
                a.primal(),
                b.primal(),
                cotangent.primal(),
                upper,
            )
            .map_err(Error::from)
        },
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(
                ctx,
                a.primal(),
                b.primal(),
                cotangent.primal(),
                upper,
            )
            .map_err(Error::from)
        },
        |ctx| {
            tenferro_linalg::solve_triangular_rrule::<T, _>(
                ctx,
                a.primal(),
                b.primal(),
                cotangent.primal(),
                upper,
            )
            .map_err(Error::from)
        },
    )
}
