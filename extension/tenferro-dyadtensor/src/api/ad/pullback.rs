use super::*;

/// Reverse pullback from a reverse-mode output tensor.
///
/// The returned map is keyed by reverse `NodeId` and includes the seed
/// cotangent for the output node itself.
pub fn pullback<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
) -> Result<HashMap<NodeId, Tensor<T>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback requires reverse-mode output tensor".to_string(),
            });
        }
    };

    let cotangent_payload = normalize_cotangent_payload(output, cotangent, "ad::pullback")?;
    reverse_tape::pullback(tape, output_node, &cotangent_payload)
}

/// Reverse pullback projected to requested `wrt` tensors.
///
/// Returns `None` for non-reverse tensors or disconnected reverse tensors.
pub fn pullback_wrt<T: Scalar + 'static>(
    output: &AdTensor<T>,
    cotangent: &AdTensor<T>,
    wrt: &[&AdTensor<T>],
) -> Result<Vec<Option<StructuredTensor<T>>>> {
    let tape = match output.as_value() {
        AdValue::Reverse { tape, .. } => *tape,
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt requires reverse-mode output tensor".to_string(),
            });
        }
    };

    let all_grads = pullback(output, cotangent)?;
    let mut out = Vec::with_capacity(wrt.len());

    for wrt_tensor in wrt {
        match wrt_tensor.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                let grad = all_grads
                    .get(node)
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

/// Reverse pullback projected to requested `wrt` tensors with a different scalar type.
///
/// This is used for mixed-domain rules such as `eig_ad` where outputs are complex
/// while inputs are real.
pub fn pullback_wrt_mixed<TOut: Scalar + 'static, TWrt: Scalar + 'static>(
    output: &AdTensor<TOut>,
    cotangent: &AdTensor<TOut>,
    wrt: &[&AdTensor<TWrt>],
) -> Result<Vec<Option<StructuredTensor<TWrt>>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt_mixed requires reverse-mode output tensor".to_string(),
            });
        }
    };

    let mut wrt_nodes = Vec::with_capacity(wrt.len());
    for wrt_tensor in wrt {
        match wrt_tensor.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                wrt_nodes.push(Some(*node));
            }
            _ => wrt_nodes.push(None),
        }
    }

    let cotangent_payload =
        normalize_cotangent_payload(output, cotangent, "ad::pullback_wrt_mixed")?;
    let grads = reverse_tape::pullback_wrt_mixed::<TOut, TWrt>(
        tape,
        output_node,
        &cotangent_payload,
        &wrt_nodes,
    )?;

    grads
        .into_iter()
        .zip(wrt.iter())
        .map(|(grad, wrt_tensor)| {
            grad.map(|payload| {
                StructuredTensor::new(
                    wrt_tensor.dims().to_vec(),
                    wrt_tensor.axis_classes().to_vec(),
                    payload,
                )
            })
            .transpose()
        })
        .collect()
}

/// Reverse pullback projected to requested scalar inputs.
///
/// This is used by tensor outputs whose reverse rule depends on scalar
/// coefficients, such as `DynAdTensor::scale` and `DynAdTensor::axpby`.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{ad, AdScalar, AdTensor, AdValue, DynAdScalar, DynAdTensor, NodeId, TapeId};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let x: DynAdTensor = AdTensor::new_reverse(
///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
///     NodeId(1),
///     TapeId(7),
///     None,
/// )
/// .unwrap()
/// .into();
/// let a = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(2), TapeId(7), None));
/// let y = x.scale(&a).unwrap();
/// let cotangent = AdTensor::new_primal(
///     Tensor::<f64>::from_slice(&[0.5, 1.25], &[2], MemoryOrder::ColumnMajor).unwrap(),
/// );
/// let a_typed = AdScalar::from(a.as_f64().unwrap().clone());
/// let grads = ad::pullback_wrt_scalars(y.as_f64().unwrap(), &cotangent, &[&a_typed]).unwrap();
/// assert_eq!(grads, vec![Some(3.0)]);
/// ```
pub fn pullback_wrt_scalars<TOut: Scalar + 'static, TWrt: ScalarAd + 'static>(
    output: &AdTensor<TOut>,
    cotangent: &AdTensor<TOut>,
    wrt: &[&AdScalar<TWrt>],
) -> Result<Vec<Option<TWrt>>> {
    let (output_node, tape) = match output.as_value() {
        AdValue::Reverse { node, tape, .. } => (*node, *tape),
        _ => {
            return Err(Error::InvalidAdTensor {
                message: "ad::pullback_wrt_scalars requires reverse-mode output tensor".to_string(),
            });
        }
    };

    let mut wrt_nodes = Vec::with_capacity(wrt.len());
    for wrt_scalar in wrt {
        match wrt_scalar.as_value() {
            AdValue::Reverse { node, tape: t, .. } => {
                if *t != tape {
                    return Err(Error::MixedReverseTape {
                        expected: tape.0,
                        found: t.0,
                    });
                }
                wrt_nodes.push(Some(*node));
            }
            _ => wrt_nodes.push(None),
        }
    }

    let cotangent_payload =
        normalize_cotangent_payload(output, cotangent, "ad::pullback_wrt_scalars")?;
    reverse_tape::pullback_wrt_scalars::<TOut, TWrt>(
        tape,
        output_node,
        &cotangent_payload,
        &wrt_nodes,
    )
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
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    let primals: Vec<&Tensor<T>> = operands.iter().map(|op| op.primal()).collect();
    super::with_einsum_runtime::<T, _>(
        "einsum_rrule",
        |ctx| {
            tf_einsum::einsum_rrule::<Standard<T>, CpuBackend>(
                ctx,
                subscripts,
                &primals,
                cotangent.primal(),
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_rrule::<Standard<T>, tenferro_prims::CudaBackend>(
                ctx,
                subscripts,
                &primals,
                cotangent.primal(),
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_rrule::<Standard<T>, tenferro_prims::RocmBackend>(
                ctx,
                subscripts,
                &primals,
                cotangent.primal(),
            )
            .map_err(Error::from)
        },
    )
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
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
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

    super::with_einsum_runtime::<T, _>(
        "einsum_frule",
        |ctx| {
            tf_einsum::einsum_frule::<Standard<T>, CpuBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_frule::<Standard<T>, tenferro_prims::CudaBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_frule::<Standard<T>, tenferro_prims::RocmBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
            )
            .map_err(Error::from)
        },
    )
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
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    CpuBackend: TensorPrims<Standard<T>, Context = CpuContext>,
    tenferro_prims::CudaBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    tenferro_prims::RocmBackend: TensorPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
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

    super::with_einsum_runtime::<T, _>(
        "einsum_hvp",
        |ctx| {
            tf_einsum::einsum_hvp::<Standard<T>, CpuBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
                cotangent.primal(),
                cotangent_tangent.primal(),
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_hvp::<Standard<T>, tenferro_prims::CudaBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
                cotangent.primal(),
                cotangent_tangent.primal(),
            )
            .map_err(Error::from)
        },
        |ctx| {
            tf_einsum::einsum_hvp::<Standard<T>, tenferro_prims::RocmBackend>(
                ctx,
                subscripts,
                &primal_refs,
                &tangent_refs,
                cotangent.primal(),
                cotangent_tangent.primal(),
            )
            .map_err(Error::from)
        },
    )
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
    T: LinalgScalar + CpuLinalgScalar,
{
    super::with_linalg_runtime::<T, _>(
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
