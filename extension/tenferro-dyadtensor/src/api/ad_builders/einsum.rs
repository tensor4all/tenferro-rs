use std::collections::HashMap;

use super::super::*;

/// Builder for AD einsum.
/// # Examples
///
/// ```text
/// // Construct `EinsumAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EinsumAdBuilder<'a, T>
where
    T: EinsumRuntimeValue,
{
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
    size_dict: Option<&'a HashMap<u32, usize>>,
}

fn structured_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &Subscripts,
    reverse_nodes: &[Option<NodeId>],
    primals: &[StructuredTensor<T>],
    cotangent: &StructuredTensor<T>,
) -> Result<Vec<(NodeId, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorPrims<Standard<T>, Context = C>,
{
    let mut input_grads = Vec::new();

    for (k, maybe_node) in reverse_nodes.iter().enumerate() {
        let Some(node) = maybe_node else {
            continue;
        };
        let rev_subs = reverse_subscripts(subscripts, k);
        let mut rev_operands: Vec<&StructuredTensor<T>> = Vec::with_capacity(primals.len());
        rev_operands.push(cotangent);
        for (idx, operand) in primals.iter().enumerate() {
            if idx != k {
                rev_operands.push(operand);
            }
        }
        let grad = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &rev_subs, &rev_operands)?;
        input_grads.push((*node, grad.into_payload()));
    }

    Ok(input_grads)
}

fn dense_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    reverse_specs: &[Option<ReverseInputSpec<T>>],
    primals: &[Tensor<T>],
    cotangent: &Tensor<T>,
) -> Result<Vec<(NodeId, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>>,
    B: TensorPrims<Standard<T>, Context = C>,
{
    let primal_refs: Vec<&Tensor<T>> = primals.iter().collect();
    let gradients =
        tf_einsum::einsum_rrule::<Standard<T>, B>(ctx, subscripts, &primal_refs, cotangent)
            .map_err(Error::from)?;
    if gradients.len() != reverse_specs.len() {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "einsum_ad pullback arity mismatch: expected {}, got {}",
                reverse_specs.len(),
                gradients.len()
            ),
        });
    }

    let mut input_grads = Vec::new();
    for (k, grad) in gradients.into_iter().enumerate() {
        let Some(spec) = &reverse_specs[k] else {
            continue;
        };
        let grad =
            compress_pullback_like_in_backend::<B, _, T>(ctx, "einsum_ad", grad, &spec.layout)?;
        input_grads.push((spec.node, grad));
    }
    Ok(input_grads)
}

fn run_einsum_ad_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    operands: &[&AdTensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue,
    B: TensorPrims<Standard<T>, Context = C>,
{
    if size_dict.is_none() && !subscripts.contains('(') {
        let subs = Subscripts::parse(subscripts).map_err(Error::from)?;
        let primals: Vec<&StructuredTensor<T>> =
            operands.iter().map(|op| op.structured_primal()).collect();
        let primal_out = einsum_with_subscripts_in_ctx::<B, _, T>(ctx, &subs, &primals)?;

        let tangents = collect_structured_ad_tangents(operands);
        let tangent_out = if has_forward(operands) || has_any_tangent(operands) {
            sum_structured_einsum_tangent_terms::<B, _, T>(ctx, &subs, &primals, &tangents)?
        } else {
            None
        };

        let out = wrap_structured_ad_output("einsum_ad", operands, primal_out, tangent_out, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let subscripts = subs.clone();
            let reverse_nodes = collect_reverse_input_nodes(operands);
            let primal_owned: Vec<StructuredTensor<T>> =
                primals.iter().map(|tensor| (*tensor).clone()).collect();
            let output_layout = out.structured_primal().clone();
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<T>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let cotangent = output_layout.with_payload_like(cotangent.clone())?;
                    dispatch_einsum_runtime!(T, "einsum_ad_pullback_structured", |ctx, Backend| {
                        structured_einsum_pullback_in_backend::<Backend, _, T>(
                            ctx,
                            &subscripts,
                            &reverse_nodes,
                            &primal_owned,
                            &cotangent,
                        )
                    })
                }),
            )?;
        }

        return Ok(out);
    }

    let needs_tangent = has_forward(operands) || has_any_tangent(operands);
    let dense_inputs: Vec<(Tensor<T>, Option<Tensor<T>>)> = operands
        .iter()
        .map(|op| dense_input_snapshot_in_backend::<B, _, T>(ctx, op, needs_tangent))
        .collect::<Result<_>>()?;
    let primal_owned: Vec<Tensor<T>> = dense_inputs
        .iter()
        .map(|(primal, _)| primal.clone())
        .collect();
    let tangent_owned: Vec<Option<Tensor<T>>> = dense_inputs
        .iter()
        .map(|(_, tangent)| tangent.clone())
        .collect();
    let primals: Vec<&Tensor<T>> = primal_owned.iter().collect();
    let primal_out = tf_einsum::einsum::<Standard<T>, B>(ctx, subscripts, &primals, size_dict)
        .map_err(Error::from)?;

    let tangents: Vec<Option<&Tensor<T>>> = tangent_owned
        .iter()
        .map(|tangent| tangent.as_ref())
        .collect();
    let tangent_out = if needs_tangent {
        sum_einsum_tangent_terms::<B, _, T>(ctx, subscripts, &primals, &tangents, size_dict)?
    } else {
        None
    };

    let out = wrap_dense_ad_output("einsum_ad", operands, primal_out, tangent_out, 0)?;

    if let AdValue::Reverse { node, tape, .. } = out.as_value() {
        let subscripts = subscripts.to_string();
        let reverse_specs = collect_reverse_input_specs(operands);
        let output_node = *node;
        let tape_id = *tape;

        reverse_tape::register_rule::<T>(
            tape_id,
            output_node,
            Box::new(move |cotangent| {
                dispatch_einsum_runtime!(T, "einsum_ad_pullback", |ctx, Backend| {
                    dense_einsum_pullback_in_backend::<Backend, _, T>(
                        ctx,
                        &subscripts,
                        &reverse_specs,
                        &primal_owned,
                        cotangent,
                    )
                })
            }),
        )?;
    }

    Ok(out)
}

impl<'a, T> EinsumAdBuilder<'a, T>
where
    T: EinsumRuntimeValue,
{
    /// Sets optional size dictionary for output-only labels.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes AD einsum with mode propagation.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>>
    where
        T: 'static,
    {
        let subscripts = self.subscripts;
        let operands = self.operands;
        let size_dict = self.size_dict;
        dispatch_einsum_runtime!(T, "einsum_ad", |ctx, Backend| {
            run_einsum_ad_in_backend::<Backend, _, T>(ctx, subscripts, operands, size_dict)
        })
    }
}

/// Creates a builder for AD einsum.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{einsum_ad, set_default_runtime, AdTensor, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let ad_a = AdTensor::new_primal(a);
/// let ad_b = AdTensor::new_primal(b);
/// let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b]).run().unwrap();
/// assert_eq!(out.dims(), &[2, 2]);
/// ```
pub fn einsum_ad<'a, T>(
    subscripts: &'a str,
    operands: &'a [&'a AdTensor<T>],
) -> EinsumAdBuilder<'a, T>
where
    T: EinsumRuntimeValue,
{
    EinsumAdBuilder {
        subscripts,
        operands,
        size_dict: None,
    }
}
