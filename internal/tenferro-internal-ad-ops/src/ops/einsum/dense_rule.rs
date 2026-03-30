use chainrules_core::{AdResult, AutodiffError};
use tenferro_tensor::Tensor;
use tidu::expert::ReverseRule;

use super::super::*;
use tenferro_prims::TensorTempPoolContext;

fn subscripts_to_string(subs: &tf_einsum::Subscripts) -> String {
    let label_to_char = |label: u32| -> char { char::from(b'a' + (label % 26) as u8) };
    let mut result = String::new();
    for (i, input) in subs.inputs.iter().enumerate() {
        if i > 0 {
            result.push(',');
        }
        for &label in input {
            result.push(label_to_char(label));
        }
    }
    result.push_str("->");
    for &label in &subs.output {
        result.push(label_to_char(label));
    }
    result
}

/// HVP-capable reverse rule for the dense einsum path (Level B).
///
/// Implements `ReverseRule<Tensor<T>>` with `forward_tangents` and
/// `pullback_with_tangents` support, delegating to `tf_einsum::einsum_rrule`
/// and `tf_einsum::einsum_frule`.
pub(crate) struct DenseEinsumRule<T: EinsumRuntimeValue> {
    pub(crate) subscripts: String,
    pub(crate) primals: Vec<Tensor<T>>,
    pub(crate) reverse_specs: Vec<Option<ReverseInputSpec<T>>>,
}

fn dense_einsum_pullback_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    reverse_specs: &[Option<ReverseInputSpec<T>>],
    primals: &[Tensor<T>],
    cotangent: &Tensor<T>,
) -> Result<Vec<(NodeId, StructuredTensor<T>)>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
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
        let grad = spec
            .layout
            .with_payload_like(compress_pullback_like_in_backend::<B, _, T>(
                ctx,
                "einsum_ad",
                grad,
                &spec.layout,
            )?)?;
        input_grads.push((spec.node, grad));
    }
    Ok(input_grads)
}

impl<T: EinsumRuntimeValue + 'static> ReverseRule<Tensor<T>> for DenseEinsumRule<T> {
    fn pullback(&self, cotangent: &Tensor<T>) -> AdResult<Vec<(NodeId, Tensor<T>)>> {
        dispatch_einsum_runtime!(T, "einsum_ad_pullback_dense_rule", |ctx, Backend| {
            dense_einsum_pullback_in_backend::<Backend, _, T>(
                ctx,
                &self.subscripts,
                &self.reverse_specs,
                &self.primals,
                cotangent,
            )
        })
        .map(|entries| {
            entries
                .into_iter()
                .map(|(node, st)| (node, st.into_payload()))
                .collect()
        })
        .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.reverse_specs
            .iter()
            .filter_map(|s| s.as_ref().map(|s| s.node))
            .collect()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<T>>,
    ) -> AdResult<Option<Tensor<T>>>
    where
        Tensor<T>: 't,
    {
        let tangents: Vec<Option<&Tensor<T>>> = self
            .reverse_specs
            .iter()
            .map(|spec| spec.as_ref().and_then(|s| input_tangents(s.node)))
            .collect();
        if tangents.iter().all(|t| t.is_none()) {
            return Ok(None);
        }
        let primals: Vec<&Tensor<T>> = self.primals.iter().collect();
        dispatch_einsum_runtime!(T, "einsum_ad_forward_tangents", |ctx, Backend| {
            tf_einsum::einsum_frule::<Standard<T>, Backend>(
                ctx,
                &self.subscripts,
                &primals,
                &tangents,
            )
            .map_err(Error::from)
        })
        .map(Some)
        .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &Tensor<T>,
        cotangent_tangent: &Tensor<T>,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<T>>,
    ) -> AdResult<Vec<(NodeId, Tensor<T>, Tensor<T>)>>
    where
        Tensor<T>: 't,
    {
        let n = self.primals.len();
        let primals: Vec<&Tensor<T>> = self.primals.iter().collect();

        dispatch_einsum_runtime!(T, "einsum_ad_pullback_with_tangents", |ctx, Backend| {
            let mut results = Vec::new();
            let subs = tf_einsum::Subscripts::parse(&self.subscripts).map_err(Error::from)?;
            for k in 0..n {
                let Some(spec) = &self.reverse_specs[k] else {
                    continue;
                };

                let mut rev_inputs_subs = vec![subs.output.clone()];
                let mut rev_operands: Vec<&Tensor<T>> = vec![cotangent];
                let mut rev_tangents: Vec<Option<&Tensor<T>>> = vec![Some(cotangent_tangent)];

                for (i, primal) in primals.iter().enumerate() {
                    if i != k {
                        rev_inputs_subs.push(subs.inputs[i].clone());
                        rev_operands.push(primal);
                        rev_tangents.push(
                            self.reverse_specs[i]
                                .as_ref()
                                .and_then(|s| input_tangents(s.node)),
                        );
                    }
                }

                let rev_subs = tf_einsum::Subscripts {
                    inputs: rev_inputs_subs,
                    output: subs.inputs[k].clone(),
                };

                let grad = tf_einsum::einsum_with_subscripts::<Standard<T>, Backend>(
                    ctx,
                    &rev_subs,
                    &rev_operands,
                    None,
                )
                .map_err(Error::from)?;

                let grad = spec
                    .layout
                    .with_payload_like(compress_pullback_like_in_backend::<Backend, _, T>(
                        ctx,
                        "einsum_ad",
                        grad,
                        &spec.layout,
                    )?)?;

                let rev_subs_str = subscripts_to_string(&rev_subs);
                let grad_tangent = tf_einsum::einsum_frule::<Standard<T>, Backend>(
                    ctx,
                    &rev_subs_str,
                    &rev_operands,
                    &rev_tangents,
                )
                .map_err(Error::from)?;

                let grad_tangent =
                    spec.layout
                        .with_payload_like(compress_pullback_like_in_backend::<Backend, _, T>(
                            ctx,
                            "einsum_ad",
                            grad_tangent,
                            &spec.layout,
                        )?)?;

                results.push((spec.node, grad.into_payload(), grad_tangent.into_payload()));
            }
            Ok(results)
        })
        .map_err(|e| AutodiffError::InvalidArgument(e.to_string()))
    }
}
