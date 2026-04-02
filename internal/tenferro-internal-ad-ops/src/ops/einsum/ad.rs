use std::collections::HashMap;
use std::marker::PhantomData;

use super::super::*;
use super::backward::{
    structured_einsum_input_grads_in_backend, structured_einsum_pullback_in_backend,
};
use super::dense_rule::DenseEinsumRule;
use crate::ops::ad::wrap_reverse_edge_output;
use tenferro_prims::TensorTempPoolContext;
use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema, Value};

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

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

#[derive(Clone)]
struct EdgeEinsumSaved<T: EinsumRuntimeValue> {
    subscripts: Subscripts,
    primals: Vec<StructuredTensor<T>>,
}

#[derive(Clone)]
struct EdgeEinsumOp<T: EinsumRuntimeValue> {
    subscripts: Subscripts,
    _marker: PhantomData<T>,
}

impl<T> Op<StructuredTensor<T>> for EdgeEinsumOp<T>
where
    T: EinsumRuntimeValue,
{
    type SavedBackward = EdgeEinsumSaved<T>;
    type SavedJvp = EdgeEinsumSaved<T>;

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let output = dispatch_einsum_runtime!(T, "edge_einsum_primal", |ctx, Backend| {
            einsum_with_subscripts_in_ctx::<Backend, _, T>(ctx, &self.subscripts, inputs)
        })
        .map_err(ad_invalid_argument)?;
        Ok(vec![output])
    }

    fn input_schema(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                };
                inputs.len()
            ],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeEinsumSaved {
            subscripts: self.subscripts.clone(),
            primals: inputs.iter().map(|input| (*input).clone()).collect(),
        })
    }

    fn save_for_jvp(
        &self,
        inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgeEinsumSaved {
            subscripts: self.subscripts.clone(),
            primals: inputs.iter().map(|input| (*input).clone()).collect(),
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(grad_out) = grad_outputs[0].as_ref() else {
            return Ok(vec![None; saved.primals.len()]);
        };
        dispatch_einsum_runtime!(T, "edge_einsum_pullback", |ctx, Backend| {
            structured_einsum_input_grads_in_backend::<Backend, _, T>(
                ctx,
                &saved.subscripts,
                &saved.primals,
                grad_out,
                input_grad_mask,
            )
        })
        .map_err(ad_invalid_argument)
    }

    fn jvp(
        &self,
        saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let tangent = dispatch_einsum_runtime!(T, "edge_einsum_jvp", |ctx, Backend| {
            let primals: Vec<_> = saved.primals.iter().collect();
            let tangents: Vec<_> = tangents.iter().map(|tangent| tangent.as_ref()).collect();
            sum_structured_einsum_tangent_terms::<Backend, _, T>(
                ctx,
                &saved.subscripts,
                &primals,
                &tangents,
            )
        })
        .map_err(ad_invalid_argument)?;
        Ok(vec![tangent])
    }
}

pub(crate) fn can_use_edge_einsum_reverse<T>(operands: &[&AdTensor<T>]) -> bool
where
    T: EinsumRuntimeValue,
{
    if operands
        .iter()
        .any(|operand| operand.structured_tangent().is_some())
    {
        return false;
    }
    let needs_reverse = operands.iter().any(|operand| operand.requires_grad());
    needs_reverse
        && operands
            .iter()
            .all(|operand| !operand.requires_grad() || operand.reverse_edge_value().is_some())
}

pub(crate) fn edge_einsum<T>(subscripts: &str, operands: &[&AdTensor<T>]) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue + 'static,
{
    let subscripts = Subscripts::parse(subscripts).map_err(Error::from)?;
    let op = EdgeEinsumOp::<T> {
        subscripts,
        _marker: PhantomData,
    };

    let edge_inputs: Vec<_> = operands
        .iter()
        .map(|operand| operand.reverse_edge_value())
        .collect();
    let plain_inputs: Vec<_> = operands
        .iter()
        .map(|operand| {
            (!operand.requires_grad()).then(|| Value::new(operand.structured_primal().clone()))
        })
        .collect();
    let values: Vec<&Value<StructuredTensor<T>>> = edge_inputs
        .iter()
        .zip(&plain_inputs)
        .map(|(edge, plain)| match edge.as_ref() {
            Some(value) => Ok(value.as_ref()),
            None => plain.as_ref().ok_or_else(|| crate::Error::InvalidAdTensor {
                message: "einsum operand has no edge or plain value for reverse eager path"
                    .to_string(),
            }),
        })
        .collect::<std::result::Result<_, crate::Error>>()?;

    let output = op.apply_one(&values).map_err(Error::from)?;
    wrap_reverse_edge_output(output)
}

fn run_einsum_ad_in_backend<B, C, T>(
    ctx: &mut C,
    subscripts: &str,
    operands: &[&AdTensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue,
    B: DenseEinsumBackend<T, C>,
    C: TensorTempPoolContext,
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

        let out =
            wrap_same_type_structured_ad_output("einsum_ad", operands, primal_out, tangent_out)?;

        if let Some((node, tape)) = out.reverse_handle() {
            let subscripts = subs.clone();
            let reverse_nodes = collect_reverse_input_nodes(operands);
            let primal_owned: Vec<StructuredTensor<T>> =
                primals.iter().map(|tensor| (*tensor).clone()).collect();

            let input_node_ids: Vec<_> = reverse_nodes.iter().filter_map(|n| *n).collect();
            tape::register_closure_rule::<T>(
                &tape,
                node,
                input_node_ids,
                Box::new(move |cotangent| {
                    dispatch_einsum_runtime!(T, "einsum_ad_pullback_structured", |ctx, Backend| {
                        structured_einsum_pullback_in_backend::<Backend, _, T>(
                            ctx,
                            &subscripts,
                            &reverse_nodes,
                            &primal_owned,
                            cotangent,
                        )
                    })
                }),
            );
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

    let out = wrap_same_type_dense_ad_output("einsum_ad", operands, primal_out, tangent_out)?;

    if let Some((node, tape)) = out.reverse_handle() {
        let subscripts = subscripts.to_string();
        let reverse_specs = collect_reverse_input_specs(operands);

        tape::register_rule::<T>(
            &tape,
            node,
            Box::new(DenseEinsumRule {
                subscripts,
                primals: primal_owned,
                reverse_specs,
            }),
        );
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
    /// ```text
    /// let _builder = builder.size_dict(&size_dict);
    /// ```
    #[allow(dead_code)]
    pub fn size_dict(mut self, size_dict: &'a HashMap<u32, usize>) -> Self {
        self.size_dict = Some(size_dict);
        self
    }

    /// Executes AD einsum with mode propagation.
    /// # Examples
    ///
    /// ```text
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
/// ```text
/// use tenferro::{einsum_ad, set_default_runtime, RuntimeContext, core::AdTensor};
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
