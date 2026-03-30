use super::super::*;
use crate::ops::ad::wrap_reverse_edge_output;
use std::marker::PhantomData;

use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema};

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

struct EdgeSumSaved<T: Scalar> {
    input_layout: StructuredTensor<T>,
}

#[derive(Clone, Copy)]
struct EdgeSumOp<T>(PhantomData<T>);

impl<T> Op<StructuredTensor<T>> for EdgeSumOp<T>
where
    T: ScalarRuntimeValue,
{
    type SavedBackward = EdgeSumSaved<T>;
    type SavedJvp = ();

    fn primal(&self, inputs: &[&StructuredTensor<T>]) -> AdResult<Vec<StructuredTensor<T>>> {
        let output = super::super::scalar::primal::scalar_full_reduction_primal(
            "edge_sum_primal",
            tenferro_prims::ScalarReductionOp::Sum,
            inputs[0].payload(),
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(output),
        )])
    }

    fn input_schema(&self, _inputs: &[&StructuredTensor<T>]) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
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
        Ok(EdgeSumSaved {
            input_layout: inputs[0].clone(),
        })
    }

    fn save_for_jvp(
        &self,
        _inputs: &[&StructuredTensor<T>],
        _outputs: &[StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(())
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let Some(grad_out) = grad_outputs[0].as_ref() else {
            return Ok(vec![None]);
        };
        let scalar = scalar_from_rank0_tensor(grad_out.payload(), "edge_sum_pullback")
            .map_err(ad_invalid_argument)?;
        let payload = broadcast_scalar_like(scalar, saved.input_layout.payload())
            .map_err(ad_invalid_argument)?;
        let grad = saved
            .input_layout
            .with_payload_like(payload)
            .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        _saved: &Self::SavedJvp,
        tangents: &[Option<StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        let output = super::super::scalar::primal::scalar_full_reduction_primal(
            "edge_sum_jvp",
            tenferro_prims::ScalarReductionOp::Sum,
            tangent.payload(),
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![Some(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(output),
        ))])
    }
}

pub(crate) fn can_use_edge_sum_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: ScalarRuntimeValue,
{
    tensor.structured_tangent().is_none()
        && tensor.requires_grad()
        && tensor.reverse_edge_value().is_some()
}

pub(crate) fn edge_sum<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: ScalarRuntimeValue + 'static,
{
    let input = tensor
        .reverse_edge_value()
        .ok_or(Error::UnsupportedAdOp { op: "edge_sum" })?;
    let output = EdgeSumOp::<T>(PhantomData)
        .apply_one(&[input.as_ref()])
        .map_err(Error::from)?;
    wrap_reverse_edge_output(output)
}

/// Builder for AD full reduction / sum.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro::sum_ad(&x).run()?;
/// ```
pub struct SumAdBuilder<'a, T>
where
    T: ScalarRuntimeValue,
{
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SumAdBuilder<'a, T>
where
    T: ScalarRuntimeValue,
{
    /// Executes AD full reduction / sum with mode propagation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = builder.run()?;
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        let operands = [self.tensor];
        let primal = StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(
            super::super::scalar::primal::scalar_full_reduction_primal(
                "sum_ad",
                tenferro_prims::ScalarReductionOp::Sum,
                self.tensor.primal(),
            )?,
        ));
        let tangent = if has_forward(&operands) || has_any_tangent(&operands) {
            let zero_tangent = zero_like(self.tensor.primal())?;
            let tangent_input = if let Some(tangent) = self.tensor.structured_tangent() {
                tangent.payload()
            } else {
                &zero_tangent
            };
            Some(StructuredTensor(
                tenferro_tensor::StructuredTensor::from_dense(
                    super::super::scalar::primal::scalar_full_reduction_primal(
                        "sum_ad",
                        tenferro_prims::ScalarReductionOp::Sum,
                        tangent_input,
                    )?,
                ),
            ))
        } else {
            None
        };

        let out = wrap_same_type_structured_ad_output("sum_ad", &operands, primal, tangent)?;

        if let Some((node, tape)) = out.reverse_handle() {
            let input_node = collect_reverse_input_nodes(&operands)
                .into_iter()
                .next()
                .flatten();
            let input_layout = self.tensor.structured_primal().clone();

            let input_node_ids: Vec<_> = input_node.into_iter().collect();
            tape::register_closure_rule::<T>(
                &tape,
                node,
                input_node_ids,
                Box::new(move |cotangent| {
                    let Some(input_node) = input_node else {
                        return Ok(Vec::new());
                    };
                    let scalar = scalar_from_rank0_tensor(cotangent.payload(), "sum_ad")?;
                    let payload = broadcast_scalar_like(scalar, input_layout.payload())?;
                    let grad = input_layout.with_payload_like(payload)?;
                    Ok(vec![(input_node, grad)])
                }),
            );
        }

        Ok(out)
    }
}

/// Creates a builder for AD full reduction / sum.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro::sum_ad(&x).run()?;
/// ```
pub fn sum_ad<'a, T>(tensor: &'a AdTensor<T>) -> SumAdBuilder<'a, T>
where
    T: ScalarRuntimeValue,
{
    SumAdBuilder { tensor }
}
