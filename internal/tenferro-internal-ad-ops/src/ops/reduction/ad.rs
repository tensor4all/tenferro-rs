use super::super::*;

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
