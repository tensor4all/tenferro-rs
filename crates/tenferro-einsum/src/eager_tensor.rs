//! EagerTensor einsum extension API.

use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager;
use tenferro_ad::EagerTensor;

use crate::eager::try_build_exact_output_binary_dot_config;
use crate::extension::{
    ensure_einsum_extension_rule_registered, register_runtime, EinsumExtensionOp,
};
use crate::optimize::{default_auto_options, EinsumPlanSpec};
use crate::{parse_einsum_subscripts, EinsumSubscripts, TensorDotAxes};

/// Execute an einsum eagerly on [`EagerTensor`] values.
pub fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
    let subscripts = parse_einsum_subscripts(subscripts)
        .map_err(|err| Error::ContractionError(err.to_string()))?;
    einsum_subscripts(inputs, &subscripts)
}

/// Execute an einsum eagerly from integer labels.
pub fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<EagerTensor> {
    if let Some(result) = try_direct_binary_dot_general(inputs, subscripts) {
        return result;
    }

    ensure_einsum_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))?;
    if let Some(first) = inputs.first() {
        first
            .runtime()
            .register_extension(register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;
    }

    let output_shape_hint = infer_eager_output_shape(subscripts, inputs)?;
    let op = Arc::new(EinsumExtensionOp::with_output_shape_hint(
        subscripts.clone(),
        output_shape_hint,
        EinsumPlanSpec::Auto(default_auto_options()),
    ));
    let mut outputs = apply_eager(op, inputs)?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("einsum extension produced no eager output".to_string()))
}

fn try_direct_binary_dot_general(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Option<Result<EagerTensor>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return None;
    }

    let lhs_labels = &subscripts.inputs[0];
    let rhs_labels = &subscripts.inputs[1];
    if lhs_labels.len() != inputs[0].data().shape().len()
        || rhs_labels.len() != inputs[1].data().shape().len()
    {
        return None;
    }

    if let Some(config) =
        try_build_exact_output_binary_dot_config(lhs_labels, rhs_labels, &subscripts.output)
    {
        return Some(inputs[0].dot_general(inputs[1], config));
    }

    try_build_exact_output_binary_dot_config(rhs_labels, lhs_labels, &subscripts.output)
        .map(|config| inputs[1].dot_general(inputs[0], config))
}

fn infer_eager_output_shape(
    subscripts: &EinsumSubscripts,
    inputs: &[&EagerTensor],
) -> Result<Vec<tenferro_runtime::SymDim>> {
    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "einsum requires at least one input tensor".into(),
        ));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(Error::ContractionError(format!(
            "einsum subscripts expect {} inputs, got {}",
            subscripts.inputs.len(),
            inputs.len()
        )));
    }

    let mut label_dims = std::collections::HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs.iter()) {
        let shape = tensor.data().shape();
        if labels.len() != shape.len() {
            return Err(Error::ContractionError(format!(
                "einsum input rank mismatch: labels={}, shape={}",
                labels.len(),
                shape.len()
            )));
        }
        for (&label, &dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(Error::ContractionError(format!(
                        "einsum label {label} has inconsistent dimensions {existing} and {dim}"
                    )));
                }
            }
        }
    }

    subscripts
        .output
        .iter()
        .map(|label| {
            label_dims
                .get(label)
                .copied()
                .map(tenferro_runtime::SymDim::from)
                .ok_or_else(|| {
                    Error::ContractionError(format!(
                        "einsum output label {label} is missing from input labels"
                    ))
                })
        })
        .collect()
}

/// Execute a NumPy-style tensor contraction on [`EagerTensor`] values.
///
/// This helper lives in the einsum extension namespace because it is
/// contraction sugar over `dot_general`, not a linear algebra facade.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Tensor;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_einsum::{eager_tensor, TensorDotAxes};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
/// let lhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]),
///     ctx.clone(),
/// );
/// let rhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]),
///     ctx,
/// );
/// let out = eager_tensor::tensordot(&lhs, &rhs, TensorDotAxes::Count(1)).unwrap();
///
/// assert_eq!(out.data().shape(), &[2, 4]);
/// ```
pub fn tensordot(
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> Result<EagerTensor> {
    let config = crate::tensordot::dot_general_config(
        axes,
        lhs.data().shape().len(),
        rhs.data().shape().len(),
    )?;
    crate::tensordot::validate_concrete_contract_dims(
        lhs.data().shape(),
        rhs.data().shape(),
        &config,
    )?;
    lhs.dot_general(rhs, config)
}

#[cfg(test)]
mod tests {
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::Tensor;

    use super::einsum;

    #[test]
    fn binary_einsum_col_major_matmul_uses_direct_dot_general_fast_path() {
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
        let lhs = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]),
            ctx.clone(),
        );
        let rhs = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]),
            ctx.clone(),
        );

        let out = einsum(&[&lhs, &rhs], "ji,kj->ki").unwrap();

        assert_eq!(out.data().shape(), &[4, 3]);
        assert_eq!(out.data().as_slice::<f64>().unwrap(), &[2.0_f64; 12]);
        assert_eq!(ctx.cache_stats().extensions.entries, 0);
    }
}
