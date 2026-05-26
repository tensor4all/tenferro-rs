//! EagerTensor einsum extension API.

use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager;
use tenferro_ad::EagerTensor;

use crate::extension::{
    ensure_einsum_extension_rule_registered, register_runtime, EinsumExtensionOp,
};
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
    ensure_einsum_extension_rule_registered().map_err(|err| Error::Internal(err.to_string()))?;
    if let Some(first) = inputs.first() {
        first
            .runtime()
            .register_extension(register_runtime)
            .map_err(|err| Error::Internal(err.to_string()))?;
    }

    let op = Arc::new(EinsumExtensionOp::new(subscripts.clone()));
    let mut outputs = apply_eager(op, inputs)?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("einsum extension produced no eager output".to_string()))
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
/// use tenferro_ad::{CpuBackend, EagerRuntime, EagerTensor};
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
