//! EagerTensor einsum extension API.

use std::sync::Arc;

use tenferro::error::{Error, Result};
use tenferro::extension::apply_eager;
use tenferro::EagerTensor;

use crate::extension::{ensure_einsum_extension_rule_registered, EinsumExtensionOp};
use crate::{parse_einsum_subscripts, EinsumSubscripts};

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

    let op = Arc::new(EinsumExtensionOp::new(subscripts.clone()));
    let mut outputs = apply_eager(op, inputs)?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("einsum extension produced no eager output".to_string()))
}
