use tenferro_einsum::Subscripts;
use tenferro_ops::std_tensor_op::EinsumSubscripts;

use crate::error::{Error, Result};

/// Parse string einsum notation into tenferro's canonical integer labels.
///
/// # Examples
///
/// ```
/// use tenferro::parse_einsum_subscripts;
///
/// let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
///
/// assert_eq!(subscripts.inputs.len(), 2);
/// assert_eq!(subscripts.output, vec![b'i' as u32, b'k' as u32]);
/// ```
pub fn parse_einsum_subscripts(notation: &str) -> Result<EinsumSubscripts> {
    let parsed =
        Subscripts::parse(notation).map_err(|err| Error::InvalidSubscripts(format!("{err}")))?;
    Ok(EinsumSubscripts {
        inputs: parsed.inputs,
        output: parsed.output,
    })
}

pub(crate) fn to_einsum_subscripts(subscripts: &EinsumSubscripts) -> Subscripts {
    Subscripts {
        inputs: subscripts.inputs.clone(),
        output: subscripts.output.clone(),
    }
}
