use crate::{Error, Result, Subscripts};

/// Canonical N-ary einsum subscripts using integer labels.
///
/// String notation is a user-facing convenience. Runtime integration layers can
/// carry this representation in extension payloads so execution, shape
/// inference, and AD do not need to parse strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinsumSubscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl EinsumSubscripts {
    /// Create subscripts from integer label arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::EinsumSubscripts;
    ///
    /// let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    ///
    /// assert_eq!(subscripts.inputs, vec![vec![0, 1], vec![1, 2]]);
    /// assert_eq!(subscripts.output, vec![0, 2]);
    /// ```
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self {
        Self {
            inputs: inputs.iter().map(|labels| labels.to_vec()).collect(),
            output: output.to_vec(),
        }
    }

    /// Number of input operands described by this specification.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::EinsumSubscripts;
    ///
    /// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
    ///
    /// assert_eq!(subscripts.n_inputs(), 2);
    /// ```
    #[must_use]
    pub fn n_inputs(&self) -> usize {
        self.inputs.len()
    }
}

impl From<Subscripts> for EinsumSubscripts {
    fn from(subscripts: Subscripts) -> Self {
        Self {
            inputs: subscripts.inputs,
            output: subscripts.output,
        }
    }
}

impl From<&Subscripts> for EinsumSubscripts {
    fn from(subscripts: &Subscripts) -> Self {
        Self {
            inputs: subscripts.inputs.clone(),
            output: subscripts.output.clone(),
        }
    }
}

impl From<EinsumSubscripts> for Subscripts {
    fn from(subscripts: EinsumSubscripts) -> Self {
        Self {
            inputs: subscripts.inputs,
            output: subscripts.output,
        }
    }
}

impl From<&EinsumSubscripts> for Subscripts {
    fn from(subscripts: &EinsumSubscripts) -> Self {
        Self {
            inputs: subscripts.inputs.clone(),
            output: subscripts.output.clone(),
        }
    }
}

/// Parse string einsum notation into canonical integer labels.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::parse_einsum_subscripts;
///
/// let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
///
/// assert_eq!(subscripts.inputs.len(), 2);
/// assert_eq!(subscripts.output, vec![b'i' as u32, b'k' as u32]);
/// ```
///
/// # Errors
///
/// Returns an error if the notation is malformed.
pub fn parse_einsum_subscripts(notation: &str) -> Result<EinsumSubscripts> {
    Subscripts::parse(notation)
        .map(EinsumSubscripts::from)
        .map_err(|err| Error::InvalidArgument(format!("invalid einsum subscripts: {err}")))
}

#[cfg(test)]
mod tests;
