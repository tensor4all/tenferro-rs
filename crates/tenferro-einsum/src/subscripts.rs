use crate::{Result, Subscripts};

/// One unresolved axis token in rank-polymorphic einsum notation.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::EinsumAxis;
/// assert_eq!(EinsumAxis::Ellipsis, EinsumAxis::Ellipsis);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EinsumAxis {
    /// An explicit integer label.
    Label(u32),
    /// A NumPy-style ellipsis whose rank is resolved from the inputs.
    Ellipsis,
}

/// Rank-unresolved einsum notation.
///
/// This is the programmatic counterpart of string notation such as
/// `"...ij,...jk->...ik"`. Resolve it through an einsum operation, after input
/// ranks are known; [`EinsumSubscripts`] remains the rank-resolved runtime form.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{EinsumAxis, EinsumNotation};
/// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
/// assert_eq!(notation.input_count(), 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinsumNotation {
    /// Axis tokens for each input tensor.
    pub inputs: Vec<Vec<EinsumAxis>>,
    /// Axis tokens for the output tensor.
    pub output: Vec<EinsumAxis>,
}

impl EinsumNotation {
    /// Create rank-unresolved notation from axis-token arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    ///
    /// let notation = EinsumNotation::new(
    ///     &[&[EinsumAxis::Ellipsis, EinsumAxis::Label(0)], &[EinsumAxis::Ellipsis, EinsumAxis::Label(0)]],
    ///     &[EinsumAxis::Ellipsis],
    /// );
    /// assert_eq!(notation.input_count(), 2);
    /// ```
    pub fn new(inputs: &[&[EinsumAxis]], output: &[EinsumAxis]) -> Self {
        Self {
            inputs: inputs.iter().map(|axes| axes.to_vec()).collect(),
            output: output.to_vec(),
        }
    }

    /// Number of input operands described by this notation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumAxis, EinsumNotation};
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    /// assert_eq!(notation.input_count(), 1);
    /// ```
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Parse flat string notation, retaining ellipsis tokens.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidSubscripts`] for malformed separators,
    /// labels, ellipses, or parenthesized contraction order.
    pub fn parse(notation: &str) -> Result<Self> {
        let (inputs_str, output_str) =
            crate::syntax::notation::split_and_validate_notation(notation)?;
        if inputs_str.contains(['(', ')']) || output_str.contains(['(', ')']) {
            return Err(crate::Error::invalid_subscripts(
                "EinsumNotation::parse does not accept parentheses; use NestedEinsum::parse for parenthesized contraction order",
            ));
        }
        let inputs = inputs_str
            .split(',')
            .map(parse_axis_term)
            .collect::<Result<Vec<_>>>()?;
        let output = parse_axis_term(output_str)?;
        Ok(Self { inputs, output })
    }
}

fn parse_axis_term(term: &str) -> Result<Vec<EinsumAxis>> {
    let mut chars = term.chars();
    let mut axes = Vec::new();
    while let Some(c) = chars.next() {
        if c == '.' {
            if chars.next() != Some('.') || chars.next() != Some('.') {
                return Err(crate::Error::invalid_subscripts(
                    "einsum ellipsis must be written as exactly three dots",
                ));
            }
            if axes.contains(&EinsumAxis::Ellipsis) {
                return Err(crate::Error::invalid_subscripts(
                    "each einsum term may contain at most one ellipsis",
                ));
            }
            axes.push(EinsumAxis::Ellipsis);
        } else {
            axes.push(EinsumAxis::Label(crate::syntax::notation::char_to_label(
                c,
            )?));
        }
    }
    Ok(axes)
}

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
    /// assert_eq!(subscripts.input_count(), 2);
    /// ```
    #[must_use]
    pub fn input_count(&self) -> usize {
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
/// Returns [`crate::Error::InvalidSubscripts`] when the notation is malformed,
/// contains an invalid label, or has an invalid input/output separator.
pub fn parse_einsum_subscripts(notation: &str) -> Result<EinsumSubscripts> {
    Subscripts::parse(notation).map(EinsumSubscripts::from)
}

/// Parse string notation into rank-unresolved axis tokens.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{parse_einsum_notation, EinsumAxis};
///
/// let notation = parse_einsum_notation("...ij,...jk->...ik").unwrap();
/// assert_eq!(notation.inputs[0][0], EinsumAxis::Ellipsis);
/// ```
///
/// # Errors
///
/// Returns [`crate::Error::InvalidSubscripts`] for malformed notation,
/// multiple ellipses in one term, or parenthesized contraction order.
pub fn parse_einsum_notation(notation: &str) -> Result<EinsumNotation> {
    EinsumNotation::parse(notation)
}

#[cfg(test)]
mod tests;
