use crate::syntax::notation::{char_to_label, split_and_validate_notation};
use crate::{Error, Result};

/// Einsum subscripts using integer labels (omeinsum-rs compatible).
///
/// Each dimension is represented by a `u32` label. Labels shared across
/// multiple input tensors are contracted (summed over). Repeated labels within
/// one input select a diagonal before any reduction; if the repeated label is
/// absent from the output, the diagonal is reduced. Repeated labels in the
/// output embed the input on a diagonal.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// // Matrix multiplication: C_{ik} = Σ_j A_{ij} * B_{jk}
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// assert_eq!(subs.inputs.len(), 2);
/// assert_eq!(subs.output, vec![0, 2]);
/// ```
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// // Parse from string notation
/// let subs = Subscripts::parse("ij,jk->ik").unwrap();
/// assert_eq!(subs.inputs.len(), 2);
/// ```
///
/// ```
/// use tenferro_einsum::Subscripts;
///
/// let trace = Subscripts::parse("ii->").unwrap();
/// let diagonal = Subscripts::parse("ii->i").unwrap();
/// let embed = Subscripts::parse("i->ii").unwrap();
/// let higher_rank = Subscripts::parse("iij->ij").unwrap();
///
/// assert!(trace.output.is_empty());
/// assert_eq!(diagonal.output, vec![b'i' as u32]);
/// assert_eq!(embed.output, vec![b'i' as u32, b'i' as u32]);
/// assert_eq!(higher_rank.inputs[0], vec![b'i' as u32, b'i' as u32, b'j' as u32]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Subscripts {
    /// Index labels for each input tensor.
    pub inputs: Vec<Vec<u32>>,
    /// Index labels for the output tensor.
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create subscripts from integer label arrays.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Index labels for each input tensor
    /// * `output` — Index labels for the output tensor
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self {
        Self {
            inputs: inputs.iter().map(|s| s.to_vec()).collect(),
            output: output.to_vec(),
        }
    }

    /// Parse subscripts from NumPy/PyTorch-style string notation.
    ///
    /// Each Unicode alphanumeric character represents a dimension label.
    /// Labels are mapped to integer IDs via Unicode scalar values (`char as u32`).
    /// Input tensors are separated by commas, and `->` separates inputs
    /// from the output.
    ///
    /// Parentheses are rejected by this flat parser. Use
    /// [`crate::NestedEinsum::parse`] when notation specifies a parenthesized
    /// contraction order.
    ///
    /// # Examples
    ///
    /// - `"ij,jk->ik"` — matrix multiplication
    /// - `"ii->"` — diagonal extraction followed by reduction (trace)
    /// - `"ii->i"` — diagonal extraction
    /// - `"i->ii"` — diagonal embedding
    /// - `"iij->ij"` — higher-rank diagonal extraction
    /// - `"ijk->"` — full contraction (scalar result)
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] if the notation is malformed or
    /// contains parenthesized contraction order.
    pub fn parse(notation: &str) -> Result<Self> {
        let (inputs_str, output_str) = split_and_validate_notation(notation)?;
        if inputs_str.contains(['(', ')']) {
            return Err(Error::invalid_subscripts(
                "Subscripts::parse does not accept parentheses; use NestedEinsum::parse to preserve parenthesized contraction order",
            ));
        }

        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

        let inputs: Vec<Vec<u32>> = inputs_str
            .split(',')
            .map(|s| s.chars().map(char_to_label).collect::<Result<_>>())
            .collect::<Result<_>>()?;

        Ok(Self { inputs, output })
    }
}

#[cfg(test)]
mod tests;
