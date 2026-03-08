use tenferro_device::Result;

use crate::notation::{char_to_label, split_and_validate_notation};

/// Einsum subscripts using integer labels (omeinsum-rs compatible).
///
/// Each dimension is represented by a `u32` label. Labels shared across
/// multiple input tensors are contracted (summed over). Labels present
/// only in the output are free indices.
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
/// ```ignore
/// use tenferro_einsum::Subscripts;
///
/// // Parse from string notation
/// let subs = Subscripts::parse("ij,jk->ik").unwrap();
/// assert_eq!(subs.inputs.len(), 2);
/// ```
#[derive(Debug, Clone)]
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
    /// Parentheses in the notation are accepted but stripped during parsing.
    /// To respect parenthesized contraction order, use [`NestedEinsum::parse`]
    /// or pass the parenthesized string directly to [`einsum`].
    ///
    /// # Examples
    ///
    /// - `"ij,jk->ik"` — matrix multiplication
    /// - `"ii->i"` — diagonal extraction
    /// - `"ijk->"` — full contraction (scalar result)
    /// - `"ij,(jk,kl)->il"` — contract B and C first, then with A
    ///
    /// # Errors
    ///
    /// Returns an error if the notation is malformed.
    pub fn parse(notation: &str) -> Result<Self> {
        let (inputs_str, output_str) = split_and_validate_notation(notation)?;

        let output: Vec<u32> = output_str
            .chars()
            .map(char_to_label)
            .collect::<Result<_>>()?;

        // Parentheses already validated by split_and_validate_notation() above.
        // Strip parentheses and parse input labels
        let clean_inputs = inputs_str.replace(['(', ')'], "");
        let inputs: Vec<Vec<u32>> = clean_inputs
            .split(',')
            .map(|s| s.chars().map(char_to_label).collect::<Result<_>>())
            .collect::<Result<_>>()?;

        Ok(Self { inputs, output })
    }
}
