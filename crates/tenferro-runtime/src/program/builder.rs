use super::value::ProgramBuilderNonce;
use super::{ProgramBuildError, ProgramInputSpec, ProgramValue};

/// Mutable validation boundary for one semantic program.
pub struct SemanticProgramBuilder {
    owner: ProgramBuilderNonce,
    input_specs: Vec<ProgramInputSpec>,
}

impl Default for SemanticProgramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticProgramBuilder {
    /// Construct an empty builder with a fresh opaque identity.
    pub fn new() -> Self {
        Self {
            owner: ProgramBuilderNonce::fresh(),
            input_specs: Vec::new(),
        }
    }

    /// Add one ordered external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] if the builder cannot
    /// represent another value slot.
    pub fn input(&mut self, spec: ProgramInputSpec) -> Result<ProgramValue, ProgramBuildError> {
        let slot =
            u32::try_from(self.input_specs.len()).map_err(|_| ProgramBuildError::TooManyValues)?;
        self.input_specs.push(spec);
        Ok(ProgramValue::new(slot, self.owner))
    }

    /// Validate that a value belongs to this builder.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a token from another
    /// builder or one that does not name an existing value.
    pub fn validate_value(&self, value: ProgramValue) -> Result<(), ProgramBuildError> {
        if value.owner != self.owner || value.slot as usize >= self.input_specs.len() {
            return Err(ProgramBuildError::ForeignValue);
        }
        Ok(())
    }
}
