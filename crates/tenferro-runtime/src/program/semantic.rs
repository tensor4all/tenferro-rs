use std::fmt;
use std::sync::Arc;

use super::identity::SemanticIdentity;
use super::op::SemanticOperation;
use super::value::ProgramBuilderNonce;
use super::{
    ProgramBindings, ProgramQueryError, ProgramValue, ProgramValueMetadata, SemanticFingerprint,
    SemanticOperationView, ShapeGuard,
};

/// Immutable backend-neutral semantic SSA program.
pub struct SemanticProgram {
    pub(crate) owner: ProgramBuilderNonce,
    pub(crate) inputs: Box<[ProgramValue]>,
    pub(crate) outputs: Box<[ProgramValue]>,
    pub(crate) values: Box<[ProgramValueMetadata]>,
    pub(crate) operations: Box<[SemanticOperation]>,
    pub(crate) shape_guards: Box<[ShapeGuard]>,
    pub(crate) identity: SemanticIdentity,
}

impl SemanticProgram {
    /// Borrow ordered external inputs.
    pub fn inputs(&self) -> &[ProgramValue] {
        &self.inputs
    }

    /// Borrow ordered program outputs.
    pub fn outputs(&self) -> &[ProgramValue] {
        &self.outputs
    }

    /// Iterate over operations in semantic order.
    pub fn operations(&self) -> impl ExactSizeIterator<Item = SemanticOperationView<'_>> + '_ {
        self.operations.iter().map(SemanticOperationView::new)
    }

    /// Borrow metadata for a value owned by this program.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramQueryError::ForeignValue`] for a foreign token.
    pub fn value_metadata(
        &self,
        value: ProgramValue,
    ) -> Result<&ProgramValueMetadata, ProgramQueryError> {
        if value.owner != self.owner {
            return Err(ProgramQueryError::ForeignValue);
        }
        self.values
            .get(value.slot as usize)
            .ok_or(ProgramQueryError::ForeignValue)
    }

    /// Borrow all symbolic guards in semantic operation order.
    pub fn shape_guards(&self) -> &[ShapeGuard] {
        &self.shape_guards
    }

    /// Return the cached normalized structural fingerprint.
    pub const fn semantic_fingerprint(&self) -> SemanticFingerprint {
        self.identity.fingerprint
    }

    /// Compare exact normalized semantics after the compact fingerprint check.
    pub fn semantic_eq(&self, other: &Self) -> bool {
        self.identity.exact_eq(self, &other.identity, other)
    }

    #[cfg(test)]
    pub(crate) fn set_fingerprint_for_test(&mut self, fingerprint: SemanticFingerprint) {
        self.identity.fingerprint = fingerprint;
    }

    #[cfg(test)]
    pub(crate) fn set_first_provenance_for_test(&mut self, label: &str) {
        if let Some(operation) = self.operations.first_mut() {
            operation.provenance = super::metadata::SemanticProvenance::builder(Some(label));
        }
    }

    #[cfg(test)]
    pub(crate) fn fingerprint_computations_for_test(&self) -> usize {
        self.identity.fingerprint_computations
    }
}

impl fmt::Debug for SemanticProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SemanticProgram")
            .field("inputs", &self.inputs.len())
            .field("outputs", &self.outputs.len())
            .field("values", &self.values.len())
            .field("operations", &self.operations.len())
            .field("shape_guards", &self.shape_guards.len())
            .finish()
    }
}

/// One atomically frozen semantic program and its separate tensor bindings.
#[derive(Clone)]
pub struct FrozenProgram {
    /// Immutable backend-neutral semantic structure.
    pub program: Arc<SemanticProgram>,
    /// Process-local tensor defaults and large constants.
    pub bindings: ProgramBindings,
}

impl fmt::Debug for FrozenProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenProgram")
            .field("program", &self.program)
            .field("bindings", &self.bindings)
            .finish()
    }
}
