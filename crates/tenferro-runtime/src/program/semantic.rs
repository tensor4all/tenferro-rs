use std::fmt;
use std::sync::Arc;

use super::op::SemanticOperation;
use super::value::ProgramBuilderNonce;
use super::{
    ProgramBindings, ProgramQueryError, ProgramValue, ProgramValueMetadata, SemanticOperationView,
    ShapeGuard,
};

/// Immutable backend-neutral semantic SSA program.
pub struct SemanticProgram {
    pub(crate) owner: ProgramBuilderNonce,
    pub(crate) inputs: Box<[ProgramValue]>,
    pub(crate) outputs: Box<[ProgramValue]>,
    pub(crate) values: Box<[ProgramValueMetadata]>,
    pub(crate) operations: Box<[SemanticOperation]>,
    pub(crate) shape_guards: Box<[ShapeGuard]>,
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
    pub program: Arc<SemanticProgram>,
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
