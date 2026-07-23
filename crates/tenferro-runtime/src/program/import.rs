use std::fmt;

use super::{ProgramBindings, ProgramValue, SemanticProgram};

/// Read-only request to import the dependency closure of ordered roots.
pub struct ProgramImport<'a> {
    pub program: &'a SemanticProgram,
    pub bindings: &'a ProgramBindings,
    pub roots: &'a [ProgramValue],
}

impl fmt::Debug for ProgramImport<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProgramImport")
            .field("program_inputs", &self.program.inputs().len())
            .field("program_outputs", &self.program.outputs().len())
            .field("bindings", &self.bindings.len())
            .field("roots", &self.roots.len())
            .finish()
    }
}

/// Destination-builder values corresponding to requested import roots.
pub struct ImportedProgramValues {
    roots: Box<[ProgramValue]>,
}

impl ImportedProgramValues {
    pub(crate) fn new(roots: Box<[ProgramValue]>) -> Self {
        Self { roots }
    }

    /// Borrow imported roots in request order, including duplicates.
    pub fn roots(&self) -> &[ProgramValue] {
        &self.roots
    }
}

impl fmt::Debug for ImportedProgramValues {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImportedProgramValues")
            .field("roots", &self.roots.len())
            .finish()
    }
}
