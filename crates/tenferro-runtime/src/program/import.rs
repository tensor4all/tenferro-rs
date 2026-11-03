use std::fmt;

use super::{ProgramBindings, ProgramValue, SemanticProgram};

/// Read-only request to import the dependency closure of ordered roots.
pub struct ProgramImport<'a> {
    /// Immutable source semantic program.
    pub program: &'a SemanticProgram,
    /// Tensor bindings frozen with the source program, when structure and
    /// bindings should travel together.
    ///
    /// A structural import passes `None` so symbolic extents stay symbolic; a
    /// caller that wants the frozen defaults re-attaches them afterwards.
    pub bindings: Option<&'a ProgramBindings>,
    /// Ordered source roots to remap; empty and duplicate roots are valid.
    pub roots: &'a [ProgramValue],
    /// Source inputs that already exist in the destination.
    ///
    /// A mapped source input reuses the destination value (and its metadata)
    /// instead of creating a fresh input, which is how a derivative program
    /// lands back on the leaves it was differentiated from.
    pub input_map: &'a [(ProgramValue, ProgramValue)],
}

impl fmt::Debug for ProgramImport<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProgramImport")
            .field("program_inputs", &self.program.inputs().len())
            .field("program_outputs", &self.program.outputs().len())
            .field("bindings", &self.bindings.map(ProgramBindings::len))
            .field("roots", &self.roots.len())
            .field("input_map", &self.input_map.len())
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
