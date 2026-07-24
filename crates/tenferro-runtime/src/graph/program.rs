use crate::exec::ExecProgram;
use crate::program::{FrozenProgram, ProgramBindings, SemanticProgram};

/// A backend-neutral compiled semantic graph with runtime-private execution staging.
///
/// Public consumers inspect only the immutable semantic program and its
/// process-local tensor bindings. Native execution staging remains owned by
/// `tenferro-runtime` and is removed in Phase 5.
#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) staging: ExecProgram,
    pub(crate) frozen: FrozenProgram,
}

impl CompiledGraph {
    pub(crate) fn new(frozen: FrozenProgram, staging: ExecProgram) -> Self {
        Self { staging, frozen }
    }

    /// Borrow the immutable backend-neutral semantic program.
    pub fn program(&self) -> &SemanticProgram {
        &self.frozen.program
    }

    /// Borrow tensor defaults and large constants kept outside semantic structure.
    pub fn bindings(&self) -> &ProgramBindings {
        &self.frozen.bindings
    }

    /// Return the number of ordered semantic inputs.
    pub fn input_count(&self) -> usize {
        self.frozen.program.inputs().len()
    }

    /// Return the number of ordered semantic outputs.
    pub fn output_count(&self) -> usize {
        self.frozen.program.outputs().len()
    }
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompiledGraph")
            .field("inputs", &self.input_count())
            .field("outputs", &self.output_count())
            .field("bindings", &self.bindings().len())
            .field(
                "semantic_fingerprint",
                &self.program().semantic_fingerprint(),
            )
            .finish()
    }
}
