use crate::compiler::CompilerOptions;
use crate::program::{FrozenProgram, ProgramBindings, SemanticProgram};
#[cfg(test)]
use crate::shape_constraint::ShapeGuard;

/// A backend-neutral compiled semantic graph with runtime-private execution staging.
///
/// Public consumers inspect only the immutable semantic program and its
/// process-local tensor bindings. Native execution staging remains owned by
/// `tenferro-runtime` and is removed in Phase 5.
#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) frozen: FrozenProgram,
    pub(crate) compiler_options: CompilerOptions,
    #[cfg(test)]
    pub(crate) test_shape_guards: Vec<ShapeGuard>,
}

impl CompiledGraph {
    pub(crate) fn new(frozen: FrozenProgram, compiler_options: CompilerOptions) -> Self {
        Self {
            frozen,
            compiler_options,
            #[cfg(test)]
            test_shape_guards: Vec::new(),
        }
    }

    #[allow(
        dead_code,
        reason = "Phase 5 runtime-owned execution consumes compiled frozen programs"
    )]
    pub(crate) fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }

    pub(crate) fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
    }

    #[cfg(test)]
    pub(crate) fn set_test_shape_guards(&mut self, guards: Vec<ShapeGuard>) {
        self.test_shape_guards = guards;
    }

    #[cfg(test)]
    pub(crate) fn test_shape_guards(&self) -> &[ShapeGuard] {
        &self.test_shape_guards
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
