use tenferro_ops::input_key::TensorInputKey;

use crate::compiler::CompilerOptions;
use crate::program::{FrozenProgram, ProgramBindings, SemanticProgram};

/// A backend-neutral compiled semantic graph with runtime-private execution staging.
///
/// Public consumers inspect only the immutable semantic program and its
/// process-local tensor bindings. Native execution staging remains owned by
/// `tenferro-runtime` and is removed in Phase 5.
#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) frozen: FrozenProgram,
    pub(crate) compiler_options: CompilerOptions,
    pub(crate) input_keys: Box<[TensorInputKey]>,
}

impl CompiledGraph {
    pub(crate) fn new(
        frozen: FrozenProgram,
        compiler_options: CompilerOptions,
        input_keys: impl Into<Box<[TensorInputKey]>>,
    ) -> Self {
        Self {
            frozen,
            compiler_options,
            input_keys: input_keys.into(),
        }
    }

    #[allow(
        dead_code,
        reason = "Phase 5 runtime-owned execution consumes compiled frozen programs"
    )]
    pub(crate) fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }

    /// Borrow the frozen semantic program and its bindings.
    #[doc(hidden)]
    pub fn frozen_program(&self) -> &FrozenProgram {
        &self.frozen
    }

    pub(crate) fn compiler_options(&self) -> CompilerOptions {
        self.compiler_options
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

    /// Borrow ordered trace input keys corresponding to semantic input order.
    #[doc(hidden)]
    pub fn input_keys(&self) -> &[TensorInputKey] {
        &self.input_keys
    }

    /// Return the semantic input index for a trace input key.
    #[doc(hidden)]
    pub fn input_key_index(&self, key: &TensorInputKey) -> Option<usize> {
        self.input_keys
            .iter()
            .position(|candidate| candidate == key)
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
            .field("input_keys", &self.input_keys.len())
            .field("bindings", &self.bindings().len())
            .field(
                "semantic_fingerprint",
                &self.program().semantic_fingerprint(),
            )
            .finish()
    }
}
