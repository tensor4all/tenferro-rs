use std::fmt;
use std::sync::Arc;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_tensor::Tensor;

use crate::extension_cache::ExtensionCacheStore;
use crate::program::{
    BindingKey, CoreSemanticOp, FrozenProgram, ProgramBindings, ProgramBuildError,
    ProgramFinishError, ProgramInputSpec, ProgramValue, ProgramValueMetadata, SemanticProgram,
    SemanticProgramBuilder,
};

/// Mutable owner of one backend-neutral semantic trace.
///
/// Values issued by one context cannot be used by another. Consuming
/// [`finish`](Self::finish) freezes the trace into an immutable
/// [`TracedGraph`].
pub struct TraceContext {
    builder: SemanticProgramBuilder,
    extension_caches: ExtensionCacheStore,
}

impl TraceContext {
    /// Construct an empty trace context.
    pub fn new() -> Self {
        Self {
            builder: SemanticProgramBuilder::new(),
            extension_caches: ExtensionCacheStore::new(),
        }
    }

    /// Add one ordered external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] if another value cannot be
    /// represented.
    pub fn input(&mut self, spec: ProgramInputSpec) -> Result<TraceValue, ProgramBuildError> {
        let value = self.builder.input(spec)?;
        Ok(TraceValue { value })
    }

    /// Add one ordered input and attach its default tensor.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] when another input cannot be
    /// represented, [`ProgramBuildError::BindingTargetNotInput`] if the new
    /// value is not accepted as an input, or
    /// [`ProgramBuildError::DuplicateBinding`] if it is already bound.
    pub fn input_with_default(
        &mut self,
        spec: ProgramInputSpec,
        tensor: Arc<Tensor>,
    ) -> Result<TraceValue, ProgramBuildError> {
        let input = self.input(spec)?;
        self.bind_input(input, tensor)?;
        Ok(input)
    }

    /// Attach a tensor default to an existing trace input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for another context's
    /// value, [`ProgramBuildError::BindingTargetNotInput`] for a computed
    /// value, or [`ProgramBuildError::DuplicateBinding`] for a repeated
    /// binding.
    pub fn bind_input(
        &mut self,
        input: TraceValue,
        tensor: Arc<Tensor>,
    ) -> Result<BindingKey, ProgramBuildError> {
        self.builder.bind_input(input.value, tensor)
    }

    /// Add one canonical core semantic operation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for an input from another
    /// context, [`ProgramBuildError::Arity`] for the wrong input count, or
    /// [`ProgramBuildError::Metadata`] /
    /// [`ProgramBuildError::OutputMetadataCount`] when inference returns
    /// invalid metadata.
    pub fn add_op(
        &mut self,
        op: CoreSemanticOp,
        inputs: &[TraceValue],
    ) -> Result<Box<[TraceValue]>, ProgramBuildError> {
        let inputs = program_values(inputs);
        self.builder.add_op(op, &inputs).map(trace_values)
    }

    /// Add one extension semantic operation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] or
    /// [`ProgramBuildError::Arity`] for invalid inputs,
    /// [`ProgramBuildError::UndeclaredExtensionEffects`] /
    /// [`ProgramBuildError::UndeclaredExtensionAliases`] for an incomplete
    /// extension contract, or [`ProgramBuildError::Metadata`] when inference
    /// fails.
    pub fn add_extension(
        &mut self,
        op: Arc<dyn ExtensionOp>,
        inputs: &[TraceValue],
    ) -> Result<Box<[TraceValue]>, ProgramBuildError> {
        let inputs = program_values(inputs);
        self.builder.add_extension(op, &inputs).map(trace_values)
    }

    /// Borrow metadata for a context-owned value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a value from another
    /// context.
    pub fn value_metadata(
        &self,
        value: TraceValue,
    ) -> Result<&ProgramValueMetadata, ProgramBuildError> {
        self.builder.value_metadata(value.value)
    }

    /// Borrow the generic trace-time extension cache store.
    pub fn extension_caches_mut(&mut self) -> &mut ExtensionCacheStore {
        &mut self.extension_caches
    }

    /// Consume and atomically freeze this context.
    ///
    /// Ordered and duplicate outputs are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramFinishError::ForeignOutput`] if an output belongs to
    /// another context, or a typed structural/binding finalization failure.
    pub fn finish(self, outputs: &[TraceValue]) -> Result<TracedGraph, ProgramFinishError> {
        let outputs = program_values(outputs);
        let frozen = self.builder.finish(&outputs)?;
        Ok(TracedGraph { frozen })
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Opaque value handle owned by one [`TraceContext`].
#[derive(Clone, Copy)]
pub struct TraceValue {
    value: ProgramValue,
}

impl fmt::Debug for TraceValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("TraceValue(<opaque>)")
    }
}

/// Immutable backend-neutral output of one completed trace.
#[derive(Clone)]
pub struct TracedGraph {
    frozen: FrozenProgram,
}

impl TracedGraph {
    /// Borrow the immutable semantic program.
    pub fn program(&self) -> &SemanticProgram {
        &self.frozen.program
    }

    /// Borrow process-local tensor defaults and large constants.
    pub fn bindings(&self) -> &ProgramBindings {
        &self.frozen.bindings
    }

    pub(crate) fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }
}

impl fmt::Debug for TracedGraph {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TracedGraph")
            .field("inputs", &self.program().inputs().len())
            .field("outputs", &self.program().outputs().len())
            .field("bindings", &self.bindings().len())
            .field(
                "semantic_fingerprint",
                &self.program().semantic_fingerprint(),
            )
            .finish()
    }
}

fn program_values(values: &[TraceValue]) -> Vec<ProgramValue> {
    values.iter().map(|value| value.value).collect()
}

fn trace_values(values: Box<[ProgramValue]>) -> Box<[TraceValue]> {
    values
        .into_vec()
        .into_iter()
        .map(|value| TraceValue { value })
        .collect()
}

#[cfg(test)]
mod tests {
    use tenferro_ops::ShapeExtent;
    use tenferro_tensor::DType;

    use super::*;

    #[test]
    fn unknown_input_extents_keep_ordered_semantic_inputs() {
        let spec = || {
            ProgramInputSpec::from_metadata(ProgramValueMetadata::from_extents(
                DType::F64,
                [ShapeExtent::Unknown],
            ))
        };
        let mut context = TraceContext::new();
        let first = context.input(spec()).unwrap();
        let second = context.input(spec()).unwrap();
        let graph = context.finish(&[first, second]).unwrap();

        assert_eq!(graph.program().inputs().len(), 2);
        for input in graph.program().inputs() {
            assert_eq!(
                graph.program().value_metadata(*input).unwrap().shape(),
                [ShapeExtent::Unknown]
            );
        }
    }
}
