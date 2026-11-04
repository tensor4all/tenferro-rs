use std::cell::{RefCell, RefMut};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_tensor::Tensor;

use crate::checkpoint::RetainedValue;
use crate::extension_cache::ExtensionCacheStore;
use crate::program::{
    BindingKey, CoreSemanticOp, FrozenProgram, FrozenProgram as SnapshotProgram, ProgramBindings,
    ProgramBuildError, ProgramFinishError, ProgramInputSpec, ProgramValue, ProgramValueMetadata,
    SemanticProgram, SemanticProgramBuilder,
};

/// One checkpointed value: a bound input whose pre-checkpoint definition stays
/// available to the AD snapshot.
pub(crate) struct CheckpointAlias {
    pub(crate) input: ProgramValue,
    pub(crate) target: ProgramValue,
    pub(crate) bound_shape: Vec<usize>,
}

struct TraceInner {
    builder: SemanticProgramBuilder,
    extension_caches: ExtensionCacheStore,
    /// Program inputs in creation order; the index is the program input index.
    inputs: Vec<TraceInput>,
    aliases: Vec<CheckpointAlias>,
}

struct TraceInput {
    value: ProgramValue,
    key: TensorInputKey,
}

impl TraceInner {
    fn new() -> Self {
        Self {
            builder: SemanticProgramBuilder::new(),
            extension_caches: ExtensionCacheStore::new(),
            inputs: Vec::new(),
            aliases: Vec::new(),
        }
    }
}

/// Shared, cloneable handle to one mutable semantic trace.
///
/// Cloning the handle does not copy the trace. Values issued by one context
/// cannot be used by another, and the handle is deliberately not `Send`: the
/// trace is a single-threaded construction value, while eager execution keeps a
/// frozen program instead of a live handle.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, TraceContext};
///
/// let trace = TraceContext::new();
/// let x = trace.from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let y = x.reshape([2])?;
///
/// // A clone is the same trace; a second context is a different one.
/// let same = trace.clone();
/// assert!(same.same_context(&trace));
/// assert!(!TraceContext::new().same_context(&trace));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TraceContext {
    inner: Rc<RefCell<TraceInner>>,
}

impl Clone for TraceContext {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for TraceContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TraceContext")
            .field("values", &self.inner.borrow().builder.operation_count())
            .finish_non_exhaustive()
    }
}

impl TraceContext {
    /// Construct an empty trace context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_runtime::TraceContext;
    ///
    /// let mut trace = TraceContext::new();
    /// assert!(trace.finish(&[]).is_ok());
    /// ```
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(TraceInner::new())),
        }
    }

    fn borrow_mut(&self) -> Result<RefMut<'_, TraceInner>, ProgramBuildError> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| ProgramBuildError::ReentrantContextBorrow)
    }

    fn borrow_mut_for_finish(&self) -> Result<RefMut<'_, TraceInner>, ProgramFinishError> {
        self.inner
            .try_borrow_mut()
            .map_err(|_| ProgramFinishError::ReentrantContextBorrow)
    }

    /// Add one ordered external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] if another value cannot be
    /// represented, or [`ProgramBuildError::ReentrantContextBorrow`] when the
    /// context is already borrowed by an operation in progress.
    pub fn input(&self, spec: ProgramInputSpec) -> Result<TraceValue, ProgramBuildError> {
        let key = crate::traced::next_input_key();
        let mut inner = self.borrow_mut()?;
        let value = inner.builder.input(spec)?;
        inner.inputs.push(TraceInput { value, key });
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
        &self,
        spec: ProgramInputSpec,
        tensor: Tensor,
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
        &self,
        input: TraceValue,
        tensor: Tensor,
    ) -> Result<BindingKey, ProgramBuildError> {
        self.borrow_mut()?.builder.bind_input(input.value, tensor)
    }

    /// Add one canonical core semantic operation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for an input from another
    /// context, [`ProgramBuildError::Arity`] for the wrong input count,
    /// [`ProgramBuildError::Metadata`] /
    /// [`ProgramBuildError::OutputMetadataCount`] when inference returns
    /// invalid metadata, or [`ProgramBuildError::ReentrantContextBorrow`] when
    /// the context is already borrowed.
    pub fn add_op(
        &self,
        op: CoreSemanticOp,
        inputs: &[TraceValue],
    ) -> Result<Box<[TraceValue]>, ProgramBuildError> {
        let inputs = program_values(inputs);
        let outputs = self.borrow_mut()?.builder.add_op(op, &inputs)?;
        Ok(trace_values(outputs))
    }

    /// Add one extension semantic operation.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] or
    /// [`ProgramBuildError::Arity`] for invalid inputs,
    /// [`ProgramBuildError::UndeclaredExtensionEffects`] /
    /// [`ProgramBuildError::UndeclaredExtensionAliases`] for an incomplete
    /// extension contract, [`ProgramBuildError::Metadata`] when inference
    /// fails, or [`ProgramBuildError::ReentrantContextBorrow`] when the context
    /// is already borrowed.
    pub fn add_extension(
        &self,
        op: Arc<dyn ExtensionOp>,
        inputs: &[TraceValue],
    ) -> Result<Box<[TraceValue]>, ProgramBuildError> {
        let inputs = program_values(inputs);
        let outputs = self.borrow_mut()?.builder.add_extension(op, &inputs)?;
        Ok(trace_values(outputs))
    }

    /// Return metadata for a context-owned value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a value from another
    /// context, or [`ProgramBuildError::ReentrantContextBorrow`] when the
    /// context is already borrowed.
    pub fn value_metadata(
        &self,
        value: TraceValue,
    ) -> Result<ProgramValueMetadata, ProgramBuildError> {
        self.borrow_mut()?
            .builder
            .value_metadata(value.value)
            .cloned()
    }

    /// Run `f` with the generic trace-time extension cache store.
    ///
    /// The store is borrowed only for the call: a live context shares interior
    /// mutability, so a `&mut` borrow cannot escape.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ReentrantContextBorrow`] when the context is
    /// already borrowed.
    pub fn with_extension_caches<R>(
        &self,
        f: impl FnOnce(&mut ExtensionCacheStore) -> R,
    ) -> Result<R, ProgramBuildError> {
        let mut inner = self.borrow_mut()?;
        Ok(f(&mut inner.extension_caches))
    }

    /// Return whether two handles refer to the same trace.
    #[doc(hidden)]
    pub fn same_context(&self, other: &TraceContext) -> bool {
        std::rc::Rc::ptr_eq(&self.inner, &other.inner)
    }

    /// Add a bound input whose shape and dtype are concrete.
    ///
    /// AD seeds are concrete runtime values, so their extents must be exact:
    /// leaving them symbolic would keep shape expressions unresolved at
    /// execution staging.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] when another input cannot be
    /// represented, or [`ProgramBuildError::ReentrantContextBorrow`] when the
    /// context is already borrowed.
    #[doc(hidden)]
    pub fn bound_input_concrete(
        &self,
        data: Arc<RetainedValue>,
        scalar_identity: Option<&'static str>,
    ) -> Result<TraceValue, ProgramBuildError> {
        let dtype = data.dtype();
        let shape = data.shape().to_vec();
        let metadata = ProgramValueMetadata::from_extents(
            dtype,
            shape.iter().map(|&size| {
                tenferro_ops::ShapeExtent::Exact(tenferro_ops::dim_expr::DimExpr::Const(size))
            }),
        );
        let mut spec = ProgramInputSpec::from_metadata(metadata);
        if let Some(identity) = scalar_identity {
            spec = spec.with_scalar_identity(identity);
        }
        self.input_bound(spec, data)
    }

    /// Return the canonical scalar identity a value's metadata declares.
    #[doc(hidden)]
    pub fn value_scalar_identity(&self, value: TraceValue) -> Option<&'static str> {
        self.inner
            .try_borrow()
            .ok()?
            .builder
            .value_metadata(value.value)
            .ok()
            .and_then(|metadata| metadata.scalar_identity())
    }

    /// Return how many external inputs the context has added.
    pub(crate) fn input_count(&self) -> usize {
        self.inner
            .try_borrow()
            .map(|inner| inner.inputs.len())
            .unwrap_or_default()
    }

    /// Add one input and bind retained data to it in one step.
    pub(crate) fn input_bound(
        &self,
        spec: ProgramInputSpec,
        data: Arc<RetainedValue>,
    ) -> Result<TraceValue, ProgramBuildError> {
        let key = crate::traced::next_input_key();
        let mut inner = self.borrow_mut()?;
        let value = inner.builder.input(spec)?;
        inner.builder.bind_input_retained(value, data)?;
        inner.inputs.push(TraceInput { value, key });
        Ok(TraceValue { value })
    }

    /// Return the leaf key of a context input value.
    #[doc(hidden)]
    pub fn input_key(&self, value: TraceValue) -> Option<TensorInputKey> {
        self.inner
            .try_borrow()
            .ok()?
            .inputs
            .iter()
            .find(|input| input.value == value.value)
            .map(|input| input.key.clone())
    }

    /// Return the program input index of `value` when it is an external input.
    #[doc(hidden)]
    pub fn input_index(&self, value: TraceValue) -> Option<usize> {
        self.inner
            .try_borrow()
            .ok()?
            .inputs
            .iter()
            .position(|input| input.value == value.value)
    }

    /// Return the context inputs in program input order.
    #[doc(hidden)]
    pub fn inputs(&self) -> Vec<(TensorInputKey, TraceValue)> {
        self.inner
            .try_borrow()
            .map(|inner| {
                inner
                    .inputs
                    .iter()
                    .map(|input| (input.key.clone(), TraceValue { value: input.value }))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Record a checkpoint alias and return the bound input that replaces it.
    #[doc(hidden)]
    pub fn checkpoint_value(
        &self,
        target: TraceValue,
        data: Arc<RetainedValue>,
    ) -> Result<TraceValue, ProgramBuildError> {
        let key = crate::traced::next_input_key();
        let shape = data.shape().to_vec();
        let metadata = ProgramValueMetadata::from_extents(
            data.dtype(),
            shape.iter().map(|&size| {
                tenferro_ops::ShapeExtent::Exact(tenferro_ops::dim_expr::DimExpr::Const(size))
            }),
        );
        let mut inner = self.borrow_mut()?;
        let input = inner
            .builder
            .input(ProgramInputSpec::from_metadata(metadata))?;
        inner.builder.bind_input_retained(input, data)?;
        inner.inputs.push(TraceInput { value: input, key });
        inner.aliases.push(CheckpointAlias {
            input,
            target: target.value,
            bound_shape: shape,
        });
        Ok(TraceValue { value: input })
    }

    /// Import a frozen program's roots into this context.
    ///
    /// Mapped source inputs reuse the destination values named in
    /// [`ProgramImport::input_map`](crate::program::ProgramImport::input_map), so
    /// a derivative program lands back on the leaves it was differentiated from.
    ///
    /// # Errors
    ///
    /// Returns the typed [`ProgramBuildError`] of the import transaction, or
    /// [`ProgramBuildError::ReentrantContextBorrow`] when the context is already
    /// borrowed.
    #[doc(hidden)]
    pub fn import_values(
        &self,
        request: crate::program::ProgramImport<'_>,
    ) -> Result<Vec<TraceValue>, ProgramBuildError> {
        let mut inner = self.borrow_mut()?;
        let imported = inner.builder.import(request)?;
        Ok(imported
            .roots()
            .iter()
            .map(|value| TraceValue { value: *value })
            .collect())
    }

    /// Build a traced tensor for a value of this context.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::RuntimeStateSource`] when the value's metadata is
    /// unavailable.
    #[doc(hidden)]
    pub fn derive_tensor(&self, value: TraceValue) -> crate::Result<crate::TracedTensor> {
        crate::trace_ops::finish_derived(self, value)
    }

    /// Freeze the closure of `outputs` with checkpoint aliases substituted.
    ///
    /// The AD source sees the pre-checkpoint definition of every checkpointed
    /// value, refined to the concrete bound shape, so a gradient reaches the
    /// original leaves.
    #[doc(hidden)]
    pub fn ad_snapshot(&self, outputs: &[TraceValue]) -> Result<FrozenProgram, ProgramFinishError> {
        let inner = self.borrow_mut_for_finish()?;
        let values = program_values(outputs);
        inner
            .builder
            .snapshot_pruned_with_aliases(&values, &inner.aliases)
            .map(|(frozen, _)| frozen)
    }

    /// Freeze the forward closure with explicit declarations for named inputs.
    ///
    /// Inputs named here keep the declared dtype and extents instead of the
    /// traced declaration.
    pub(crate) fn snapshot_with_input_specs(
        &self,
        outputs: &[TraceValue],
        specs: &[(TensorInputKey, ProgramInputSpec)],
    ) -> Result<(FrozenProgram, Vec<(TensorInputKey, TraceValue)>), ProgramFinishError> {
        let inner = self.borrow_mut_for_finish()?;
        let values = program_values(outputs);
        let declared: Vec<(usize, ProgramInputSpec)> = specs
            .iter()
            .filter_map(|(key, spec)| {
                inner
                    .inputs
                    .iter()
                    .position(|input| input.key == *key)
                    .map(|index| (index, spec.clone()))
            })
            .collect();
        let (frozen, kept) = inner
            .builder
            .snapshot_pruned_with_input_specs(&values, &declared)?;
        let kept = kept
            .into_iter()
            .filter_map(|value| {
                inner
                    .inputs
                    .iter()
                    .find(|input| input.value == value)
                    .map(|input| (input.key.clone(), TraceValue { value }))
            })
            .collect();
        Ok((frozen, kept))
    }

    /// Freeze the forward closure and report which context inputs it kept.
    pub(crate) fn plain_snapshot_with_inputs(
        &self,
        outputs: &[TraceValue],
    ) -> Result<(FrozenProgram, Vec<(TensorInputKey, TraceValue)>), ProgramFinishError> {
        let inner = self.borrow_mut_for_finish()?;
        let values = program_values(outputs);
        let (frozen, kept) = inner.builder.snapshot_pruned(&values)?;
        let kept = kept
            .into_iter()
            .filter_map(|value| {
                inner
                    .inputs
                    .iter()
                    .find(|input| input.value == value)
                    .map(|input| (input.key.clone(), TraceValue { value }))
            })
            .collect();
        Ok((frozen, kept))
    }

    /// Freeze the AD source and report which context inputs it kept.
    #[doc(hidden)]
    pub fn ad_snapshot_with_inputs(
        &self,
        outputs: &[TraceValue],
    ) -> Result<(FrozenProgram, Vec<(TensorInputKey, TraceValue)>), ProgramFinishError> {
        let inner = self.borrow_mut_for_finish()?;
        let values = program_values(outputs);
        let (frozen, kept) = inner
            .builder
            .snapshot_pruned_with_aliases(&values, &inner.aliases)?;
        let kept = kept
            .into_iter()
            .filter_map(|value| {
                inner
                    .inputs
                    .iter()
                    .find(|input| input.value == value)
                    .map(|input| (input.key.clone(), TraceValue { value }))
            })
            .collect();
        Ok((frozen, kept))
    }

    /// Freeze the ordered closure of `outputs` without consuming the context.
    ///
    /// Later appends do not change the returned graph, and the context stays
    /// usable for further tracing.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramFinishError::ForeignOutput`] if an output belongs to
    /// another context, [`ProgramFinishError::ReentrantContextBorrow`] when the
    /// context is already borrowed, or a typed structural/binding finalization
    /// failure.
    pub fn finish(&self, outputs: &[TraceValue]) -> Result<TracedGraph, ProgramFinishError> {
        let frozen = self.frozen_snapshot(outputs)?;
        Ok(TracedGraph { frozen })
    }

    /// Freeze the ordered closure of `outputs` as a copy of the program.
    #[doc(hidden)]
    pub fn frozen_snapshot(
        &self,
        outputs: &[TraceValue],
    ) -> Result<SnapshotProgram, ProgramFinishError> {
        let inner = self.borrow_mut_for_finish()?;
        let values = program_values(outputs);
        inner
            .builder
            .snapshot_pruned(&values)
            .map(|(frozen, _)| frozen)
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

impl TraceValue {
    /// Return the program value behind this handle.
    #[doc(hidden)]
    pub fn program_value(self) -> ProgramValue {
        self.value
    }
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
    use tenferro_ops::dim_expr::DimExpr;
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
        use crate::TraceContext;
        let context = TraceContext::new();
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

    #[test]
    fn finish_snapshots_without_consuming_the_context() {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
        let frozen = builder.finish(&[output]).unwrap();

        let context = TraceContext::new();
        let value = context
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let negated = context.add_op(CoreSemanticOp::Neg, &[value]).unwrap()[0];
        let first = context.finish(&[negated]).unwrap();
        // The context keeps growing after the snapshot; the snapshot does not.
        let doubled = context.add_op(CoreSemanticOp::Neg, &[negated]).unwrap()[0];
        let second = context.finish(&[doubled]).unwrap();

        assert_eq!(first.program().operations().count(), 1);
        assert_eq!(second.program().operations().count(), 2);
        assert_eq!(
            first
                .program()
                .value_metadata(first.program().outputs()[0])
                .unwrap(),
            frozen
                .program
                .value_metadata(frozen.program.outputs()[0])
                .unwrap()
        );
    }

    #[test]
    fn foreign_values_are_rejected_by_metadata_and_ops() {
        let first = TraceContext::new();
        let second = TraceContext::new();
        let value = first
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1)]))
            .unwrap();
        let foreign = second
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(1)]))
            .unwrap();

        assert!(first.value_metadata(value).is_ok());
        assert!(matches!(
            first.value_metadata(foreign),
            Err(ProgramBuildError::ForeignValue)
        ));
        assert!(matches!(
            first.add_op(CoreSemanticOp::Neg, &[foreign]),
            Err(ProgramBuildError::ForeignValue)
        ));
    }

    #[test]
    fn extension_caches_are_reached_through_a_scoped_borrow() {
        let context = TraceContext::new();
        let entries = context
            .with_extension_caches(|caches| caches.len())
            .unwrap();
        assert_eq!(entries, 0);
    }
}
