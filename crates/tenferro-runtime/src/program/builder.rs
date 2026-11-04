use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAlias, ExtensionAliasDeclaration, ExtensionEffectAccess, ExtensionEffectDeclaration,
    ExtensionOp,
};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::Tensor;

use crate::checkpoint::RetainedValue;

use super::bindings::PendingBinding;
use super::identity::SemanticIdentity;
use super::metadata::SemanticProvenance;
use super::op::{SemanticOp, SemanticOperation};
use super::value::ProgramBuilderNonce;
use super::{
    Alias, BindingKey, CoreSemanticOp, Effect, EffectAccess, EffectResource, FrozenProgram,
    ImportedProgramValues, ProgramBindingError, ProgramBindings, ProgramBuildError,
    ProgramFinishError, ProgramImport, ProgramInputSpec, ProgramShapeRelation,
    ProgramStructuralError, ProgramValue, ProgramValueMetadata, SemanticPlacementConstraint,
    SemanticProgram, ShapeGuard,
};

/// Mutable validation boundary for one semantic program.
pub struct SemanticProgramBuilder {
    owner: ProgramBuilderNonce,
    inputs: Vec<ProgramValue>,
    input_specs: Vec<ProgramInputSpec>,
    values: Vec<ProgramValueMetadata>,
    operations: Vec<SemanticOperation>,
    bindings: Vec<PendingBinding>,
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
            inputs: Vec::new(),
            input_specs: Vec::new(),
            values: Vec::new(),
            operations: Vec::new(),
            bindings: Vec::new(),
        }
    }

    /// Attach a tensor default or large constant to one external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a token from another
    /// builder, [`ProgramBuildError::BindingTargetNotInput`] for a computed
    /// value, or [`ProgramBuildError::DuplicateBinding`] when the input already
    /// has a binding.
    pub fn bind_input(
        &mut self,
        input: ProgramValue,
        tensor: Tensor,
    ) -> Result<BindingKey, ProgramBuildError> {
        self.validate_value(input)?;
        if !self.inputs.contains(&input) {
            return Err(ProgramBuildError::BindingTargetNotInput);
        }
        if self.bindings.iter().any(|binding| binding.input == input) {
            return Err(ProgramBuildError::DuplicateBinding);
        }
        self.bind_input_retained(input, Arc::new(RetainedValue::from_tensor(tensor)))
    }

    /// Attach an already-retained tensor default to one external input.
    ///
    /// Owner-scoped hook for the eager semantic record, which retains its leaf
    /// values and must not copy them into an owned [`Tensor`] to bind them.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a token from another
    /// builder, [`ProgramBuildError::BindingTargetNotInput`] for a computed
    /// value, or [`ProgramBuildError::DuplicateBinding`] when the input already
    /// has a binding.
    #[doc(hidden)]
    pub fn bind_input_retained(
        &mut self,
        input: ProgramValue,
        tensor: Arc<RetainedValue>,
    ) -> Result<BindingKey, ProgramBuildError> {
        self.validate_value(input)?;
        if !self.inputs.contains(&input) {
            return Err(ProgramBuildError::BindingTargetNotInput);
        }
        if self.bindings.iter().any(|binding| binding.input == input) {
            return Err(ProgramBuildError::DuplicateBinding);
        }
        let key = BindingKey::new(input.slot, self.owner);
        self.bindings.push(PendingBinding { key, input, tensor });
        Ok(key)
    }

    /// Add one ordered external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] if the builder cannot
    /// represent another value slot.
    pub fn input(&mut self, spec: ProgramInputSpec) -> Result<ProgramValue, ProgramBuildError> {
        require_scalar_identity(spec.metadata(), "a program input")?;
        let slot = self.next_value_slot()?;
        let value = ProgramValue::new(slot, self.owner);
        self.values.push(spec.metadata().clone());
        self.inputs.push(value);
        self.input_specs.push(spec);
        Ok(value)
    }

    /// Validate that a value belongs to this builder.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a token from another
    /// builder or one that does not name an existing value.
    pub fn validate_value(&self, value: ProgramValue) -> Result<(), ProgramBuildError> {
        if value.owner != self.owner || value.slot as usize >= self.values.len() {
            return Err(ProgramBuildError::ForeignValue);
        }
        Ok(())
    }

    /// Borrow metadata for a builder-local value.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a foreign token.
    pub fn value_metadata(
        &self,
        value: ProgramValue,
    ) -> Result<&ProgramValueMetadata, ProgramBuildError> {
        self.validate_value(value)?;
        Ok(&self.values[value.slot as usize])
    }

    /// Return the number of semantic operations added so far.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    #[cfg(test)]
    pub(crate) fn add_shape_guards_to_output(
        &mut self,
        output: ProgramValue,
        guards: impl IntoIterator<Item = ShapeGuard>,
    ) -> Result<(), ProgramBuildError> {
        self.validate_value(output)?;
        let operation = self
            .operations
            .iter_mut()
            .find(|operation| operation.outputs.contains(&output))
            .ok_or(ProgramBuildError::GuardTargetNotOperationOutput)?;
        let mut combined = operation.shape_guards.to_vec();
        combined.extend(guards);
        operation.shape_guards = combined.into_boxed_slice();
        Ok(())
    }

    /// Import the dependency closure of ordered source roots atomically.
    ///
    /// Empty and duplicate roots are preserved. Tensor bindings remain
    /// separate and are remapped only for imported source inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignImportRoot`] for a root outside the
    /// source program, [`ProgramBuildError::ForeignBindings`] for bindings
    /// frozen with another program, [`ProgramBuildError::InvalidImport`] for
    /// invalid source structure, or [`ProgramBuildError::TooManyValues`] when
    /// the destination cannot represent the imported values. On error this
    /// builder is unchanged.
    pub fn import(
        &mut self,
        request: ProgramImport<'_>,
    ) -> Result<ImportedProgramValues, ProgramBuildError> {
        let transaction = ImportTransaction::prepare(self, request)?;
        let roots = transaction.roots.clone();
        self.inputs.extend(transaction.inputs);
        self.input_specs.extend(transaction.input_specs);
        self.values.extend(transaction.values);
        self.operations.extend(transaction.operations);
        self.bindings.extend(transaction.bindings);
        Ok(ImportedProgramValues::new(roots))
    }

    /// Snapshot the live closure of `outputs`, pruning dead operation outputs.
    ///
    /// Only values reachable from `outputs` are kept. An extension operation whose
    /// dead outputs are prunable is replaced by its pruned form (for example
    /// `Svd` used for its values only becomes `SvdVals`), so a values-only use
    /// never executes the unused kernels.
    ///
    /// # Errors
    ///
    /// Returns the typed failures of [`finish`](Self::finish), or
    /// [`ProgramFinishError::SnapshotRebuild`] when replaying the live closure
    /// fails.
    pub(crate) fn snapshot_pruned(
        &self,
        outputs: &[ProgramValue],
    ) -> Result<(FrozenProgram, Vec<ProgramValue>), ProgramFinishError> {
        self.snapshot_pruned_with_aliases_and_specs(outputs, &[], &[])
    }

    /// Snapshot with explicit declarations for kept inputs.
    ///
    /// A declared input spec replaces the traced declaration, which is how
    /// `compile_with_input_specs` specializes placeholder extents to the concrete
    /// shapes the caller declares.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as
    /// [`snapshot_pruned_with_aliases`](Self::snapshot_pruned_with_aliases).
    pub(crate) fn snapshot_pruned_with_input_specs(
        &self,
        outputs: &[ProgramValue],
        input_specs: &[(usize, ProgramInputSpec)],
    ) -> Result<(FrozenProgram, Vec<ProgramValue>), ProgramFinishError> {
        self.snapshot_pruned_with_aliases_and_specs(outputs, &[], input_specs)
    }

    /// Snapshot the live closure of `outputs`, substituting checkpoint aliases.
    ///
    /// An aliased input is replaced by a prefix `Slice` of its pre-checkpoint
    /// value, inserted just before its first use so the rebuilt SSA order stays
    /// valid. Dead operation outputs are pruned at the same time.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as
    /// [`snapshot_pruned`](Self::snapshot_pruned).
    pub(crate) fn snapshot_pruned_with_aliases(
        &self,
        outputs: &[ProgramValue],
        aliases: &[crate::trace::CheckpointAlias],
    ) -> Result<(FrozenProgram, Vec<ProgramValue>), ProgramFinishError> {
        self.snapshot_pruned_with_aliases_and_specs(outputs, aliases, &[])
    }

    fn snapshot_pruned_with_aliases_and_specs(
        &self,
        outputs: &[ProgramValue],
        aliases: &[crate::trace::CheckpointAlias],
        input_specs: &[(usize, ProgramInputSpec)],
    ) -> Result<(FrozenProgram, Vec<ProgramValue>), ProgramFinishError> {
        let value_count = self.values.len();
        let mut live = vec![false; value_count];
        for output in outputs {
            if output.owner != self.owner || output.slot as usize >= value_count {
                return Err(ProgramFinishError::ForeignOutput);
            }
            live[output.slot as usize] = true;
        }
        // An effectful operation is observable even when no caller asks for its
        // outputs, so the operation itself seeds liveness like a requested output,
        // and its outputs stay live so the rebuilt operation keeps its arity.
        let mut live_operation = vec![false; self.operations.len()];
        for (index, operation) in self.operations.iter().enumerate() {
            if operation.effects.is_empty() {
                continue;
            }
            live_operation[index] = true;
            for output in &operation.outputs {
                live[output.slot as usize] = true;
            }
        }
        for alias in aliases {
            let slot = alias.input.slot as usize;
            if live[slot] {
                live[alias.target.slot as usize] = true;
            }
        }
        let alias_of = |value: ProgramValue| {
            aliases
                .iter()
                .find(|alias| alias.input == value)
                .map(|alias| (alias.target, alias.bound_shape.clone()))
        };
        for _ in 0..=aliases.len() {
            let mut changed = false;
            for (index, operation) in self.operations.iter().enumerate().rev() {
                if !live_operation[index]
                    && !operation
                        .outputs
                        .iter()
                        .any(|output| live[output.slot as usize])
                {
                    continue;
                }
                live_operation[index] = true;
                for input in &operation.inputs {
                    if !live[input.slot as usize] {
                        live[input.slot as usize] = true;
                        changed = true;
                    }
                }
            }
            for alias in aliases {
                if live[alias.input.slot as usize] && !live[alias.target.slot as usize] {
                    live[alias.target.slot as usize] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // An extension op that cannot prune its outputs must keep all of them, or
        // the rebuilt operation would not match the recorded output count.
        loop {
            let mut changed = false;
            for operation in self.operations.iter().rev() {
                let SemanticOp::Extension(extension) = &operation.op else {
                    continue;
                };
                if !operation.effects.is_empty() {
                    continue;
                }
                let live_outputs: Vec<bool> = operation
                    .outputs
                    .iter()
                    .map(|output| live[output.slot as usize])
                    .collect();
                if !live_outputs.iter().any(|live| *live) || live_outputs.iter().all(|live| *live) {
                    continue;
                }
                if extension.prune_outputs(&live_outputs).is_some() {
                    continue;
                }
                for output in &operation.outputs {
                    live[output.slot as usize] = true;
                }
                changed = true;
            }
            if !changed {
                break;
            }
        }

        // Declared inputs are created first, in the declared order, so a caller's
        // explicit input order becomes the compiled program's input order.
        let mut creation_order: Vec<(usize, Option<ProgramInputSpec>)> =
            Vec::with_capacity(self.inputs.len());
        for (source_index, declared) in input_specs {
            if self.input_specs.get(*source_index).is_some() {
                creation_order.push((*source_index, Some(declared.clone())));
            }
        }
        for index in 0..self.inputs.len() {
            if input_specs.iter().any(|(declared, _)| *declared == index) {
                continue;
            }
            creation_order.push((index, None));
        }

        // Program dimension expressions name inputs by program input position; the
        // kept inputs are renumbered into creation order, so their declarations are
        // rewritten against that order.
        let mut input_index = std::collections::HashMap::new();
        for (source_index, _) in &creation_order {
            let Some(input) = self.inputs.get(*source_index) else {
                continue;
            };
            if !live[input.slot as usize] || alias_of(*input).is_some() {
                continue;
            }
            input_index.insert(*source_index, input_index.len());
        }
        fn remap_expr(
            expr: &DimExpr,
            input_index: &std::collections::HashMap<usize, usize>,
        ) -> Option<DimExpr> {
            Some(match expr {
                DimExpr::Const(value) => DimExpr::Const(*value),
                DimExpr::InputDim { input_idx, axis } => DimExpr::InputDim {
                    input_idx: *input_index.get(input_idx)?,
                    axis: *axis,
                },
                DimExpr::Add(lhs, rhs) => DimExpr::Add(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
                DimExpr::Sub(lhs, rhs) => DimExpr::Sub(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
                DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
                DimExpr::FloorDiv(lhs, rhs) => DimExpr::FloorDiv(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
                DimExpr::Min(lhs, rhs) => DimExpr::Min(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
                DimExpr::Max(lhs, rhs) => DimExpr::Max(
                    Box::new(remap_expr(lhs, input_index)?),
                    Box::new(remap_expr(rhs, input_index)?),
                ),
            })
        }

        let mut rebuild = Self::new();
        let mut kept_inputs = Vec::new();
        let mut mapped: Vec<Option<ProgramValue>> = vec![None; value_count];
        for (source_index, declared) in creation_order {
            let (Some(input), Some(spec)) = (
                self.inputs.get(source_index),
                self.input_specs.get(source_index),
            ) else {
                continue;
            };
            if !live[input.slot as usize] || alias_of(*input).is_some() {
                continue;
            }
            kept_inputs.push(*input);
            let built = match declared {
                Some(declared) => declared,
                None => {
                    let metadata = spec.metadata();
                    let extents: Vec<ShapeExtent<DimExpr>> = metadata
                        .shape()
                        .iter()
                        .map(
                            |extent| -> Result<ShapeExtent<DimExpr>, ProgramFinishError> {
                                let invalid = || ProgramFinishError::SnapshotRebuild {
                                    source: ProgramBuildError::InvalidImport {
                                        source: ProgramStructuralError::InvalidValueReference,
                                    },
                                };
                                Ok(match extent {
                                    ShapeExtent::Exact(expr) => ShapeExtent::Exact(
                                        remap_expr(expr, &input_index).ok_or_else(invalid)?,
                                    ),
                                    ShapeExtent::UpperBound(expr) => ShapeExtent::UpperBound(
                                        remap_expr(expr, &input_index).ok_or_else(invalid)?,
                                    ),
                                    ShapeExtent::Unknown => ShapeExtent::Unknown,
                                })
                            },
                        )
                        .collect::<Result<_, _>>()?;
                    let mut built = ProgramInputSpec::from_metadata(
                        ProgramValueMetadata::from_extents(metadata.dtype(), extents),
                    );
                    if let Some(identity) = metadata.scalar_identity() {
                        built = built.with_scalar_identity(identity);
                    }
                    built
                }
            };
            let rebuilt = rebuild.input(built).map_err(snapshot_rebuild)?;
            if let Some(binding) = self.bindings.iter().find(|binding| binding.input == *input) {
                rebuild
                    .bind_input_retained(rebuilt, Arc::clone(&binding.tensor))
                    .map_err(snapshot_rebuild)?;
            }
            mapped[input.slot as usize] = Some(rebuilt);
        }

        let materialize_alias = |rebuild: &mut Self,
                                 mapped: &mut Vec<Option<ProgramValue>>,
                                 input: ProgramValue|
         -> Result<(), ProgramFinishError> {
            if mapped[input.slot as usize].is_some() {
                return Ok(());
            }
            let Some((target, shape)) = alias_of(input) else {
                return Err(ProgramFinishError::SnapshotRebuild {
                    source: ProgramBuildError::InvalidImport {
                        source: ProgramStructuralError::InvalidValueReference,
                    },
                });
            };
            let Some(mapped_target) = mapped[target.slot as usize] else {
                return Err(ProgramFinishError::SnapshotRebuild {
                    source: ProgramBuildError::InvalidImport {
                        source: ProgramStructuralError::InvalidSsaOrder,
                    },
                });
            };
            let rank = shape.len();
            let slice = CoreSemanticOp::Slice(tenferro_tensor::SliceConfig {
                starts: vec![0; rank],
                limits: shape,
                strides: vec![1; rank],
            });
            let sliced = rebuild
                .add_op(slice, &[mapped_target])
                .map_err(snapshot_rebuild)?[0];
            mapped[input.slot as usize] = Some(sliced);
            Ok(())
        };

        for (index, operation) in self.operations.iter().enumerate() {
            if !live_operation[index] {
                continue;
            }
            let live_outputs: Vec<bool> = operation
                .outputs
                .iter()
                .map(|output| live[output.slot as usize])
                .collect();
            for input in &operation.inputs {
                if alias_of(*input).is_some() {
                    materialize_alias(&mut rebuild, &mut mapped, *input)?;
                }
            }
            let inputs = operation
                .inputs
                .iter()
                .map(|input| {
                    mapped[input.slot as usize].ok_or(ProgramFinishError::SnapshotRebuild {
                        source: ProgramBuildError::InvalidImport {
                            source: ProgramStructuralError::InvalidSsaOrder,
                        },
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let rebuilt_outputs = match &operation.op {
                SemanticOp::Core(core) => rebuild.add_op(core.clone(), &inputs),
                SemanticOp::Extension(extension) => rebuild.add_extension(
                    extension
                        .prune_outputs(&live_outputs)
                        .unwrap_or_else(|| extension.clone_arc()),
                    &inputs,
                ),
            }
            .map_err(snapshot_rebuild)?;
            discharge_static_guards(&mut rebuild);
            let kept: Vec<ProgramValue> = operation
                .outputs
                .iter()
                .zip(live_outputs.iter())
                .filter(|(_, live)| **live)
                .map(|(output, _)| *output)
                .collect();
            if rebuilt_outputs.len() != kept.len() {
                return Err(ProgramFinishError::SnapshotRebuild {
                    source: ProgramBuildError::OutputMetadataCount {
                        expected: kept.len(),
                        actual: rebuilt_outputs.len(),
                    },
                });
            }
            for (source, rebuilt) in kept.iter().zip(rebuilt_outputs.iter()) {
                mapped[source.slot as usize] = Some(*rebuilt);
            }
        }

        for output in outputs {
            if alias_of(*output).is_some() {
                materialize_alias(&mut rebuild, &mut mapped, *output)?;
            }
        }
        let mapped_outputs = outputs
            .iter()
            .map(|output| {
                mapped[output.slot as usize].ok_or(ProgramFinishError::SnapshotRebuild {
                    source: ProgramBuildError::InvalidImport {
                        source: ProgramStructuralError::InvalidValueReference,
                    },
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((rebuild.finish(&mapped_outputs)?, kept_inputs))
    }

    /// Consume this builder and atomically freeze semantic structure and bindings.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramFinishError::ForeignOutput`] for an output outside this
    /// builder, [`ProgramFinishError::StructuralValidation`] for invalid SSA
    /// structure, or [`ProgramFinishError::BindingFinalization`] when a tensor
    /// binding does not match its input declaration.
    pub fn finish(self, outputs: &[ProgramValue]) -> Result<FrozenProgram, ProgramFinishError> {
        if outputs
            .iter()
            .any(|output| output.owner != self.owner || output.slot as usize >= self.values.len())
        {
            return Err(ProgramFinishError::ForeignOutput);
        }

        validate_structure(
            self.owner,
            &self.inputs,
            self.values.len(),
            &self.operations,
        )?;
        validate_bindings(&self.inputs, &self.input_specs, &self.bindings)?;

        let inputs = self.inputs.into_boxed_slice();
        let outputs: Box<[ProgramValue]> = outputs.into();
        let values = self.values.into_boxed_slice();
        let operations = self.operations.into_boxed_slice();
        let shape_guards: Box<[ShapeGuard]> = operations
            .iter()
            .flat_map(|operation| operation.shape_guards.iter().cloned())
            .collect();
        let identity =
            SemanticIdentity::build(&inputs, &outputs, &values, &operations, &shape_guards);
        let bindings = ProgramBindings::freeze(self.owner, self.bindings);
        let program = SemanticProgram {
            owner: self.owner,
            inputs,
            outputs,
            values,
            operations,
            shape_guards,
            identity,
        };
        Ok(FrozenProgram {
            program: Arc::new(program),
            bindings,
        })
    }

    /// Translate a shape payload written in program coordinates into the
    /// coordinates of the operation that will carry it.
    ///
    /// A rule derives a `Reshape`/`BroadcastInDim` payload from value metadata, so
    /// its `InputDim`s name *program* inputs. The operation it lands on names only
    /// its own inputs, so every referenced program input is appended to
    /// `operation_inputs` as an auxiliary shape reference. Extents stay symbolic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::dim_expr::DimExpr;
    /// use tenferro_runtime::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
    /// use tenferro_runtime::DType;
    ///
    /// let mut builder = SemanticProgramBuilder::new();
    /// let operand = builder.input(ProgramInputSpec::new(
    ///     DType::F64,
    ///     [DimExpr::InputDim { input_idx: 0, axis: 0 }],
    /// ))?;
    /// let mut inputs = vec![operand];
    /// let shape = builder.translate_input_shape_payload(
    ///     &[DimExpr::InputDim { input_idx: 0, axis: 0 }],
    ///     &mut inputs,
    /// );
    /// assert_eq!(shape, [DimExpr::InputDim { input_idx: 0, axis: 0 }]);
    /// assert_eq!(inputs.len(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn translate_input_shape_payload(
        &self,
        payload: &[DimExpr],
        operation_inputs: &mut Vec<ProgramValue>,
    ) -> Vec<DimExpr> {
        payload
            .iter()
            .map(|extent| self.translate_input_shape_extent(extent, operation_inputs))
            .collect()
    }

    fn translate_input_shape_extent(
        &self,
        extent: &DimExpr,
        operation_inputs: &mut Vec<ProgramValue>,
    ) -> DimExpr {
        match extent {
            DimExpr::Const(value) => DimExpr::Const(*value),
            DimExpr::InputDim { input_idx, axis } => {
                let Some(input) = self.inputs.get(*input_idx) else {
                    return extent.clone();
                };
                let positioned = match operation_inputs.iter().position(|value| value == input) {
                    Some(position) => position,
                    None => {
                        operation_inputs.push(*input);
                        operation_inputs.len() - 1
                    }
                };
                DimExpr::InputDim {
                    input_idx: positioned,
                    axis: *axis,
                }
            }
            DimExpr::Add(lhs, rhs) => DimExpr::Add(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
            DimExpr::Sub(lhs, rhs) => DimExpr::Sub(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
            DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
            DimExpr::FloorDiv(lhs, rhs) => DimExpr::FloorDiv(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
            DimExpr::Min(lhs, rhs) => DimExpr::Min(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
            DimExpr::Max(lhs, rhs) => DimExpr::Max(
                Box::new(self.translate_input_shape_extent(lhs, operation_inputs)),
                Box::new(self.translate_input_shape_extent(rhs, operation_inputs)),
            ),
        }
    }

    /// Borrow the program inputs in program input order.
    #[doc(hidden)]
    pub fn inputs(&self) -> &[ProgramValue] {
        &self.inputs
    }

    /// Resolve `value`'s declared extents against the shapes bound to the inputs.
    ///
    /// A program keeps symbolic extents so it stays shape-polymorphic; a caller
    /// that needs a literal length reads it here instead of asking the program to
    /// become concrete.
    #[doc(hidden)]
    pub fn resolved_extents(
        &self,
        value: ProgramValue,
    ) -> Result<Vec<ShapeExtent<DimExpr>>, ProgramBuildError> {
        self.value_metadata(value)?
            .shape()
            .iter()
            .enumerate()
            .map(|(axis, _)| self.resolved_extent(value, axis))
            .collect()
    }

    /// Resolve one of `value`'s declared extents against the shapes bound to the
    /// inputs it names.
    ///
    /// A caller asking about a single axis pays only for the inputs that axis
    /// references; an expression that is already constant needs no binding at all.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::dim_expr::DimExpr;
    /// use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
    /// use tenferro_runtime::DType;
    ///
    /// let mut builder = SemanticProgramBuilder::new();
    /// let input = builder.input(ProgramInputSpec::new(
    ///     DType::F64,
    ///     [DimExpr::InputDim { input_idx: 0, axis: 0 }],
    /// ))?;
    /// let extent = builder.resolved_extent(input, 0)?;
    /// // Nothing is bound, so the symbolic declaration is unchanged.
    /// assert_eq!(
    ///     extent,
    ///     tenferro_ops::ShapeExtent::Exact(DimExpr::InputDim { input_idx: 0, axis: 0 })
    /// );
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::ForeignValue`] for a value of another builder,
    /// or a metadata error when `axis` is outside the value's rank.
    #[doc(hidden)]
    pub fn resolved_extent(
        &self,
        value: ProgramValue,
        axis: usize,
    ) -> Result<ShapeExtent<DimExpr>, ProgramBuildError> {
        let metadata = self.value_metadata(value)?;
        let extent =
            metadata
                .shape()
                .get(axis)
                .ok_or_else(|| ProgramBuildError::InvalidImport {
                    source: ProgramStructuralError::InvalidValueReference,
                })?;
        // Resolve only the inputs this one expression names: a constant extent, or
        // one whose inputs carry no binding, needs no lookup at all.
        let resolve = |expr: &DimExpr| -> DimExpr {
            // A constant extent, or a builder with no bindings at all, needs no work.
            if expr.max_input_idx().is_none() || self.bindings.is_empty() {
                return expr.clone();
            }
            // Substitute one referenced input at a time, so nothing proportional to
            // the input count is allocated per axis query.
            let mut resolved = expr.clone();
            let mut positions = Vec::new();
            collect_input_positions(expr, &mut positions);
            positions.sort_unstable();
            positions.dedup();
            for position in positions {
                let Some(input) = self.inputs.get(position) else {
                    continue;
                };
                let Some(bound) = self
                    .bindings
                    .iter()
                    .find(|binding| binding.input == *input)
                    .map(|binding| binding.tensor.shape().to_vec())
                else {
                    continue;
                };
                resolved = substitute_input_extents(&resolved, position, &bound);
            }
            resolved
        };
        Ok(match extent {
            ShapeExtent::Exact(expr) => ShapeExtent::Exact(resolve(expr)),
            ShapeExtent::UpperBound(expr) => ShapeExtent::UpperBound(resolve(expr)),
            ShapeExtent::Unknown => ShapeExtent::Unknown,
        })
    }

    #[cfg(test)]
    pub(crate) fn operation_views_for_test(
        &self,
    ) -> impl ExactSizeIterator<Item = super::SemanticOperationView<'_>> + '_ {
        self.operations
            .iter()
            .map(super::SemanticOperationView::new)
    }

    /// Add one canonical core semantic operation.
    ///
    /// # Errors
    ///
    /// Returns a typed build error for foreign values, wrong arity, invalid
    /// metadata, or an unrepresentable output count.
    pub fn add_op(
        &mut self,
        op: CoreSemanticOp,
        inputs: &[ProgramValue],
    ) -> Result<Box<[ProgramValue]>, ProgramBuildError> {
        self.validate_inputs(inputs)?;
        validate_arity(op.input_count(), inputs.len())?;
        reject_core_external_dtype(&op)?;
        let output_count = op.output_count();
        let metadata = self.infer_core_metadata(&op, inputs)?;
        validate_output_count(output_count, metadata.len())?;
        let aliases = (0..output_count).map(Alias::fresh).collect();
        self.append_operation(
            SemanticOp::Core(op),
            inputs,
            metadata,
            Vec::new(),
            aliases,
            Vec::new(),
        )
    }

    /// Add one extension semantic operation with explicit effects and aliases.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    /// use std::hash::Hasher;
    /// use std::sync::Arc;
    /// use tenferro_ops::dim_expr::DimExpr;
    /// use tenferro_ops::ext_op::{
    ///     ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp,
    /// };
    /// use tenferro_ops::{ExtensionShapeContext, SymDim};
    /// use tenferro_runtime::program::{ProgramInputSpec, SemanticProgramBuilder};
    /// use tenferro_tensor::DType;
    ///
    /// #[derive(Clone, Debug)]
    /// struct Identity;
    /// impl ExtensionOp for Identity {
    ///     fn family_id(&self) -> &'static str { "example.identity.v1" }
    ///     fn payload_hash(&self, hasher: &mut dyn Hasher) {
    ///         hasher.write_u8(1);
    ///     }
    ///     fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
    ///         other.as_any().is::<Self>()
    ///     }
    ///     fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
    ///         Arc::new(self.clone())
    ///     }
    ///     fn as_any(&self) -> &dyn Any { self }
    ///     fn input_count(&self) -> usize { 1 }
    ///     fn output_count(&self) -> usize { 1 }
    ///     fn infer_output_meta(
    ///         &self,
    ///         context: &mut ExtensionShapeContext<'_>,
    ///     ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
    ///         Ok(vec![(
    ///             context.input_dtype(0)?,
    ///             context.input_shape(0)?.to_vec(),
    ///         )])
    ///     }
    ///     fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
    ///         ExtensionEffectDeclaration::Declared(&[])
    ///     }
    ///     fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
    ///         ExtensionAliasDeclaration::AllFresh
    ///     }
    /// }
    ///
    /// let mut builder = SemanticProgramBuilder::new();
    /// let input = builder.input(ProgramInputSpec::new(
    ///     DType::F64,
    ///     [DimExpr::Const(2)],
    /// ))?;
    /// let output = builder.add_extension(Arc::new(Identity), &[input])?[0];
    /// let frozen = builder.finish(&[output])?;
    /// assert_eq!(frozen.program.operations().count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed build error when the payload leaves effects or aliases
    /// undeclared, metadata inference fails, or any value/arity/alias is
    /// invalid.
    pub fn add_extension(
        &mut self,
        op: Arc<dyn ExtensionOp>,
        inputs: &[ProgramValue],
    ) -> Result<Box<[ProgramValue]>, ProgramBuildError> {
        self.validate_inputs(inputs)?;
        validate_arity(op.input_count(), inputs.len())?;
        let effects = extension_effects(op.as_ref())?;
        let aliases = extension_aliases(op.as_ref())?;
        validate_aliases(&aliases, inputs.len(), op.output_count())?;
        let (metadata, guards) = self.infer_extension_metadata(op.as_ref(), inputs)?;
        validate_output_count(op.output_count(), metadata.len())?;
        self.append_operation(
            SemanticOp::Extension(op),
            inputs,
            metadata,
            effects,
            aliases,
            guards,
        )
    }

    fn next_value_slot(&self) -> Result<u32, ProgramBuildError> {
        u32::try_from(self.values.len()).map_err(|_| ProgramBuildError::TooManyValues)
    }

    fn validate_inputs(&self, inputs: &[ProgramValue]) -> Result<(), ProgramBuildError> {
        inputs
            .iter()
            .try_for_each(|&value| self.validate_value(value))
    }

    fn input_metadata(
        &self,
        inputs: &[ProgramValue],
    ) -> Result<Vec<&ProgramValueMetadata>, ProgramBuildError> {
        inputs
            .iter()
            .map(|&value| self.value_metadata(value))
            .collect()
    }

    fn infer_core_metadata(
        &self,
        op: &CoreSemanticOp,
        inputs: &[ProgramValue],
    ) -> Result<Vec<ProgramValueMetadata>, ProgramBuildError> {
        let input_metadata = self.input_metadata(inputs)?;
        let precision = input_extent_precision(&input_metadata);
        let input_dtypes: Vec<_> = input_metadata
            .iter()
            .map(|metadata| metadata.dtype())
            .collect();
        let input_shapes = inference_shapes(&input_metadata);
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let standard = tenferro_ops::std_tensor_op::StdTensorOp::from(op);
        let dtype = crate::shape_infer::infer_output_dtype(&standard, &input_dtypes)
            .map_err(metadata_error)?;
        if core_output_uses_local_shape_coordinates(op) {
            let local_input_shapes: Vec<_> = input_metadata
                .iter()
                .enumerate()
                .map(|(input_idx, metadata)| {
                    DimExpr::input_shape(input_idx, metadata.shape().len())
                })
                .collect();
            let local_input_shape_refs: Vec<_> =
                local_input_shapes.iter().map(Vec::as_slice).collect();
            let output_extents =
                crate::shape_infer::infer_output_extents(&standard, &local_input_shape_refs)
                    .map_err(metadata_error)?;
            output_extents
                .into_iter()
                .map(|shape| {
                    resolve_inferred_extents(shape, precision, &input_shape_refs)
                        .map(|shape| ProgramValueMetadata::from_extents(dtype, shape))
                })
                .collect()
        } else {
            let output_extents =
                crate::shape_infer::infer_output_extents(&standard, &input_shape_refs)
                    .map_err(metadata_error)?;
            Ok(output_extents
                .into_iter()
                .map(|shape| {
                    ProgramValueMetadata::from_extents(
                        dtype,
                        conservatively_bound_extents(shape, precision),
                    )
                })
                .collect())
        }
    }

    fn infer_extension_metadata(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[ProgramValue],
    ) -> Result<(Vec<ProgramValueMetadata>, Vec<ShapeGuard>), ProgramBuildError> {
        let input_metadata = self.input_metadata(inputs)?;
        let precision = input_extent_precision(&input_metadata);
        let input_dtypes: Vec<_> = input_metadata
            .iter()
            .map(|metadata| metadata.dtype())
            .collect();
        let input_shapes = inference_shapes(&input_metadata);
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let inferred = crate::shape_infer::infer_extension_output_meta_with_constraints(
            op,
            &input_dtypes,
            &input_shape_refs,
        )
        .map_err(metadata_error)?;
        let metadata = inferred
            .output_metas
            .into_iter()
            .map(|(dtype, shape)| {
                ProgramValueMetadata::from_extents(
                    dtype,
                    conservatively_bound_extents(
                        shape.into_iter().map(ShapeExtent::Exact),
                        precision,
                    ),
                )
            })
            .collect();
        let guards = inferred
            .constraints
            .into_iter()
            .map(|constraint| {
                let relation = match constraint.relation {
                    tenferro_ops::ShapeRelation::Equal => ProgramShapeRelation::Equal,
                };
                ShapeGuard::new(relation, constraint.lhs, constraint.rhs)
            })
            .collect();
        Ok((metadata, guards))
    }

    fn append_operation(
        &mut self,
        op: SemanticOp,
        inputs: &[ProgramValue],
        metadata: Vec<ProgramValueMetadata>,
        effects: Vec<Effect>,
        aliases: Vec<Alias>,
        shape_guards: Vec<ShapeGuard>,
    ) -> Result<Box<[ProgramValue]>, ProgramBuildError> {
        // An extension that carries an externally defined scalar declares its
        // canonical identity, which every external value of this operation carries
        // so the program can be given a reproducible identity.
        let identity = match &op {
            SemanticOp::Core(_) => None,
            SemanticOp::Extension(extension) => extension.scalar_identity(),
        };
        let mut metadata = metadata;
        for value in &mut metadata {
            if matches!(value.dtype(), tenferro_tensor::DType::External(_))
                && let Some(identity) = identity
            {
                *value = value.clone().with_scalar_identity(identity);
            }
            require_scalar_identity(value, "an operation output")?;
        }
        let provenance = match &op {
            SemanticOp::Core(_) => SemanticProvenance::builder(None),
            SemanticOp::Extension(extension) => {
                SemanticProvenance::builder(Some(extension.family_id()))
            }
        };
        let start = self.values.len();
        let end = start
            .checked_add(metadata.len())
            .ok_or(ProgramBuildError::TooManyValues)?;
        if end > u32::MAX as usize {
            return Err(ProgramBuildError::TooManyValues);
        }
        let outputs: Box<[_]> = (start..end)
            .map(|slot| ProgramValue::new(slot as u32, self.owner))
            .collect();
        self.values.extend(metadata);
        self.operations.push(SemanticOperation {
            op,
            inputs: inputs.into(),
            outputs: outputs.clone(),
            effects: effects.into(),
            aliases: aliases.into(),
            shape_guards: shape_guards.into(),
            placement: SemanticPlacementConstraint::any(),
            provenance,
        });
        Ok(outputs)
    }
}

/// Drop shape guards whose both sides are already constant and equal.
///
/// A snapshot of a fully concrete program has nothing left to check at
/// execution; keeping the guard would make the runtime re-evaluate an equality
/// the builder already decided.
fn discharge_static_guards(builder: &mut SemanticProgramBuilder) {
    let Some(operation) = builder.operations.last_mut() else {
        return;
    };
    let kept: Vec<ShapeGuard> = operation
        .shape_guards
        .iter()
        .filter(|guard| {
            let lhs = static_dim_value(guard.lhs());
            let rhs = static_dim_value(guard.rhs());
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => lhs != rhs,
                _ => true,
            }
        })
        .cloned()
        .collect();
    operation.shape_guards = kept.into_boxed_slice();
}

fn static_dim_value(expr: &DimExpr) -> Option<usize> {
    Some(match expr {
        DimExpr::Const(value) => *value,
        DimExpr::InputDim { .. } => return None,
        DimExpr::Add(lhs, rhs) => static_dim_value(lhs)?.checked_add(static_dim_value(rhs)?)?,
        DimExpr::Sub(lhs, rhs) => static_dim_value(lhs)?.checked_sub(static_dim_value(rhs)?)?,
        DimExpr::Mul(lhs, rhs) => static_dim_value(lhs)?.checked_mul(static_dim_value(rhs)?)?,
        DimExpr::FloorDiv(lhs, rhs) => {
            static_dim_value(lhs)?.checked_div(static_dim_value(rhs)?)?
        }
        DimExpr::Min(lhs, rhs) => static_dim_value(lhs)?.min(static_dim_value(rhs)?),
        DimExpr::Max(lhs, rhs) => static_dim_value(lhs)?.max(static_dim_value(rhs)?),
    })
}

fn snapshot_rebuild(source: ProgramBuildError) -> ProgramFinishError {
    ProgramFinishError::SnapshotRebuild { source }
}

fn core_output_uses_local_shape_coordinates(op: &CoreSemanticOp) -> bool {
    matches!(
        op,
        CoreSemanticOp::Reshape { .. }
            | CoreSemanticOp::BroadcastInDim { .. }
            | CoreSemanticOp::GatherDynamicSliceSizes { .. }
    )
}

/// Reject a core operation that names an externally defined scalar.
///
/// A core operation has no contribution-owned kernel for a scalar tenferro does not
/// declare, so no core operation may name one. Carrying an external scalar through a
/// program is what an extension operation is for.
pub(super) fn reject_core_external_dtype(op: &CoreSemanticOp) -> Result<(), ProgramBuildError> {
    let external = match op {
        CoreSemanticOp::Convert { from, to } => [Some(*from), Some(*to)]
            .into_iter()
            .flatten()
            .find(|dtype| matches!(dtype, tenferro_tensor::DType::External(_))),
        CoreSemanticOp::Constant { dtype, .. } => {
            matches!(dtype, tenferro_tensor::DType::External(_)).then_some(*dtype)
        }
        _ => None,
    };
    match external {
        Some(dtype) => Err(ProgramBuildError::ExternalScalarWithoutIdentity {
            dtype,
            site: "a core operation",
        }),
        None => Ok(()),
    }
}

/// Reject an external scalar the program cannot give a canonical identity.
///
/// A semantic program's identity must be reproducible across processes, and an
/// externally defined tag is a process-local `TypeId`. A program that carries one
/// declares the stable name, and a value without one is rejected explicitly instead
/// of being encoded as an unstable code.
pub(super) fn require_scalar_identity(
    metadata: &ProgramValueMetadata,
    site: &'static str,
) -> Result<(), ProgramBuildError> {
    match (metadata.dtype(), metadata.scalar_identity()) {
        (tenferro_tensor::DType::External(_), None) => {
            Err(ProgramBuildError::ExternalScalarWithoutIdentity {
                dtype: metadata.dtype(),
                site,
            })
        }
        _ => Ok(()),
    }
}

/// Keep the canonical identity a source program declared on a rebuilt value.
///
/// An imported value's metadata is resolved against the destination program, so it is
/// rebuilt from a dtype and shape; the declared identity of an externally defined
/// scalar is not derivable from those and has to be carried across.
fn keep_scalar_identity(
    identity: &Option<&'static str>,
    metadata: ProgramValueMetadata,
) -> ProgramValueMetadata {
    match identity {
        Some(identity) => metadata.with_scalar_identity(identity),
        None => metadata,
    }
}

fn validate_arity(expected: usize, actual: usize) -> Result<(), ProgramBuildError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ProgramBuildError::Arity { expected, actual })
    }
}

fn validate_output_count(expected: usize, actual: usize) -> Result<(), ProgramBuildError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ProgramBuildError::OutputMetadataCount { expected, actual })
    }
}

impl FrozenProgram {
    /// Return ordered input metadata after resolving bound input dimensions.
    ///
    /// Bound tensor shapes are process-local and intentionally live outside the
    /// semantic identity, but AD/runtime caches that prepare shape-specialized
    /// programs must distinguish those concrete shapes.
    #[doc(hidden)]
    pub fn input_metadata_with_bound_shapes(&self) -> Box<[ProgramValueMetadata]> {
        let bound_input_shapes: Vec<Option<Vec<DimExpr>>> = self
            .program
            .inputs
            .iter()
            .map(|input| {
                self.bindings.tensor_for_input(*input).map(|tensor| {
                    tensor
                        .shape()
                        .iter()
                        .map(|&size| DimExpr::Const(size))
                        .collect()
                })
            })
            .collect();

        self.program
            .inputs
            .iter()
            .map(|input| {
                let metadata = self.program.values[input.slot as usize].clone();
                ProgramValueMetadata::from_extents(
                    metadata.dtype(),
                    metadata.shape().iter().map(|extent| match extent {
                        ShapeExtent::Exact(expr) => ShapeExtent::Exact(
                            resolve_dim_expr_from_input_shapes(expr, &bound_input_shapes),
                        ),
                        ShapeExtent::UpperBound(expr) => ShapeExtent::UpperBound(
                            resolve_dim_expr_from_input_shapes(expr, &bound_input_shapes),
                        ),
                        ShapeExtent::Unknown => ShapeExtent::Unknown,
                    }),
                )
            })
            .collect()
    }

    /// Return a clone of this frozen program with tensor bindings copied from
    /// `source` onto this program's input prefix.
    ///
    /// This is intentionally narrow: semantic AD transforms import every source
    /// primal input first, then append derivative seed inputs. Cached derivative
    /// program structure can therefore be reused across source programs with the
    /// same normalized semantics while still carrying the current source's
    /// process-local tensor defaults.
    #[doc(hidden)]
    pub fn with_input_prefix_bindings_from(
        &self,
        source: &FrozenProgram,
    ) -> Result<FrozenProgram, ProgramFinishError> {
        if self.program.inputs.len() < source.program.inputs.len() {
            return Err(ProgramFinishError::StructuralValidation {
                source: ProgramStructuralError::InvalidValueReference,
            });
        }

        let mut bindings = Vec::new();
        for (source_input, destination_input) in
            source.program.inputs.iter().zip(self.program.inputs.iter())
        {
            if let Some(tensor) = source.bindings.tensor_for_input(*source_input) {
                bindings.push(PendingBinding {
                    key: BindingKey::new(destination_input.slot, self.program.owner),
                    input: *destination_input,
                    tensor,
                });
            }
        }

        let input_specs: Vec<_> = self
            .program
            .inputs
            .iter()
            .map(|input| {
                ProgramInputSpec::from_metadata(self.program.values[input.slot as usize].clone())
            })
            .collect();
        validate_bindings(&self.program.inputs, &input_specs, &bindings)?;

        Ok(FrozenProgram {
            program: Arc::clone(&self.program),
            bindings: ProgramBindings::freeze(self.program.owner, bindings),
        })
    }
}

struct ImportTransaction {
    inputs: Vec<ProgramValue>,
    input_specs: Vec<ProgramInputSpec>,
    values: Vec<ProgramValueMetadata>,
    operations: Vec<SemanticOperation>,
    bindings: Vec<PendingBinding>,
    roots: Box<[ProgramValue]>,
}

impl ImportTransaction {
    fn prepare(
        destination: &SemanticProgramBuilder,
        request: ProgramImport<'_>,
    ) -> Result<Self, ProgramBuildError> {
        let source = request.program;
        if let Some(bindings) = request.bindings
            && !bindings.belongs_to(source.owner)
        {
            return Err(ProgramBuildError::ForeignBindings);
        }
        for (mapped_source, mapped_destination) in request.input_map {
            if mapped_source.owner != source.owner
                || mapped_source.slot as usize >= source.values.len()
            {
                return Err(ProgramBuildError::ForeignValue);
            }
            destination.validate_value(*mapped_destination)?;
        }
        if request
            .roots
            .iter()
            .any(|root| root.owner != source.owner || root.slot as usize >= source.values.len())
        {
            return Err(ProgramBuildError::ForeignImportRoot);
        }

        let mut producer = vec![None; source.values.len()];
        for (operation_index, operation) in source.operations.iter().enumerate() {
            for output in &operation.outputs {
                producer[output.slot as usize] = Some(operation_index);
            }
        }

        let mut needed_values = vec![false; source.values.len()];
        let mut needed_operations = vec![false; source.operations.len()];
        let mut pending: Vec<_> = request
            .roots
            .iter()
            .map(|root| root.slot as usize)
            .collect();
        if let Some(bindings) = request.bindings {
            pending.extend(bindings.bound_inputs().map(|input| input.slot as usize));
        }
        for (operation_index, operation) in source.operations.iter().enumerate() {
            if !operation.effects.is_empty() {
                needed_operations[operation_index] = true;
                for output in &operation.outputs {
                    needed_values[output.slot as usize] = true;
                }
                pending.extend(operation.inputs.iter().map(|input| input.slot as usize));
            }
        }
        while let Some(slot) = pending.pop() {
            if needed_values[slot] {
                continue;
            }
            needed_values[slot] = true;
            if let Some(operation_index) = producer[slot]
                && !needed_operations[operation_index]
            {
                needed_operations[operation_index] = true;
                let operation = &source.operations[operation_index];
                for output in &operation.outputs {
                    needed_values[output.slot as usize] = true;
                }
                pending.extend(operation.inputs.iter().map(|input| input.slot as usize));
            }
        }

        // A kept value's declared extent and a kept operation's guard name source
        // inputs by position, which is a dependency the value graph does not carry.
        // Pull those inputs in, or the declaration would have to be dropped.
        loop {
            let mut positions = Vec::new();
            for input in &source.inputs {
                if !needed_values[input.slot as usize] {
                    continue;
                }
                for extent in source.values[input.slot as usize].shape() {
                    if let ShapeExtent::Exact(expr) | ShapeExtent::UpperBound(expr) = extent {
                        collect_input_positions(expr, &mut positions);
                    }
                }
            }
            for (operation, needed) in source.operations.iter().zip(&needed_operations) {
                if !needed {
                    continue;
                }
                for guard in operation.shape_guards.iter() {
                    collect_input_positions(guard.lhs(), &mut positions);
                    collect_input_positions(guard.rhs(), &mut positions);
                }
            }
            let mut changed = false;
            for position in positions {
                let Some(input) = source.inputs.get(position) else {
                    continue;
                };
                if !needed_values[input.slot as usize] {
                    needed_values[input.slot as usize] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let imported_input_count = source
            .inputs
            .iter()
            .filter(|input| needed_values[input.slot as usize])
            .count();
        let imported_output_count: usize = source
            .operations
            .iter()
            .zip(&needed_operations)
            .filter(|(_, needed)| **needed)
            .map(|(operation, _)| operation.outputs.len())
            .sum();
        let imported_value_count = imported_input_count
            .checked_add(imported_output_count)
            .ok_or(ProgramBuildError::TooManyValues)?;
        let final_value_count = destination
            .values
            .len()
            .checked_add(imported_value_count)
            .ok_or(ProgramBuildError::TooManyValues)?;
        if final_value_count > u32::MAX as usize {
            return Err(ProgramBuildError::TooManyValues);
        }

        let mut transaction = Self {
            inputs: Vec::with_capacity(imported_input_count),
            input_specs: Vec::with_capacity(imported_input_count),
            values: Vec::with_capacity(imported_value_count),
            operations: Vec::with_capacity(
                needed_operations.iter().filter(|needed| **needed).count(),
            ),
            bindings: Vec::new(),
            roots: Box::new([]),
        };
        // Imported metadata keeps the source's expressions: resolving `InputDim`
        // against the bound shapes would turn a symbolic program into a concrete one,
        // and an operation payload built from that metadata cannot be made symbolic
        // again. A rule that needs a concrete length reads it from the binding.
        let mapped_input: std::collections::HashMap<ProgramValue, ProgramValue> =
            request.input_map.iter().copied().collect();

        let mut remap = vec![None; source.values.len()];
        // Program dimension expressions name inputs by program input *position*,
        // and the destination already owns `destination.inputs.len()` inputs, so
        // the map is keyed by source position and holds destination positions.
        let mut source_input_index = vec![None; source.inputs.len()];

        // Reserve one destination input per needed source input first: an imported
        // declaration can name any imported input, so every position must be known
        // before any declaration is written in destination coordinates.
        let mut reserved = Vec::new();
        for (position, &input) in source.inputs.iter().enumerate() {
            if !needed_values[input.slot as usize] {
                continue;
            }
            if let Some(&existing) = mapped_input.get(&input) {
                remap[input.slot as usize] = Some(existing);
                continue;
            }
            let slot = destination
                .values
                .len()
                .checked_add(transaction.values.len())
                .and_then(|slot| slot.checked_add(reserved.len()))
                .ok_or(ProgramBuildError::TooManyValues)?;
            let slot = u32::try_from(slot).map_err(|_| ProgramBuildError::TooManyValues)?;
            let imported = ProgramValue::new(slot, destination.owner);
            source_input_index[position] = Some(destination.inputs.len() + reserved.len());
            let tensor = request
                .bindings
                .and_then(|bindings| bindings.tensor_for_input(input));
            reserved.push((imported, &source.values[input.slot as usize], tensor));
            remap[input.slot as usize] = Some(imported);
        }

        for (imported, source_metadata, tensor) in reserved {
            let identity = source_metadata.scalar_identity();
            let metadata = ProgramValueMetadata::from_extents(
                source_metadata.dtype(),
                source_metadata
                    .shape()
                    .iter()
                    .cloned()
                    .map(|extent| {
                        remap_imported_extent(
                            extent,
                            source,
                            &source_input_index,
                            &mapped_input,
                            destination,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let metadata = keep_scalar_identity(&identity, metadata);
            transaction.inputs.push(imported);
            transaction
                .input_specs
                .push(ProgramInputSpec::from_metadata(metadata.clone()));
            transaction.values.push(metadata);
            if let Some(tensor) = tensor {
                transaction.bindings.push(PendingBinding {
                    key: BindingKey::new(imported.slot, destination.owner),
                    input: imported,
                    tensor,
                });
            }
        }

        for (operation, needed) in source.operations.iter().zip(needed_operations) {
            if !needed {
                continue;
            }
            let inputs: Box<[_]> = operation
                .inputs
                .iter()
                .map(|input| {
                    remap[input.slot as usize].ok_or(ProgramBuildError::InvalidImport {
                        source: ProgramStructuralError::InvalidSsaOrder,
                    })
                })
                .collect::<Result<_, _>>()?;
            let mut outputs = Vec::with_capacity(operation.outputs.len());
            for output in &operation.outputs {
                let imported =
                    transaction.next_value(destination.values.len(), destination.owner)?;
                let meta = source.values[output.slot as usize].clone();
                let identity = meta.scalar_identity();
                let resolved = ProgramValueMetadata::from_extents(
                    meta.dtype(),
                    meta.shape()
                        .iter()
                        .cloned()
                        .map(|extent| {
                            remap_imported_extent(
                                extent,
                                source,
                                &source_input_index,
                                &mapped_input,
                                destination,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                transaction
                    .values
                    .push(keep_scalar_identity(&identity, resolved));
                remap[output.slot as usize] = Some(imported);
                outputs.push(imported);
            }
            let op = match &operation.op {
                SemanticOp::Core(op) => SemanticOp::Core(op.clone()),
                SemanticOp::Extension(op) => SemanticOp::Extension(op.clone_arc()),
            };
            let guards = operation
                .shape_guards
                .iter()
                .filter_map(|guard| {
                    let resolve = |expr: &DimExpr| -> Option<DimExpr> {
                        remap_imported_dim_expr(
                            expr,
                            source,
                            &source_input_index,
                            &mapped_input,
                            destination,
                        )
                    };
                    Some(ShapeGuard::new(
                        guard.relation(),
                        resolve(guard.lhs())?,
                        resolve(guard.rhs())?,
                    ))
                })
                .collect::<Vec<_>>();
            transaction.operations.push(SemanticOperation {
                op,
                inputs,
                outputs: outputs.into(),
                effects: operation.effects.clone(),
                aliases: operation.aliases.clone(),
                shape_guards: guards.into_boxed_slice(),
                placement: operation.placement,
                provenance: operation.provenance.clone(),
            });
        }

        transaction.roots = request
            .roots
            .iter()
            .map(|root| {
                remap[root.slot as usize].ok_or(ProgramBuildError::InvalidImport {
                    source: ProgramStructuralError::InvalidValueReference,
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(transaction)
    }

    fn next_value(
        &self,
        destination_value_count: usize,
        owner: ProgramBuilderNonce,
    ) -> Result<ProgramValue, ProgramBuildError> {
        let slot = destination_value_count
            .checked_add(self.values.len())
            .ok_or(ProgramBuildError::TooManyValues)?;
        let slot = u32::try_from(slot).map_err(|_| ProgramBuildError::TooManyValues)?;
        Ok(ProgramValue::new(slot, owner))
    }
}

#[derive(Clone, Copy)]
enum InputExtentPrecision {
    Exact,
    Bounded,
    Unknown,
}

/// Replace every `InputDim { input_idx: position, .. }` in `expr` with the
/// corresponding axis of `bound`.
///
/// Substitution is per input, so resolving an extent costs one pass over the
/// expression per referenced input and nothing proportional to the input count.
fn substitute_input_extents(expr: &DimExpr, position: usize, bound: &[usize]) -> DimExpr {
    match expr {
        DimExpr::Const(value) => DimExpr::Const(*value),
        DimExpr::InputDim { input_idx, axis } if *input_idx == position => bound
            .get(*axis)
            .map_or_else(|| expr.clone(), |size| DimExpr::Const(*size)),
        DimExpr::InputDim { .. } => expr.clone(),
        DimExpr::Add(lhs, rhs) => DimExpr::Add(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
        DimExpr::Sub(lhs, rhs) => DimExpr::Sub(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
        DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
        DimExpr::FloorDiv(lhs, rhs) => DimExpr::FloorDiv(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
        DimExpr::Min(lhs, rhs) => DimExpr::Min(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
        DimExpr::Max(lhs, rhs) => DimExpr::Max(
            Box::new(substitute_input_extents(lhs, position, bound)),
            Box::new(substitute_input_extents(rhs, position, bound)),
        ),
    }
}

/// Collect every program input position `expr` names.
fn collect_input_positions(expr: &DimExpr, positions: &mut Vec<usize>) {
    match expr {
        DimExpr::Const(_) => {}
        DimExpr::InputDim { input_idx, .. } => positions.push(*input_idx),
        DimExpr::Add(lhs, rhs)
        | DimExpr::Sub(lhs, rhs)
        | DimExpr::Mul(lhs, rhs)
        | DimExpr::FloorDiv(lhs, rhs)
        | DimExpr::Min(lhs, rhs)
        | DimExpr::Max(lhs, rhs) => {
            collect_input_positions(lhs, positions);
            collect_input_positions(rhs, positions);
        }
    }
}

/// Translate one imported extent into the destination input space.
///
/// An extent the destination cannot express becomes `Unknown` rather than keeping
/// source coordinates that would name a different destination input.
fn remap_imported_extent(
    extent: ShapeExtent<DimExpr>,
    source: &SemanticProgram,
    source_input_index: &[Option<usize>],
    mapped_input: &std::collections::HashMap<ProgramValue, ProgramValue>,
    destination: &SemanticProgramBuilder,
) -> ShapeExtent<DimExpr> {
    match extent {
        ShapeExtent::Exact(expr) => {
            remap_imported_dim_expr(&expr, source, source_input_index, mapped_input, destination)
                .map(ShapeExtent::Exact)
                .unwrap_or(ShapeExtent::Unknown)
        }
        ShapeExtent::UpperBound(expr) => {
            remap_imported_dim_expr(&expr, source, source_input_index, mapped_input, destination)
                .map(ShapeExtent::UpperBound)
                .unwrap_or(ShapeExtent::Unknown)
        }
        ShapeExtent::Unknown => ShapeExtent::Unknown,
    }
}

/// Rewrite one imported guard expression into the destination input space.
///
/// A kept source input keeps its own extent with the destination index; an input
/// that landed on an existing destination value is replaced by that value's
/// declared exact extent; an expression the destination cannot express makes the
/// guard unrepresentable, and the guard is dropped.
fn remap_imported_dim_expr(
    expr: &DimExpr,
    source: &SemanticProgram,
    source_input_index: &[Option<usize>],
    mapped_input: &std::collections::HashMap<ProgramValue, ProgramValue>,
    destination: &SemanticProgramBuilder,
) -> Option<DimExpr> {
    match expr {
        DimExpr::Const(value) => Some(DimExpr::Const(*value)),
        DimExpr::InputDim { input_idx, axis } => {
            if let Some(index) = source_input_index.get(*input_idx).copied().flatten() {
                return Some(DimExpr::InputDim {
                    input_idx: index,
                    axis: *axis,
                });
            }
            let source_input = source.inputs.get(*input_idx)?;
            let destination_value = mapped_input.get(source_input)?;
            match destination
                .values
                .get(destination_value.slot as usize)?
                .shape()
                .get(*axis)?
            {
                ShapeExtent::Exact(expr) => Some(expr.clone()),
                _ => None,
            }
        }
        DimExpr::Add(lhs, rhs) => Some(DimExpr::Add(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
        DimExpr::Sub(lhs, rhs) => Some(DimExpr::Sub(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
        DimExpr::Mul(lhs, rhs) => Some(DimExpr::Mul(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
        DimExpr::FloorDiv(lhs, rhs) => Some(DimExpr::FloorDiv(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
        DimExpr::Min(lhs, rhs) => Some(DimExpr::Min(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
        DimExpr::Max(lhs, rhs) => Some(DimExpr::Max(
            Box::new(remap_imported_dim_expr(
                lhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
            Box::new(remap_imported_dim_expr(
                rhs,
                source,
                source_input_index,
                mapped_input,
                destination,
            )?),
        )),
    }
}

fn input_extent_precision(metadata: &[&ProgramValueMetadata]) -> InputExtentPrecision {
    let mut precision = InputExtentPrecision::Exact;
    for extent in metadata.iter().flat_map(|metadata| metadata.shape()) {
        match extent {
            ShapeExtent::Unknown => return InputExtentPrecision::Unknown,
            ShapeExtent::UpperBound(_) => precision = InputExtentPrecision::Bounded,
            ShapeExtent::Exact(_) => {}
        }
    }
    precision
}

fn conservatively_bound_extents(
    extents: impl IntoIterator<Item = ShapeExtent<DimExpr>>,
    precision: InputExtentPrecision,
) -> impl Iterator<Item = ShapeExtent<DimExpr>> {
    extents.into_iter().map(move |extent| match precision {
        InputExtentPrecision::Exact => extent,
        InputExtentPrecision::Bounded => match extent {
            ShapeExtent::Exact(expression) | ShapeExtent::UpperBound(expression) => {
                ShapeExtent::UpperBound(expression)
            }
            ShapeExtent::Unknown => ShapeExtent::Unknown,
        },
        InputExtentPrecision::Unknown => ShapeExtent::Unknown,
    })
}

fn resolve_inferred_extents(
    extents: impl IntoIterator<Item = ShapeExtent<DimExpr>>,
    precision: InputExtentPrecision,
    input_shapes: &[&[DimExpr]],
) -> Result<Vec<ShapeExtent<DimExpr>>, ProgramBuildError> {
    extents
        .into_iter()
        .map(|extent| {
            if matches!(precision, InputExtentPrecision::Unknown) {
                return Ok(ShapeExtent::Unknown);
            }
            let resolved = match extent {
                ShapeExtent::Exact(expression) => ShapeExtent::Exact(
                    crate::shape_infer::resolve_dim_expr_from_shapes(&expression, input_shapes)
                        .map_err(metadata_error)?,
                ),
                ShapeExtent::UpperBound(expression) => ShapeExtent::UpperBound(
                    crate::shape_infer::resolve_dim_expr_from_shapes(&expression, input_shapes)
                        .map_err(metadata_error)?,
                ),
                ShapeExtent::Unknown => ShapeExtent::Unknown,
            };
            Ok(match precision {
                InputExtentPrecision::Exact => resolved,
                InputExtentPrecision::Bounded => match resolved {
                    ShapeExtent::Exact(expression) | ShapeExtent::UpperBound(expression) => {
                        ShapeExtent::UpperBound(expression)
                    }
                    ShapeExtent::Unknown => ShapeExtent::Unknown,
                },
                InputExtentPrecision::Unknown => unreachable!("handled above"),
            })
        })
        .collect()
}

fn inference_shapes(metadata: &[&ProgramValueMetadata]) -> Vec<Vec<DimExpr>> {
    metadata
        .iter()
        .enumerate()
        .map(|(input_idx, metadata)| {
            metadata
                .shape()
                .iter()
                .enumerate()
                .map(|(axis, extent)| match extent {
                    ShapeExtent::Exact(expression) | ShapeExtent::UpperBound(expression) => {
                        expression.clone()
                    }
                    ShapeExtent::Unknown => DimExpr::InputDim { input_idx, axis },
                })
                .collect()
        })
        .collect()
}

fn metadata_error(source: crate::Error) -> ProgramBuildError {
    ProgramBuildError::Metadata {
        source: Box::new(source),
    }
}

fn extension_effects(op: &dyn ExtensionOp) -> Result<Vec<Effect>, ProgramBuildError> {
    let family = op.family_id();
    let effects = match op.semantic_effects() {
        ExtensionEffectDeclaration::Undeclared => {
            return Err(ProgramBuildError::UndeclaredExtensionEffects { family })
        }
        ExtensionEffectDeclaration::Declared(effects) => effects,
    };
    effects
        .iter()
        .map(|effect| {
            let resource = EffectResource::new(effect.family, effect.key)
                .map_err(|source| ProgramBuildError::InvalidEffectResource { family, source })?;
            let access = match effect.access {
                ExtensionEffectAccess::Read => EffectAccess::Read,
                ExtensionEffectAccess::Write => EffectAccess::Write,
            };
            Ok(Effect::new(resource, access))
        })
        .collect()
}

fn extension_aliases(op: &dyn ExtensionOp) -> Result<Vec<Alias>, ProgramBuildError> {
    let family = op.family_id();
    match op.semantic_aliases() {
        ExtensionAliasDeclaration::Undeclared => {
            Err(ProgramBuildError::UndeclaredExtensionAliases { family })
        }
        ExtensionAliasDeclaration::AllFresh => {
            Ok((0..op.output_count()).map(Alias::fresh).collect())
        }
        ExtensionAliasDeclaration::Declared(aliases) => aliases
            .iter()
            .map(|alias| match *alias {
                ExtensionAlias::Fresh { output } => Ok(Alias::fresh(output)),
                ExtensionAlias::ViewOf { output, input } => Ok(Alias::view_of(output, input)),
                ExtensionAlias::MustAlias { output, input } => Ok(Alias::must_alias(output, input)),
                ExtensionAlias::ExternalAlias {
                    output,
                    family: resource_family,
                    key,
                } => EffectResource::new(resource_family, key)
                    .map(|resource| Alias::external(output, resource))
                    .map_err(|source| ProgramBuildError::InvalidEffectResource { family, source }),
            })
            .collect(),
    }
}

fn validate_aliases(
    aliases: &[Alias],
    input_count: usize,
    output_count: usize,
) -> Result<(), ProgramBuildError> {
    let mut seen = vec![false; output_count];
    for &alias in aliases {
        let output = alias.output();
        let input = alias.input();
        if output >= output_count || input.is_some_and(|input| input >= input_count) {
            return Err(ProgramBuildError::AliasOutOfBounds {
                output,
                output_count,
                input,
                input_count,
            });
        }
        if seen[output] {
            return Err(ProgramBuildError::AliasCoverage {
                expected: output_count,
                actual: seen.iter().filter(|&&present| present).count(),
            });
        }
        seen[output] = true;
    }
    let actual = seen.iter().filter(|&&present| present).count();
    if actual != output_count {
        return Err(ProgramBuildError::AliasCoverage {
            expected: output_count,
            actual,
        });
    }
    Ok(())
}

fn validate_structure(
    owner: ProgramBuilderNonce,
    inputs: &[ProgramValue],
    value_count: usize,
    operations: &[SemanticOperation],
) -> Result<(), ProgramFinishError> {
    let mut covered = vec![false; value_count];
    for input in inputs {
        if input.owner != owner
            || input.slot as usize >= value_count
            || covered[input.slot as usize]
        {
            return Err(ProgramFinishError::StructuralValidation {
                source: ProgramStructuralError::InvalidValueReference,
            });
        }
        covered[input.slot as usize] = true;
    }
    let mut previous_output = None;
    for operation in operations {
        let Some(first_output) = operation.outputs.first() else {
            if operation.inputs.iter().any(|value| {
                value.owner != owner
                    || value.slot as usize >= value_count
                    || !covered[value.slot as usize]
            }) {
                return Err(ProgramFinishError::StructuralValidation {
                    source: ProgramStructuralError::InvalidValueReference,
                });
            }
            continue;
        };
        let output_start = first_output.slot as usize;
        let valid_input = operation.inputs.iter().all(|value| {
            value.owner == owner
                && (value.slot as usize) < output_start
                && (value.slot as usize) < value_count
                && covered[value.slot as usize]
        });
        let valid_output = operation.outputs.iter().enumerate().all(|(offset, value)| {
            value.owner == owner
                && value.slot as usize == output_start + offset
                && (value.slot as usize) < value_count
                && !covered[value.slot as usize]
        });
        let ordered = previous_output.is_none_or(|previous| output_start > previous);
        if !valid_input || !valid_output || !ordered {
            let source = if operation
                .inputs
                .iter()
                .chain(operation.outputs.iter())
                .any(|value| value.owner != owner || value.slot as usize >= value_count)
            {
                ProgramStructuralError::InvalidValueReference
            } else {
                ProgramStructuralError::InvalidSsaOrder
            };
            return Err(ProgramFinishError::StructuralValidation { source });
        }
        for output in &operation.outputs {
            covered[output.slot as usize] = true;
        }
        previous_output = operation.outputs.last().map(|value| value.slot as usize);
    }
    if covered.iter().any(|covered| !covered) {
        return Err(ProgramFinishError::StructuralValidation {
            source: ProgramStructuralError::InvalidSsaOrder,
        });
    }
    Ok(())
}

fn validate_bindings(
    inputs: &[ProgramValue],
    input_specs: &[ProgramInputSpec],
    bindings: &[PendingBinding],
) -> Result<(), ProgramFinishError> {
    for binding in bindings {
        let input_index = inputs
            .iter()
            .position(|input| *input == binding.input)
            .ok_or(ProgramFinishError::BindingFinalization {
                source: ProgramBindingError::InvalidTarget,
            })?;
        let spec = &input_specs[input_index];
        let metadata = spec.metadata();
        let actual_dtype = binding.tensor.dtype();
        if actual_dtype != metadata.dtype() {
            return Err(ProgramFinishError::BindingFinalization {
                source: ProgramBindingError::DTypeMismatch {
                    expected: metadata.dtype(),
                    actual: actual_dtype,
                },
            });
        }
        let actual_shape = binding.tensor.shape();
        if actual_shape.len() != metadata.shape().len() {
            return Err(ProgramFinishError::BindingFinalization {
                source: ProgramBindingError::RankMismatch {
                    expected: metadata.shape().len(),
                    actual: actual_shape.len(),
                },
            });
        }
        for (axis, (extent, &actual)) in
            metadata.shape().iter().zip(actual_shape.iter()).enumerate()
        {
            match extent {
                ShapeExtent::Exact(DimExpr::Const(expected)) if *expected != actual => {
                    return Err(ProgramFinishError::BindingFinalization {
                        source: ProgramBindingError::ExactExtentMismatch {
                            axis,
                            expected: *expected,
                            actual,
                        },
                    });
                }
                ShapeExtent::UpperBound(DimExpr::Const(bound)) if actual > *bound => {
                    return Err(ProgramFinishError::BindingFinalization {
                        source: ProgramBindingError::UpperBoundExceeded {
                            axis,
                            bound: *bound,
                            actual,
                        },
                    });
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Resolve [`DimExpr::InputDim`] references using concrete bound-input shapes.
///
/// When an input has a known tensor binding, its concrete shape replaces the
/// symbolic `InputDim { input_idx, axis }` reference. References to unbound
/// inputs are left unchanged.
fn resolve_dim_expr_from_input_shapes(
    expr: &DimExpr,
    bound_input_shapes: &[Option<Vec<DimExpr>>],
) -> DimExpr {
    match expr {
        DimExpr::Const(_) => expr.clone(),
        DimExpr::InputDim { input_idx, axis } => {
            if let Some(Some(shape)) = bound_input_shapes.get(*input_idx)
                && let Some(dim) = shape.get(*axis)
            {
                return dim.clone();
            }
            expr.clone()
        }
        DimExpr::Add(a, b) => DimExpr::add(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
        DimExpr::Sub(a, b) => DimExpr::sub(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
        DimExpr::Mul(a, b) => DimExpr::mul(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
        DimExpr::FloorDiv(a, b) => DimExpr::floor_div(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
        DimExpr::Min(a, b) => DimExpr::min(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
        DimExpr::Max(a, b) => DimExpr::max(
            resolve_dim_expr_from_input_shapes(a, bound_input_shapes),
            resolve_dim_expr_from_input_shapes(b, bound_input_shapes),
        ),
    }
}
