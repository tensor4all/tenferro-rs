use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAlias, ExtensionAliasDeclaration, ExtensionEffectAccess, ExtensionEffectDeclaration,
    ExtensionOp,
};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::Tensor;

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
        tensor: Arc<Tensor>,
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

fn core_output_uses_local_shape_coordinates(op: &CoreSemanticOp) -> bool {
    matches!(
        op,
        CoreSemanticOp::Reshape { .. }
            | CoreSemanticOp::BroadcastInDim { .. }
            | CoreSemanticOp::GatherDynamicSliceSizes { .. }
    )
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
        if !request.bindings.belongs_to(source.owner) {
            return Err(ProgramBuildError::ForeignBindings);
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
        pending.extend(
            request
                .bindings
                .bound_inputs()
                .map(|input| input.slot as usize),
        );
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
            if let Some(operation_index) = producer[slot] {
                if !needed_operations[operation_index] {
                    needed_operations[operation_index] = true;
                    let operation = &source.operations[operation_index];
                    for output in &operation.outputs {
                        needed_values[output.slot as usize] = true;
                    }
                    pending.extend(operation.inputs.iter().map(|input| input.slot as usize));
                }
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
        // Pre-resolve concrete input shapes from tensor bindings for InputDim
        // resolution in imported metadata.
        let bound_input_shapes: Vec<Option<Vec<DimExpr>>> = source
            .inputs
            .iter()
            .map(|input| {
                request.bindings.tensor_for_input(*input).map(|tensor| {
                    tensor
                        .shape()
                        .iter()
                        .map(|&size| DimExpr::Const(size))
                        .collect()
                })
            })
            .collect();

        let resolve_extent = |extent: &ShapeExtent<DimExpr>| -> ShapeExtent<DimExpr> {
            match extent {
                ShapeExtent::Exact(expr) => ShapeExtent::Exact(resolve_dim_expr_from_input_shapes(
                    expr,
                    &bound_input_shapes,
                )),
                ShapeExtent::UpperBound(expr) => ShapeExtent::UpperBound(
                    resolve_dim_expr_from_input_shapes(expr, &bound_input_shapes),
                ),
                ShapeExtent::Unknown => ShapeExtent::Unknown,
            }
        };

        let mut remap = vec![None; source.values.len()];

        for &input in &source.inputs {
            if !needed_values[input.slot as usize] {
                continue;
            }
            let metadata = source.values[input.slot as usize].clone();
            let metadata = ProgramValueMetadata::from_extents(
                metadata.dtype(),
                metadata
                    .shape()
                    .iter()
                    .map(|extent| resolve_extent(extent))
                    .collect::<Vec<_>>(),
            );
            let imported = transaction.next_value(destination.values.len(), destination.owner)?;
            transaction.inputs.push(imported);
            transaction
                .input_specs
                .push(ProgramInputSpec::from_metadata(metadata.clone()));
            transaction.values.push(metadata);
            remap[input.slot as usize] = Some(imported);
            if let Some(tensor) = request.bindings.tensor_for_input(input) {
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
                let resolved = ProgramValueMetadata::from_extents(
                    meta.dtype(),
                    meta.shape()
                        .iter()
                        .map(|extent| resolve_extent(extent))
                        .collect::<Vec<_>>(),
                );
                transaction.values.push(resolved);
                remap[output.slot as usize] = Some(imported);
                outputs.push(imported);
            }
            let op = match &operation.op {
                SemanticOp::Core(op) => SemanticOp::Core(op.clone()),
                SemanticOp::Extension(op) => SemanticOp::Extension(op.clone_arc()),
            };
            transaction.operations.push(SemanticOperation {
                op,
                inputs,
                outputs: outputs.into(),
                effects: operation.effects.clone(),
                aliases: operation.aliases.clone(),
                shape_guards: operation.shape_guards.clone(),
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
            if let Some(Some(shape)) = bound_input_shapes.get(*input_idx) {
                if let Some(dim) = shape.get(*axis) {
                    return dim.clone();
                }
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
