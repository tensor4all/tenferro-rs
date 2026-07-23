use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAlias, ExtensionAliasDeclaration, ExtensionEffectAccess, ExtensionEffectDeclaration,
    ExtensionOp,
};
use tenferro_ops::shape_extent::ShapeExtent;

use super::op::{SemanticOp, SemanticOperation};
use super::value::ProgramBuilderNonce;
use super::{
    Alias, CoreSemanticOp, Effect, EffectAccess, EffectResource, ProgramBuildError,
    ProgramInputSpec, ProgramShapeRelation, ProgramValue, ProgramValueMetadata,
    SemanticPlacementConstraint, ShapeGuard,
};

/// Mutable validation boundary for one semantic program.
pub struct SemanticProgramBuilder {
    owner: ProgramBuilderNonce,
    input_specs: Vec<ProgramInputSpec>,
    values: Vec<ProgramValueMetadata>,
    operations: Vec<SemanticOperation>,
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
            input_specs: Vec::new(),
            values: Vec::new(),
            operations: Vec::new(),
        }
    }

    /// Add one ordered external input.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramBuildError::TooManyValues`] if the builder cannot
    /// represent another value slot.
    pub fn input(&mut self, spec: ProgramInputSpec) -> Result<ProgramValue, ProgramBuildError> {
        let slot = self.next_value_slot()?;
        self.values.push(spec.metadata().clone());
        self.input_specs.push(spec);
        Ok(ProgramValue::new(slot, self.owner))
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
        let input_dtypes: Vec<_> = input_metadata
            .iter()
            .map(|metadata| metadata.dtype())
            .collect();
        let input_shapes = inference_shapes(&input_metadata);
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let standard = tenferro_ops::std_tensor_op::StdTensorOp::from(op);
        let dtype = crate::shape_infer::infer_output_dtype(&standard, &input_dtypes)
            .map_err(metadata_error)?;
        let output_extents = crate::shape_infer::infer_output_extents(&standard, &input_shape_refs)
            .map_err(metadata_error)?;
        Ok(output_extents
            .into_iter()
            .map(|shape| ProgramValueMetadata::from_extents(dtype, shape))
            .collect())
    }

    fn infer_extension_metadata(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[ProgramValue],
    ) -> Result<(Vec<ProgramValueMetadata>, Vec<ShapeGuard>), ProgramBuildError> {
        let input_metadata = self.input_metadata(inputs)?;
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
            .map(|(dtype, shape)| ProgramValueMetadata::new(dtype, shape))
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
        });
        Ok(outputs)
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
