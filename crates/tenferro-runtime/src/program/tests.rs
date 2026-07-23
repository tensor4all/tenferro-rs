use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAliasDeclaration, ExtensionEffect, ExtensionEffectAccess, ExtensionEffectDeclaration,
    ExtensionOp,
};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::sym_dim::SymDim;
use tenferro_ops::ExtensionShapeContext;
use tenferro_tensor::{DType, Tensor, TypedTensor};

use super::{
    Alias, CoreSemanticOp, Effect, EffectAccess, EffectResource, ProgramBuildError,
    ProgramFinishError, ProgramImport, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticProgramBuilder,
    SemanticProvenanceKind, ShapeGuard,
};

#[derive(Clone, Debug)]
struct IdentityExtension {
    declares_semantics: bool,
    effectful: bool,
}

impl ExtensionOp for IdentityExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.identity.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.declares_semantics));
        hasher.write_u8(u8::from(self.effectful));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|other| {
            other.declares_semantics == self.declares_semantics && other.effectful == self.effectful
        })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        context: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if self.effectful {
            let extent = context.input_axis(0, 0)?;
            context.require_equal(extent, SymDim::from(2))?;
        }
        Ok(vec![(
            context.input_dtype(0)?,
            context.input_shape(0)?.to_vec(),
        )])
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        if self.declares_semantics {
            if self.effectful {
                static EFFECTS: [ExtensionEffect; 1] = [ExtensionEffect {
                    family: "tenferro-tests.state.v1",
                    key: 0,
                    access: ExtensionEffectAccess::Write,
                }];
                ExtensionEffectDeclaration::Declared(&EFFECTS)
            } else {
                ExtensionEffectDeclaration::Declared(&[])
            }
        } else {
            ExtensionEffectDeclaration::Undeclared
        }
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        if self.declares_semantics {
            ExtensionAliasDeclaration::AllFresh
        } else {
            ExtensionAliasDeclaration::Undeclared
        }
    }
}

#[test]
fn tokens_and_metadata_reject_foreign_values_and_hide_identity() {
    let spec = ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]);
    let mut left = SemanticProgramBuilder::new();
    let right = SemanticProgramBuilder::new();
    let foreign = left.input(spec.clone()).unwrap();

    assert!(matches!(
        right.validate_value(foreign),
        Err(ProgramBuildError::ForeignValue)
    ));
    assert_eq!(format!("{foreign:?}"), "ProgramValue(<opaque>)");

    let metadata = spec.metadata();
    assert_eq!(metadata.dtype(), DType::F64);
    assert_eq!(metadata.shape(), &[ShapeExtent::Exact(DimExpr::Const(2))]);
}

#[test]
fn tokens_and_metadata_cover_guards_effects_aliases_and_placement() {
    let metadata = ProgramValueMetadata::new(
        DType::F32,
        [
            DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            DimExpr::Const(4),
        ],
    );
    assert_eq!(metadata.dtype(), DType::F32);
    assert_eq!(metadata.shape().len(), 2);
    let bounded = ProgramValueMetadata::from_extents(
        DType::F32,
        [
            ShapeExtent::upper_bound(DimExpr::Const(32)),
            ShapeExtent::Unknown,
        ],
    );
    assert_eq!(
        bounded.shape(),
        &[
            ShapeExtent::UpperBound(DimExpr::Const(32)),
            ShapeExtent::Unknown,
        ]
    );

    let guard = ShapeGuard::new(
        ProgramShapeRelation::LessEqual,
        DimExpr::InputDim {
            input_idx: 0,
            axis: 0,
        },
        DimExpr::Const(16),
    );
    assert_eq!(guard.relation(), ProgramShapeRelation::LessEqual);
    assert_eq!(guard.rhs(), &DimExpr::Const(16));

    let resource = EffectResource::new("example.random.v1", 7).unwrap();
    let read = Effect::new(resource, EffectAccess::Read);
    let write = Effect::new(resource, EffectAccess::Write);
    assert_eq!(read.resource(), resource);
    assert_eq!(write.access(), EffectAccess::Write);

    assert_eq!(Alias::fresh(0).output(), 0);
    assert_eq!(Alias::view_of(0, 1).input(), Some(1));
    assert_eq!(Alias::must_alias(0, 2).input(), Some(2));
    assert_eq!(Alias::external(0, resource).resource(), Some(resource));

    let placement = SemanticPlacementConstraint::same_as_input(1);
    assert_eq!(placement.input(), Some(1));
}

#[test]
fn operations_validate_arity_and_infer_ordered_metadata() {
    let mut builder = SemanticProgramBuilder::new();
    let x = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();

    assert!(matches!(
        builder.add_op(CoreSemanticOp::Add, &[x]),
        Err(ProgramBuildError::Arity {
            expected: 2,
            actual: 1,
        })
    ));

    let y = builder.add_op(CoreSemanticOp::Neg, &[x]).unwrap()[0];
    assert_eq!(builder.value_metadata(y).unwrap().dtype(), DType::F64);
    assert_eq!(
        builder.value_metadata(y).unwrap().shape(),
        &[ShapeExtent::Exact(DimExpr::Const(2))]
    );
    assert_eq!(builder.operation_count(), 1);
    let operation = builder.operation_views_for_test().next().unwrap();
    assert!(matches!(
        operation.op(),
        super::SemanticOpRef::Core(CoreSemanticOp::Neg)
    ));
    assert_eq!(operation.inputs(), &[x]);
    assert_eq!(operation.outputs(), &[y]);
    assert!(operation.effects().is_empty());
    assert_eq!(operation.aliases(), &[Alias::fresh(0)]);
    assert!(operation.shape_guards().is_empty());
    assert_eq!(operation.placement(), SemanticPlacementConstraint::any());
    assert_eq!(
        operation.provenance().kind(),
        SemanticProvenanceKind::Builder
    );
}

#[test]
fn extension_operations_require_explicit_effect_and_alias_declarations() {
    let mut builder = SemanticProgramBuilder::new();
    let x = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();

    let undeclared: Arc<dyn ExtensionOp> = Arc::new(IdentityExtension {
        declares_semantics: false,
        effectful: false,
    });
    assert!(matches!(
        builder.add_extension(undeclared, &[x]),
        Err(ProgramBuildError::UndeclaredExtensionEffects {
            family: "tenferro-tests.identity.v1",
        })
    ));
    assert_eq!(builder.operation_count(), 0);

    let declared: Arc<dyn ExtensionOp> = Arc::new(IdentityExtension {
        declares_semantics: true,
        effectful: false,
    });
    let y = builder.add_extension(declared, &[x]).unwrap()[0];
    assert_eq!(builder.value_metadata(y).unwrap().dtype(), DType::F64);
    assert_eq!(builder.operation_count(), 1);
}

#[test]
fn bindings_freeze_separately_and_reject_foreign_or_duplicate_inputs() {
    let tensor = Arc::new(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
    ));
    let spec = ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]);
    let mut builder = SemanticProgramBuilder::new();
    let input = builder.input(spec.clone()).unwrap();
    let key = builder.bind_input(input, Arc::clone(&tensor)).unwrap();

    assert!(matches!(
        builder.bind_input(input, Arc::clone(&tensor)),
        Err(ProgramBuildError::DuplicateBinding)
    ));

    let mut foreign_builder = SemanticProgramBuilder::new();
    let foreign_input = foreign_builder.input(spec).unwrap();
    let foreign_key = foreign_builder
        .bind_input(foreign_input, Arc::clone(&tensor))
        .unwrap();
    assert!(matches!(
        builder.bind_input(foreign_input, Arc::clone(&tensor)),
        Err(ProgramBuildError::ForeignValue)
    ));

    let frozen = builder.finish(&[input]).unwrap();
    assert_eq!(frozen.bindings.len(), 1);
    assert!(!frozen.bindings.is_empty());
    assert!(std::ptr::eq(
        frozen.bindings.get(key).unwrap(),
        tensor.as_ref()
    ));
    assert!(frozen.bindings.get(foreign_key).is_none());
    assert_eq!(frozen.bindings.iter().count(), 1);
    assert_eq!(format!("{key:?}"), "BindingKey(<opaque>)");
    assert_eq!(
        format!("{:?}", frozen.bindings),
        "ProgramBindings { len: 1 }"
    );
}

#[test]
fn finish_accepts_unbound_inputs_and_publishes_ordered_read_only_structure() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
    let frozen = builder.finish(&[output, input, output]).unwrap();

    assert!(frozen.bindings.is_empty());
    assert_eq!(frozen.program.inputs(), &[input]);
    assert_eq!(frozen.program.outputs(), &[output, input, output]);
    assert_eq!(frozen.program.operations().count(), 1);
    assert_eq!(
        frozen.program.value_metadata(output).unwrap().dtype(),
        DType::F64
    );
    assert_eq!(
        format!("{:?}", frozen.program),
        "SemanticProgram { inputs: 1, outputs: 3, values: 2, operations: 1, shape_guards: 0 }"
    );
}

#[test]
fn finish_rejects_foreign_outputs_before_publishing_either_frozen_half() {
    let mut builder = SemanticProgramBuilder::new();
    let local = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let mut other = SemanticProgramBuilder::new();
    let foreign = other
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();

    let result = builder.finish(&[local, foreign]);
    assert!(matches!(result, Err(ProgramFinishError::ForeignOutput)));
}

#[test]
fn finish_reports_binding_finalization_without_publishing_structure() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let wrong_dtype = Arc::new(Tensor::F32(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
    ));
    builder.bind_input(input, wrong_dtype).unwrap();

    let result = builder.finish(&[input]);
    assert!(matches!(
        result,
        Err(ProgramFinishError::BindingFinalization { .. })
    ));
}

#[test]
fn inputs_may_be_declared_between_operations_without_confusing_bindings() {
    let mut builder = SemanticProgramBuilder::new();
    let first = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[first]).unwrap()[0];
    let second = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let tensor = Arc::new(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap(),
    ));
    let key = builder.bind_input(second, Arc::clone(&tensor)).unwrap();

    let frozen = builder.finish(&[output, second]).unwrap();
    assert_eq!(frozen.program.inputs(), &[first, second]);
    assert!(std::ptr::eq(
        frozen.bindings.get(key).unwrap(),
        tensor.as_ref()
    ));
}

#[test]
fn import_preserves_ordered_duplicate_roots_structure_binding_and_provenance() {
    let tensor = Arc::new(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![5.0, 6.0]).unwrap(),
    ));
    let mut source = SemanticProgramBuilder::new();
    let x = source
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let y = source.add_op(CoreSemanticOp::Neg, &[x]).unwrap()[0];
    let source_key = source.bind_input(x, Arc::clone(&tensor)).unwrap();
    let source = source.finish(&[y, x]).unwrap();

    let mut destination = SemanticProgramBuilder::new();
    let imported = destination
        .import(ProgramImport {
            program: source.program.as_ref(),
            bindings: &source.bindings,
            roots: &[y, x, y],
        })
        .unwrap();
    assert_eq!(imported.roots().len(), 3);
    assert_eq!(imported.roots()[0], imported.roots()[2]);
    assert_ne!(imported.roots()[0], imported.roots()[1]);

    let frozen = destination.finish(imported.roots()).unwrap();
    assert_eq!(frozen.program.inputs().len(), 1);
    assert_eq!(frozen.program.operations().count(), 1);
    assert_eq!(
        frozen
            .program
            .operations()
            .next()
            .unwrap()
            .provenance()
            .kind(),
        source
            .program
            .operations()
            .next()
            .unwrap()
            .provenance()
            .kind()
    );
    assert_eq!(frozen.bindings.len(), 1);
    let imported_key = frozen.bindings.iter().next().unwrap().0;
    assert!(std::ptr::eq(
        frozen.bindings.get(imported_key).unwrap(),
        source.bindings.get(source_key).unwrap()
    ));
}

#[test]
fn import_empty_roots_is_a_noop_and_foreign_roots_roll_back() {
    let mut source = SemanticProgramBuilder::new();
    let source_input = source
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let source = source.finish(&[source_input]).unwrap();

    let mut other = SemanticProgramBuilder::new();
    let foreign = other
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();

    let mut destination = SemanticProgramBuilder::new();
    let local = destination
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let empty = destination
        .import(ProgramImport {
            program: source.program.as_ref(),
            bindings: &source.bindings,
            roots: &[],
        })
        .unwrap();
    assert!(empty.roots().is_empty());
    assert!(matches!(
        destination.import(ProgramImport {
            program: source.program.as_ref(),
            bindings: &source.bindings,
            roots: &[source_input, foreign],
        }),
        Err(ProgramBuildError::ForeignImportRoot)
    ));

    let frozen = destination.finish(&[local]).unwrap();
    assert_eq!(frozen.program.inputs(), &[local]);
    assert_eq!(frozen.program.operations().count(), 0);
    assert!(frozen.bindings.is_empty());
}

#[test]
fn import_rejects_bindings_from_another_frozen_program() {
    let mut left = SemanticProgramBuilder::new();
    let left_input = left
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let left = left.finish(&[left_input]).unwrap();

    let mut right = SemanticProgramBuilder::new();
    let right_input = right
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let right = right.finish(&[right_input]).unwrap();

    let mut destination = SemanticProgramBuilder::new();
    assert!(matches!(
        destination.import(ProgramImport {
            program: left.program.as_ref(),
            bindings: &right.bindings,
            roots: &[left_input],
        }),
        Err(ProgramBuildError::ForeignBindings)
    ));
}

#[test]
fn import_uses_value_dependency_closure_and_keeps_observable_effects() {
    let mut source = SemanticProgramBuilder::new();
    let selected = source
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let unrelated = source
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let selected_output = source.add_op(CoreSemanticOp::Neg, &[selected]).unwrap()[0];
    let _unrelated_output = source.add_op(CoreSemanticOp::Neg, &[unrelated]).unwrap()[0];
    let effect: Arc<dyn ExtensionOp> = Arc::new(IdentityExtension {
        declares_semantics: true,
        effectful: true,
    });
    let _effect_output = source.add_extension(effect, &[unrelated]).unwrap()[0];
    let source = source.finish(&[selected_output]).unwrap();

    let mut destination = SemanticProgramBuilder::new();
    let imported = destination
        .import(ProgramImport {
            program: source.program.as_ref(),
            bindings: &source.bindings,
            roots: &[selected_output],
        })
        .unwrap();
    let frozen = destination.finish(imported.roots()).unwrap();

    assert_eq!(frozen.program.inputs().len(), 2);
    assert_eq!(frozen.program.operations().count(), 2);
    assert_eq!(frozen.program.shape_guards().len(), 1);
    assert_eq!(
        frozen
            .program
            .operations()
            .filter(|operation| !operation.effects().is_empty())
            .count(),
        1
    );
}
