use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAlias, ExtensionAliasDeclaration, ExtensionEffect, ExtensionEffectAccess,
    ExtensionEffectDeclaration, ExtensionOp,
};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::sym_dim::SymDim;
use tenferro_ops::ExtensionShapeContext;
use tenferro_tensor::{DType, Tensor, TypedTensor};

use super::{
    Alias, CoreSemanticOp, Effect, EffectAccess, EffectResource, ProgramBuildError,
    ProgramFinishError, ProgramImport, ProgramInputSpec, ProgramShapeRelation,
    ProgramValueMetadata, SemanticPlacementConstraint, SemanticProgramBuilder,
    SemanticProvenanceKind, SemanticTransform, SemanticTransformContext, SemanticTransformError,
    ShapeGuard, TransformIdentity,
};

#[derive(Clone, Debug)]
struct IdentityExtension {
    declares_semantics: bool,
    effectful: bool,
    guarded: bool,
    view_alias: bool,
}

impl ExtensionOp for IdentityExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.identity.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.declares_semantics));
        hasher.write_u8(u8::from(self.effectful));
        hasher.write_u8(u8::from(self.guarded));
        hasher.write_u8(u8::from(self.view_alias));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|other| {
            other.declares_semantics == self.declares_semantics
                && other.effectful == self.effectful
                && other.guarded == self.guarded
                && other.view_alias == self.view_alias
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
        if self.guarded {
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
            if self.view_alias {
                static ALIASES: [ExtensionAlias; 1] = [ExtensionAlias::ViewOf {
                    output: 0,
                    input: 0,
                }];
                ExtensionAliasDeclaration::Declared(&ALIASES)
            } else {
                ExtensionAliasDeclaration::AllFresh
            }
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
fn operation_metadata_preserves_bounded_and_unknown_input_guarantees() {
    let mut bounded_builder = SemanticProgramBuilder::new();
    let bounded = bounded_builder
        .input(ProgramInputSpec::from_metadata(
            ProgramValueMetadata::from_extents(
                DType::F64,
                [ShapeExtent::UpperBound(DimExpr::Const(32))],
            ),
        ))
        .unwrap();
    let bounded_output = bounded_builder
        .add_op(CoreSemanticOp::Neg, &[bounded])
        .unwrap()[0];
    assert_eq!(
        bounded_builder
            .value_metadata(bounded_output)
            .unwrap()
            .shape(),
        &[ShapeExtent::UpperBound(DimExpr::Const(32))]
    );

    let mut unknown_builder = SemanticProgramBuilder::new();
    let unknown = unknown_builder
        .input(ProgramInputSpec::from_metadata(
            ProgramValueMetadata::from_extents(DType::F64, [ShapeExtent::Unknown]),
        ))
        .unwrap();
    let unknown_output = unknown_builder
        .add_op(CoreSemanticOp::Neg, &[unknown])
        .unwrap()[0];
    assert_eq!(
        unknown_builder
            .value_metadata(unknown_output)
            .unwrap()
            .shape(),
        &[ShapeExtent::Unknown]
    );
}

#[test]
fn operation_metadata_resolves_local_shape_axes_to_program_inputs() {
    let mut builder = SemanticProgramBuilder::new();
    let _first = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }],
        ))
        .unwrap();
    let second = builder
        .input(ProgramInputSpec::new(
            DType::F64,
            [DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        ))
        .unwrap();

    let output = builder
        .add_op(
            CoreSemanticOp::Reshape {
                to_shape: vec![DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }],
            },
            &[second],
        )
        .unwrap()[0];

    assert_eq!(
        builder.value_metadata(output).unwrap().shape(),
        &[ShapeExtent::Exact(DimExpr::InputDim {
            input_idx: 1,
            axis: 0,
        })]
    );
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
        guarded: false,
        view_alias: false,
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
        guarded: false,
        view_alias: false,
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
        guarded: true,
        view_alias: false,
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

#[test]
fn semantic_identity_is_normalized_cached_and_excludes_bindings() {
    fn build(with_binding: bool) -> super::FrozenProgram {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        if with_binding {
            let tensor = Arc::new(Tensor::F64(
                TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap(),
            ));
            builder.bind_input(input, tensor).unwrap();
        }
        let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
        builder.finish(&[output]).unwrap()
    }

    let unbound = build(false);
    let bound = build(true);
    assert_eq!(
        unbound.program.semantic_fingerprint(),
        bound.program.semantic_fingerprint()
    );
    assert!(unbound.program.semantic_eq(bound.program.as_ref()));
    assert_eq!(
        unbound.program.semantic_fingerprint().as_bytes(),
        unbound.program.semantic_fingerprint().as_bytes()
    );
    assert_eq!(unbound.program.fingerprint_computations_for_test(), 1);
    assert!(unbound.bindings.is_empty());
    assert_eq!(bound.bindings.len(), 1);

    let mut imported = SemanticProgramBuilder::new();
    let roots = imported
        .import(ProgramImport {
            program: bound.program.as_ref(),
            bindings: &bound.bindings,
            roots: bound.program.outputs(),
        })
        .unwrap();
    let imported = imported.finish(roots.roots()).unwrap();
    assert!(bound.program.semantic_eq(imported.program.as_ref()));

    let mut changed_provenance = build(false);
    Arc::get_mut(&mut changed_provenance.program)
        .unwrap()
        .set_first_provenance_for_test("diagnostic-only");
    assert_eq!(
        unbound.program.semantic_fingerprint(),
        changed_provenance.program.semantic_fingerprint()
    );
    assert!(unbound
        .program
        .semantic_eq(changed_provenance.program.as_ref()));
}

#[test]
fn semantic_identity_detects_changes_even_after_a_forced_fingerprint_collision() {
    let mut left = SemanticProgramBuilder::new();
    let left_input = left
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let left_output = left.add_op(CoreSemanticOp::Neg, &[left_input]).unwrap()[0];
    let left = left.finish(&[left_output]).unwrap();

    let mut right = SemanticProgramBuilder::new();
    let right_input = right
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let right_output = right.add_op(CoreSemanticOp::Abs, &[right_input]).unwrap()[0];
    let mut right = right.finish(&[right_output]).unwrap();

    assert_ne!(
        left.program.semantic_fingerprint(),
        right.program.semantic_fingerprint()
    );
    Arc::get_mut(&mut right.program)
        .unwrap()
        .set_fingerprint_for_test(left.program.semantic_fingerprint());
    assert_eq!(
        left.program.semantic_fingerprint(),
        right.program.semantic_fingerprint()
    );
    assert!(!left.program.semantic_eq(right.program.as_ref()));
}

#[test]
fn semantic_identity_covers_guards_effects_aliases_and_constants() {
    fn build_extension(effectful: bool, guarded: bool, view_alias: bool) -> super::FrozenProgram {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
            .unwrap();
        let extension: Arc<dyn ExtensionOp> = Arc::new(IdentityExtension {
            declares_semantics: true,
            effectful,
            guarded,
            view_alias,
        });
        let output = builder.add_extension(extension, &[input]).unwrap()[0];
        builder.finish(&[output]).unwrap()
    }

    let plain = build_extension(false, false, false);
    let effectful = build_extension(true, false, false);
    let guarded = build_extension(false, true, false);
    let view = build_extension(false, false, true);
    for changed in [&effectful, &guarded, &view] {
        assert_ne!(
            plain.program.semantic_fingerprint(),
            changed.program.semantic_fingerprint()
        );
        assert!(!plain.program.semantic_eq(changed.program.as_ref()));
    }

    fn build_constant(byte: u8) -> super::FrozenProgram {
        let mut builder = SemanticProgramBuilder::new();
        let output = builder
            .add_op(
                CoreSemanticOp::Constant {
                    dtype: DType::Bool,
                    bytes: vec![byte],
                },
                &[],
            )
            .unwrap()[0];
        builder.finish(&[output]).unwrap()
    }
    let zero = build_constant(0);
    let one = build_constant(1);
    assert_ne!(
        zero.program.semantic_fingerprint(),
        one.program.semantic_fingerprint()
    );
    assert!(!zero.program.semantic_eq(one.program.as_ref()));
}

#[test]
fn shape_guard_diagnostic_family_is_preserved_but_excluded_from_identity() {
    fn guarded(family: &'static str) -> super::FrozenProgram {
        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(
                DType::F64,
                [DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }],
            ))
            .unwrap();
        let output = builder.add_op(CoreSemanticOp::Neg, &[input]).unwrap()[0];
        builder
            .add_shape_guards_to_output(
                output,
                [ShapeGuard::new(
                    ProgramShapeRelation::Equal,
                    DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    },
                    DimExpr::Const(2),
                )
                .with_source_family(family)],
            )
            .unwrap();
        builder.finish(&[output]).unwrap()
    }

    let left = guarded("tenferro-tests.guard-left.v1");
    let right = guarded("tenferro-tests.guard-right.v1");

    assert_eq!(
        left.program.shape_guards()[0].source_family(),
        Some("tenferro-tests.guard-left.v1")
    );
    assert_eq!(
        right.program.shape_guards()[0].source_family(),
        Some("tenferro-tests.guard-right.v1")
    );
    assert_eq!(
        left.program.semantic_fingerprint(),
        right.program.semantic_fingerprint()
    );
    assert!(left.program.semantic_eq(&right.program));
}

struct IdentityTransform {
    identity: TransformIdentity,
}

impl SemanticTransform for IdentityTransform {
    fn identity(&self) -> TransformIdentity {
        self.identity
    }

    fn apply(
        &self,
        context: &mut SemanticTransformContext<'_>,
        input: &super::FrozenProgram,
    ) -> Result<Box<[super::ProgramValue]>, SemanticTransformError> {
        Ok(context
            .import_program(input, input.program.outputs())?
            .roots()
            .into())
    }
}

struct AppendTransform {
    identity: TransformIdentity,
    op: CoreSemanticOp,
}

impl SemanticTransform for AppendTransform {
    fn identity(&self) -> TransformIdentity {
        self.identity
    }

    fn apply(
        &self,
        context: &mut SemanticTransformContext<'_>,
        input: &super::FrozenProgram,
    ) -> Result<Box<[super::ProgramValue]>, SemanticTransformError> {
        let roots: Box<[_]> = context
            .import_program(input, input.program.outputs())?
            .roots()
            .into();
        let output = context.builder().add_op(self.op.clone(), &[roots[0]])?[0];
        Ok(Box::new([output]))
    }
}

struct ForeignOutputTransform {
    identity: TransformIdentity,
}

struct DroppingBindingTransform {
    identity: TransformIdentity,
}

impl SemanticTransform for DroppingBindingTransform {
    fn identity(&self) -> TransformIdentity {
        self.identity
    }

    fn apply(
        &self,
        context: &mut SemanticTransformContext<'_>,
        _input: &super::FrozenProgram,
    ) -> Result<Box<[super::ProgramValue]>, SemanticTransformError> {
        let replacement = context
            .builder()
            .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))?;
        Ok(Box::new([replacement]))
    }
}

impl SemanticTransform for ForeignOutputTransform {
    fn identity(&self) -> TransformIdentity {
        self.identity
    }

    fn apply(
        &self,
        _context: &mut SemanticTransformContext<'_>,
        input: &super::FrozenProgram,
    ) -> Result<Box<[super::ProgramValue]>, SemanticTransformError> {
        Ok(input.program.outputs().into())
    }
}

#[test]
fn semantic_transform_is_object_safe_and_identity_preserves_unused_bindings() {
    fn assert_object_safe(_: Arc<dyn SemanticTransform>) {}
    let transform = Arc::new(IdentityTransform {
        identity: TransformIdentity::from_bytes([1; 16]),
    });
    assert_object_safe(transform.clone());

    let mut builder = SemanticProgramBuilder::new();
    let used = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let unused_bound = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let tensor = Arc::new(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![7.0, 8.0]).unwrap(),
    ));
    builder
        .bind_input(unused_bound, Arc::clone(&tensor))
        .unwrap();
    let output = builder.add_op(CoreSemanticOp::Neg, &[used]).unwrap()[0];
    let input = builder.finish(&[output]).unwrap();

    let transformed =
        super::transform::apply_semantic_transform(&input, transform.as_ref()).unwrap();
    assert!(input.program.semantic_eq(transformed.program.as_ref()));
    assert_eq!(transformed.bindings.len(), 1);
    assert!(std::ptr::eq(
        transformed.bindings.iter().next().unwrap().1,
        tensor.as_ref()
    ));

    let dropping = DroppingBindingTransform {
        identity: TransformIdentity::from_bytes([9; 16]),
    };
    assert!(matches!(
        super::transform::apply_semantic_transform(&input, &dropping),
        Err(SemanticTransformError::DroppedBindings)
    ));
}

#[test]
fn semantic_transform_pipeline_is_ordered_and_rejects_foreign_results() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let input = builder.finish(&[input]).unwrap();
    let neg = AppendTransform {
        identity: TransformIdentity::from_bytes([2; 16]),
        op: CoreSemanticOp::Neg,
    };
    let abs = AppendTransform {
        identity: TransformIdentity::from_bytes([3; 16]),
        op: CoreSemanticOp::Abs,
    };
    let forward = super::transform::apply_semantic_transforms(&input, &[&neg, &abs]).unwrap();
    let reverse = super::transform::apply_semantic_transforms(&input, &[&abs, &neg]).unwrap();
    assert!(!forward.program.semantic_eq(reverse.program.as_ref()));

    let foreign = ForeignOutputTransform {
        identity: TransformIdentity::from_bytes([4; 16]),
    };
    assert!(matches!(
        super::transform::apply_semantic_transform(&input, &foreign),
        Err(SemanticTransformError::ForeignReturnedValue)
    ));
}

#[test]
fn transform_cache_checks_exact_input_on_collision_and_stays_unchanged_on_failure() {
    let mut left = SemanticProgramBuilder::new();
    let left_input = left
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let left_output = left.add_op(CoreSemanticOp::Neg, &[left_input]).unwrap()[0];
    let left = left.finish(&[left_output]).unwrap();

    let mut right = SemanticProgramBuilder::new();
    let right_input = right
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let right_output = right.add_op(CoreSemanticOp::Abs, &[right_input]).unwrap()[0];
    let mut right = right.finish(&[right_output]).unwrap();
    Arc::get_mut(&mut right.program)
        .unwrap()
        .set_fingerprint_for_test(left.program.semantic_fingerprint());

    let identity = IdentityTransform {
        identity: TransformIdentity::from_bytes([5; 16]),
    };
    let mut cache = super::transform::SemanticTransformCache::new();
    let cached_left = cache.apply(&left, &[&identity]).unwrap();
    let cached_right = cache.apply(&right, &[&identity]).unwrap();
    assert!(!cached_left
        .program
        .semantic_eq(cached_right.program.as_ref()));
    assert_eq!(cache.len(), 2);

    let foreign = ForeignOutputTransform {
        identity: TransformIdentity::from_bytes([6; 16]),
    };
    assert!(cache.apply(&left, &[&foreign]).is_err());
    assert_eq!(cache.len(), 2);
}
