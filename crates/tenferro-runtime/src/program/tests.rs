use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::sym_dim::SymDim;
use tenferro_ops::ExtensionShapeContext;
use tenferro_tensor::{DType, Tensor, TypedTensor};

use super::{
    Alias, CoreSemanticOp, Effect, EffectAccess, EffectResource, ProgramBuildError,
    ProgramFinishError, ProgramInputSpec, ProgramShapeRelation, ProgramValueMetadata,
    SemanticPlacementConstraint, SemanticProgramBuilder, SemanticProvenanceKind, ShapeGuard,
};

#[derive(Clone, Debug)]
struct IdentityExtension {
    declares_semantics: bool,
}

impl ExtensionOp for IdentityExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.identity.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.declares_semantics));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.declares_semantics == self.declares_semantics)
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
        Ok(vec![(
            context.input_dtype(0)?,
            context.input_shape(0)?.to_vec(),
        )])
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        if self.declares_semantics {
            ExtensionEffectDeclaration::Declared(&[])
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
