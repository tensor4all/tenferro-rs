use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::sym_dim::SymDim;
use tenferro_ops::ExtensionShapeContext;
use tenferro_tensor::DType;

use super::{
    Alias, CoreSemanticOp, Effect, EffectAccess, EffectResource, ProgramBuildError,
    ProgramInputSpec, ProgramShapeRelation, ProgramValueMetadata, SemanticPlacementConstraint,
    SemanticProgramBuilder, ShapeGuard,
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
