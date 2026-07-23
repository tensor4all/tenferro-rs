use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::DType;

use super::{
    Alias, Effect, EffectAccess, EffectResource, ProgramBuildError, ProgramInputSpec,
    ProgramShapeRelation, ProgramValueMetadata, SemanticPlacementConstraint,
    SemanticProgramBuilder, ShapeGuard,
};

#[test]
fn tokens_and_metadata_reject_foreign_values_and_hide_identity() {
    let spec = ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]);
    let mut left = SemanticProgramBuilder::new();
    let right = SemanticProgramBuilder::new();
    let foreign = left.input(spec.clone()).unwrap();

    assert_eq!(
        right.validate_value(foreign),
        Err(ProgramBuildError::ForeignValue)
    );
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
