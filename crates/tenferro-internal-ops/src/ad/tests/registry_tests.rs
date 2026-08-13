use crate::ad::transpose_input::TransposeInputRef;
use crate::ad::{ADRuleKind, PrimitiveTransposeInput, ResidualSpec};
use computegraph::graph::GraphBuilder;
use computegraph::types::OperationRole;
use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};

use crate::ad::context::ShapeGuardContext;
use crate::ad::registry::primitive_ad_rule;
use crate::std_tensor_op::StdTensorOp;

#[test]
#[should_panic(expected = "residual mask")]
fn residual_mask_detector_rejects_undeclared_input_access() {
    // The undeclared-access detector must fail when a transpose rule reads a
    // tensor value its residual mask did not declare. `fixed_value` on an
    // undeclared input panics under debug assertions.
    let input = PrimitiveTransposeInput::<StdTensorOp>::Residual(super::input_key(1));
    let reference = TransposeInputRef::new(&input, 0, ResidualSpec::none());
    let _ = reference.fixed_value("test_rule", 0).unwrap();
}

#[test]
fn residual_mask_detector_accepts_declared_input_access() {
    let input = PrimitiveTransposeInput::<StdTensorOp>::Residual(super::input_key(1));
    let reference = TransposeInputRef::new(&input, 0, ResidualSpec::input(0));
    assert!(reference.fixed_value("test_rule", 0).is_ok());
    assert!(reference.shape_source_value("test_rule", 0).is_ok());
    // Metadata-only access stays allowed for undeclared indices.
    let undeclared = TransposeInputRef::new(&input, 0, ResidualSpec::none());
    assert_eq!(
        undeclared.metadata_value(),
        computegraph::types::ValueRef::External(super::input_key(1))
    );
}

#[test]
fn primitive_ad_registry_has_representative_rules() {
    assert!(primitive_ad_rule(PrimitiveOpKind::Add).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::DotGeneral).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::DynamicUpdateSlice).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::ShapeOf).is_some());
}

#[test]
fn primitive_ad_registry_covers_catalog_in_order() {
    assert_eq!(
        all_primitive_descriptors().len(),
        PrimitiveOpKind::COUNT,
        "catalog count should match dense PrimitiveOpKind indices"
    );
    for descriptor in all_primitive_descriptors() {
        let rule = primitive_ad_rule(descriptor.kind)
            .unwrap_or_else(|| panic!("missing AD rule for {:?}", descriptor.kind));
        assert_eq!(rule.kind(), descriptor.kind);
    }
}

#[test]
fn primitive_ad_registry_is_trait_based_and_direct_indexed() {
    let source = include_str!("../registry.rs");

    assert!(
        source.contains("trait PrimitiveAdRule"),
        "primitive AD registry should expose a graph-level PrimitiveAdRule trait"
    );
    assert!(
        !source.contains(".iter().find"),
        "primitive AD lookup should not linearly scan the registry"
    );
    assert!(
        !source.contains("struct PrimitiveAdRule"),
        "PrimitiveAdRule should be a trait, not a function-pointer data record"
    );
    assert!(
        !source.contains("unreachable!(\"catalog kind mismatch\")"),
        "registry rule payload mismatches should return ADRuleError instead of panicking"
    );
}

#[test]
fn primitive_ad_registry_returns_errors_for_catalog_mismatches() {
    let rule = primitive_ad_rule(PrimitiveOpKind::DotGeneral).unwrap();
    let op = StdTensorOp::Transpose { perm: vec![0] };
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();

    let linearize_err = rule
        .linearize(&op, &mut builder, &[], &[], &[], &mut ctx)
        .unwrap_err();
    assert_eq!(linearize_err.rule(), ADRuleKind::Jvp);
    assert!(linearize_err.to_string().contains("mismatched operation"));

    let transpose_err = rule
        .transpose_rule(
            &op,
            &mut builder,
            &[],
            &[],
            &OperationRole::Primary,
            &mut ctx,
        )
        .unwrap_err();
    assert_eq!(transpose_err.rule(), ADRuleKind::Transpose);
    assert!(transpose_err.to_string().contains("mismatched operation"));
}
