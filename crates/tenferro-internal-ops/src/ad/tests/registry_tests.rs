use computegraph::graph::GraphBuilder;
use computegraph::types::OperationRole;
use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tidu::ADRuleKind;

use crate::ad::context::ShapeGuardContext;
use crate::ad::registry::primitive_ad_rule;
use crate::std_tensor_op::StdTensorOp;

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
