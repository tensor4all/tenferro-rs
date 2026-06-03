use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};

use crate::ad::registry::primitive_ad_rule;

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
}
