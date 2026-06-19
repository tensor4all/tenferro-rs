use super::{all_primitive_ad_support, primitive_ad_support, promote_dtype, AdRuleSupport};
use crate::ad::registry;
use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tenferro_tensor::DType;

#[test]
fn primitive_ad_support_manifest_covers_core_catalog_order() {
    let manifest = all_primitive_ad_support();
    assert_eq!(manifest.len(), PrimitiveOpKind::COUNT);

    for descriptor in all_primitive_descriptors() {
        let entry = primitive_ad_support(descriptor.kind);
        assert_eq!(entry.kind, descriptor.kind);
        assert_eq!(manifest[descriptor.kind.as_index()], *entry);
    }
}

#[test]
fn primitive_ad_support_manifest_matches_registered_rule_table() {
    for entry in all_primitive_ad_support() {
        let rule = registry::primitive_ad_rule(entry.kind).expect("primitive AD rule must exist");
        assert_eq!(rule.kind(), entry.kind);
        assert_ne!(entry.linearize, AdRuleSupport::Unsupported);
        assert_ne!(entry.transpose, AdRuleSupport::Unsupported);
    }
}

#[test]
fn primitive_ad_support_manifest_marks_known_non_differentiable_ops() {
    for kind in [
        PrimitiveOpKind::Compare,
        PrimitiveOpKind::ShapeOf,
        PrimitiveOpKind::Constant,
    ] {
        let entry = primitive_ad_support(kind);
        assert_eq!(entry.linearize, AdRuleSupport::NonDifferentiable);
        assert_eq!(entry.transpose, AdRuleSupport::NonDifferentiable);
    }
}

#[test]
fn promote_dtype_covers_supported_pairs_without_runtime_unreachable() {
    let source = include_str!("../support.rs");
    assert!(
        !source.contains("promote_dtype: unhandled pair"),
        "dtype promotion should use exhaustive DType matching, not a runtime unreachable"
    );

    assert_eq!(promote_dtype(DType::Bool, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::I32, DType::I64), DType::I64);
    assert_eq!(promote_dtype(DType::I64, DType::F32), DType::F64);
    assert_eq!(promote_dtype(DType::F32, DType::C32), DType::C32);
    assert_eq!(promote_dtype(DType::F64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::C32, DType::C64), DType::C64);
}
