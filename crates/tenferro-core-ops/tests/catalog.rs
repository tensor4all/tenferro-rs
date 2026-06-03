use tenferro_core_ops::{
    all_primitive_descriptors, descriptor, DTypePolicy, OpCategory, PrimitiveOpKind,
};

#[test]
fn catalog_contains_core_primitives_only() {
    let names: Vec<_> = all_primitive_descriptors()
        .iter()
        .map(|entry| entry.name)
        .collect();

    assert!(names.contains(&"add"));
    assert!(names.contains(&"dot_general"));
    assert!(names.contains(&"dynamic_update_slice"));
    assert!(!names.iter().any(|name| name.contains("svd")));
    assert!(!names.iter().any(|name| name.contains("fft")));
    assert!(!names.iter().any(|name| name.contains("einsum")));
}

#[test]
fn descriptor_lookup_is_total_for_declared_kinds() {
    for entry in all_primitive_descriptors() {
        assert_eq!(descriptor(entry.kind).kind, entry.kind);
        assert!(!entry.name.is_empty());
    }
}

#[test]
fn representative_dtype_policies_are_explicit() {
    assert_eq!(
        descriptor(PrimitiveOpKind::Add).dtype_policy,
        DTypePolicy::SameNumeric
    );
    assert_eq!(
        descriptor(PrimitiveOpKind::Compare).dtype_policy,
        DTypePolicy::CompareToBool
    );
    assert_eq!(
        descriptor(PrimitiveOpKind::ShapeOf).category,
        OpCategory::Host
    );
}
