use super::*;

#[test]
fn gpu_descriptors_have_catalog_dtype_policy() {
    let add = gpu_descriptor(PrimitiveOpKind::Add).unwrap();
    assert_eq!(add.dtype_policy, DTypePolicy::SameNumeric);

    let compare = gpu_descriptor(PrimitiveOpKind::Compare).unwrap();
    assert_eq!(compare.dtype_policy, DTypePolicy::CompareToBool);
}

#[test]
fn host_only_primitives_have_no_gpu_descriptor() {
    assert_eq!(gpu_descriptor(PrimitiveOpKind::ShapeOf), None);
    assert_eq!(gpu_descriptor(PrimitiveOpKind::Constant), None);
}
