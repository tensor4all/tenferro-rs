use super::*;

#[test]
fn gpu_descriptors_have_catalog_dtype_policy() {
    let add = gpu_descriptor(PrimitiveOpKind::Add).unwrap();
    assert_eq!(add.dtype_policy, DTypePolicy::SameNumeric);
    assert_eq!(add.launch, GpuLaunchKind::BinaryFloatComplexInt);

    let div = gpu_descriptor(PrimitiveOpKind::Div).unwrap();
    assert_eq!(div.launch, GpuLaunchKind::BinaryFloatComplex);

    let compare = gpu_descriptor(PrimitiveOpKind::Compare).unwrap();
    assert_eq!(compare.dtype_policy, DTypePolicy::CompareToBool);
    assert_eq!(compare.launch, GpuLaunchKind::CompareFloatIntToBool);

    let select = gpu_descriptor(PrimitiveOpKind::Select).unwrap();
    assert_eq!(select.launch, GpuLaunchKind::SelectBoolFloatInt);
}

#[test]
fn host_only_primitives_have_no_gpu_descriptor() {
    assert_eq!(gpu_descriptor(PrimitiveOpKind::ShapeOf), None);
    assert_eq!(gpu_descriptor(PrimitiveOpKind::Constant), None);
}
