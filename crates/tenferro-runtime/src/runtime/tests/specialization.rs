use crate::runtime::{
    InputSignature, InputSignatureEntry, InputSpecializationRequirements, LayoutClass,
    PlacementProjection, PlacementSpecialization, SpecializationRequirements,
};
use tenferro_tensor::{DType, MemoryKind, Placement, ShapeVec, StrideVec};

fn shape(values: &[usize]) -> ShapeVec {
    values.iter().copied().collect()
}

fn strides(values: &[isize]) -> StrideVec {
    values.iter().copied().collect()
}

fn requirements(axes: Vec<u32>, alignment_log2: Option<u8>) -> InputSpecializationRequirements {
    let mut builder = InputSpecializationRequirements::builder();
    builder
        .rank(!axes.is_empty())
        .concrete_dimensions(axes)
        .alignment_log2(alignment_log2);
    builder.build().unwrap()
}

fn storage_class_for_other(payload: &str) -> String {
    let entry = InputSignatureEntry::new(
        DType::F64,
        shape(&[1]),
        Placement {
            memory_kind: MemoryKind::Other(payload.into()),
            device: None,
            cpu_affinity: None,
        },
        LayoutClass::new("tenferro.layout.compact-col-major.v1").unwrap(),
        strides(&[1]),
        None,
    )
    .unwrap();
    let signature = InputSignature::new(vec![entry]);
    let mut builder = InputSpecializationRequirements::builder();
    builder.placement(PlacementSpecialization::StorageClass);
    let aggregate = SpecializationRequirements::new(vec![builder.build().unwrap()]);
    let projection = aggregate.project(&signature).unwrap();

    match projection.inputs()[0].placement().unwrap() {
        PlacementProjection::StorageClass(storage) => storage.as_str().to_owned(),
        PlacementProjection::Device(_) => panic!("storage specialization must produce a class"),
    }
}

#[test]
fn specialization_storage_projection_other_utf8_hex_uses_fixed_utf8_examples_and_distinguishes_payloads(
) {
    let empty = storage_class_for_other("");
    let ascii = storage_class_for_other("Az-9");
    let non_ascii = storage_class_for_other("é雪");
    let different_ascii = storage_class_for_other("Az-8");

    assert_eq!(empty, "tenferro.storage.other-empty.v1");
    assert_eq!(ascii, "tenferro.storage.other-utf8-417a2d39.v1");
    assert_eq!(non_ascii, "tenferro.storage.other-utf8-c3a9e99baa.v1");
    assert_ne!(ascii, different_ascii);
    assert_ne!(ascii, non_ascii);
    assert_ne!(different_ascii, non_ascii);
}

#[test]
fn specialization_rank_zero_through_sixty_four_finite_chains_terminate() {
    for rank in 0_u32..=64 {
        let mut axes = (0..rank).collect::<Vec<_>>();
        let mut previous = SpecializationRequirements::new(vec![requirements(axes.clone(), None)]);
        let mut edges = 0_u32;

        while !axes.is_empty() {
            axes.pop();
            let next = SpecializationRequirements::new(vec![requirements(axes.clone(), None)]);
            assert!(next.strictly_widens(&previous));
            previous = next;
            edges += 1;
        }

        assert_eq!(edges, rank);
        assert!(!previous.strictly_widens(&previous));
    }

    let maximum = u8::try_from(usize::BITS - 1).unwrap();
    let mut alignment = Some(maximum);
    let mut previous = SpecializationRequirements::new(vec![requirements(Vec::new(), alignment)]);
    let mut edges = 0_u32;
    loop {
        let next_alignment = alignment.and_then(|value| value.checked_sub(1));
        let next = SpecializationRequirements::new(vec![requirements(Vec::new(), next_alignment)]);
        assert!(next.strictly_widens(&previous));
        previous = next;
        alignment = next_alignment;
        edges += 1;
        if alignment.is_none() {
            break;
        }
    }

    assert_eq!(edges, usize::BITS);
    assert!(!previous.strictly_widens(&previous));
}
