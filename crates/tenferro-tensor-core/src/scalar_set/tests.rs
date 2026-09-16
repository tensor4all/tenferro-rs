use crate::{DType, DefaultScalars, HostTensor, ScalarSet};

#[test]
fn the_default_set_reports_every_member() {
    let values = [
        (
            DefaultScalars::F32(HostTensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap()),
            DType::F32,
        ),
        (
            DefaultScalars::F64(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap()),
            DType::F64,
        ),
        (
            DefaultScalars::I32(HostTensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap()),
            DType::I32,
        ),
        (
            DefaultScalars::I64(HostTensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap()),
            DType::I64,
        ),
        (
            DefaultScalars::Bool(HostTensor::from_vec_col_major(vec![1], vec![true]).unwrap()),
            DType::Bool,
        ),
        (
            DefaultScalars::C32(
                HostTensor::from_vec_col_major(
                    vec![1],
                    vec![num_complex::Complex32::new(1.0, 0.0)],
                )
                .unwrap(),
            ),
            DType::C32,
        ),
        (
            DefaultScalars::C64(
                HostTensor::from_vec_col_major(
                    vec![1],
                    vec![num_complex::Complex64::new(1.0, 0.0)],
                )
                .unwrap(),
            ),
            DType::C64,
        ),
    ];

    assert_eq!(<DefaultScalars as ScalarSet>::TAGS.len(), values.len());
    for (index, (value, expected)) in values.iter().enumerate() {
        assert_eq!(value.tag(), *expected);
        assert_eq!(<DefaultScalars as ScalarSet>::TAGS[index], *expected);
    }
}

#[test]
fn a_locally_declared_set_reports_its_own_members() {
    crate::define_scalar_set! {
        /// Tag for the test set.
        pub enum TestTag {
            /// Double precision.
            F64 => f64 : Float 1 64,
            /// Single precision.
            F32 => f32 : Float 0 32,
        }
        /// Value enum for the test set.
        pub enum TestSet;
    }

    let wide = TestSet::F64(HostTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap());
    let narrow = TestSet::F32(HostTensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap());

    assert_eq!(wide.tag(), TestTag::F64);
    assert_eq!(narrow.tag(), TestTag::F32);
    assert_eq!(<TestSet as ScalarSet>::TAGS, &[TestTag::F64, TestTag::F32]);
}

#[test]
fn the_tag_carries_an_externally_defined_member_without_listing_it() {
    use crate::{promote_specs, DType, MemberKind};

    let external = DType::External(core::any::TypeId::of::<u128>());
    let other = DType::External(core::any::TypeId::of::<u64>());

    assert_ne!(external, other);
    assert_eq!(external.spec().kind, MemberKind::External);
    // The member tables list the declared members only, so the external tag is not
    // a member of the set even though it is part of the tag.
    assert!(!DType::TAGS.contains(&external));
    assert_eq!(DType::SPECS.len(), DType::TAGS.len());
    // An externally defined scalar promotes to itself, because tenferro declares
    // no facts that could relate it to one of its own members.
    assert_eq!(
        promote_specs(external.spec(), DType::F64.spec()).kind,
        MemberKind::External
    );
    assert_eq!(
        <DefaultScalars as ScalarSet>::promote(external, DType::F64),
        external
    );
    assert_eq!(
        <DefaultScalars as ScalarSet>::promote(DType::F64, external),
        external
    );
}

#[test]
fn the_tag_stays_within_its_measured_size() {
    // The tag was one byte before it gained the externally defined variant, and
    // the variant carries a `TypeId`. 24 bytes is the measured cost; a change here
    // should be a deliberate decision because the tag is embedded in metadata,
    // cache keys, and error types.
    assert!(core::mem::size_of::<crate::DType>() <= 24);
}

#[test]
fn promotion_between_two_external_scalars_is_a_known_gap() {
    use crate::{DType, MemberKind, ScalarSet};

    let first = DType::External(core::any::TypeId::of::<u128>());
    let second = DType::External(core::any::TypeId::of::<u64>());

    // Two external scalars are two unrelated types, so tenferro cannot relate
    // them. The current answer returns the left operand, which is recorded here
    // rather than hidden: the checked rejection belongs at the entry point where a
    // promotion drives execution, together with the extension boundary. See
    // docs/design/scalar-composition.md section 5.4.
    assert_eq!(first.spec().kind, MemberKind::External);
    assert_eq!(<DefaultScalars as ScalarSet>::promote(first, second), first);
}
