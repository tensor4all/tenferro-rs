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
            F64 => f64,
            /// Single precision.
            F32 => f32,
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
