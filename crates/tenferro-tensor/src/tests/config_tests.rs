use crate::{DotGeneralConfig, Error, ValidationError};

#[test]
fn dot_general_axes_inline_spill_and_value_identity() {
    use std::hash::{DefaultHasher, Hash, Hasher};

    for count in 0..=9 {
        let config = DotGeneralConfig {
            lhs_contracting_dims: (0..2 * count).step_by(2).rev().collect(),
            rhs_contracting_dims: (0..2 * count).step_by(2).collect(),
            lhs_batch_dims: (1..2 * count).step_by(2).collect(),
            rhs_batch_dims: (1..2 * count).step_by(2).rev().collect(),
        };
        config
            .validate_dims_with_ranks(2 * count, 2 * count)
            .unwrap();
        for axes in [
            &config.lhs_contracting_dims,
            &config.rhs_contracting_dims,
            &config.lhs_batch_dims,
            &config.rhs_batch_dims,
        ] {
            assert_eq!(axes.spilled(), count > 4);
        }
        let mut reserved = config.clone();
        reserved.lhs_contracting_dims.reserve(16);
        assert!(reserved.lhs_contracting_dims.spilled());
        assert_eq!(config, reserved);
        let hash = |value: &DotGeneralConfig| {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&config), hash(&reserved));
        if count >= 2 {
            reserved.lhs_contracting_dims.swap(0, 1);
            assert_ne!(config, reserved);
        }
    }
}

#[test]
fn dot_general_inline_axes_preserve_all_validation_errors() {
    let invalid: [[&[usize]; 4]; 12] = [
        [&[2], &[0], &[], &[]],
        [&[1], &[2], &[], &[]],
        [&[1], &[0], &[2], &[1]],
        [&[1], &[0], &[0], &[2]],
        [&[1, 1], &[0, 1], &[], &[]],
        [&[0, 1], &[0, 0], &[], &[]],
        [&[], &[], &[0, 0], &[0, 1]],
        [&[], &[], &[0, 1], &[1, 1]],
        [&[1], &[0], &[1], &[1]],
        [&[1], &[0], &[0], &[0]],
        [&[1], &[], &[], &[]],
        [&[], &[], &[0], &[]],
    ];
    for [lc, rc, lb, rb] in invalid {
        let config = DotGeneralConfig {
            lhs_contracting_dims: lc.into(),
            rhs_contracting_dims: rc.into(),
            lhs_batch_dims: lb.into(),
            rhs_batch_dims: rb.into(),
        };
        assert!(config.validate_dims_with_ranks(2, 2).is_err(), "{config:?}");
    }
}

#[test]
fn validate_dims_with_explicit_ranks_rejects_out_of_range_contract() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let err = config
        .validate_dims_with_ranks(2, 2)
        .expect_err("dim index 2 is out of range for rank 2");
    assert!(matches!(
        err,
        Error::Validation {
            op: "dot_general",
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        }
    ));
    assert!(err.to_string().contains("out of bounds"));
}

#[test]
fn validate_dims_with_explicit_ranks_accepts_valid_config() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    config.validate_dims_with_ranks(2, 2).unwrap();
}
