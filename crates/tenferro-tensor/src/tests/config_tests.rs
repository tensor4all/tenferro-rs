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
            assert_eq!(axes.is_heap(), count > 4);
        }
        let mut reserved = config.clone();
        reserved.lhs_contracting_dims.reserve(16);
        assert!(reserved.lhs_contracting_dims.is_heap());
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
            lhs_contracting_dims: lc.iter().copied().collect(),
            rhs_contracting_dims: rc.iter().copied().collect(),
            lhs_batch_dims: lb.iter().copied().collect(),
            rhs_batch_dims: rb.iter().copied().collect(),
        };
        assert!(config.validate_dims_with_ranks(2, 2).is_err(), "{config:?}");
    }
}

#[test]
fn validate_dims_with_explicit_ranks_rejects_out_of_range_contract() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().iter().copied().collect(),
        rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
        lhs_batch_dims: [].as_slice().iter().copied().collect(),
        rhs_batch_dims: [].as_slice().iter().copied().collect(),
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
        lhs_contracting_dims: [1].as_slice().iter().copied().collect(),
        rhs_contracting_dims: [0].as_slice().iter().copied().collect(),
        lhs_batch_dims: [].as_slice().iter().copied().collect(),
        rhs_batch_dims: [].as_slice().iter().copied().collect(),
    };
    config.validate_dims_with_ranks(2, 2).unwrap();
}

/// #2165: no contracting and no batch dims is an outer product, not a Hadamard.
///
/// `dot_general_output_shape` makes that form's output `lhs ++ rhs`, because
/// every operand axis is free. Reading equal operand shapes as a Hadamard here
/// made the eager and CUDA rewrites return `mul(a, b)` - rank 2 for rank-2
/// operands - where the graph declared rank 4, and made the compiler's
/// degenerate rewrite emit the same mismatch.
#[test]
fn degenerate_no_contract_is_an_outer_product_even_for_equal_shapes() {
    use crate::{degenerate_dot_kind, DegenerateDotKind};

    let no_dims = || tinyvec::TinyVec::new();
    let empty = DotGeneralConfig {
        lhs_contracting_dims: no_dims(),
        rhs_contracting_dims: no_dims(),
        lhs_batch_dims: no_dims(),
        rhs_batch_dims: no_dims(),
    };

    // Equal shapes are the arm that used to answer `Hadamard`.
    assert_eq!(
        degenerate_dot_kind(&empty, &[2, 2], &[2, 2]),
        Some(DegenerateDotKind::OuterProduct)
    );
    // Different shapes were already an outer product.
    assert_eq!(
        degenerate_dot_kind(&empty, &[2, 3], &[4, 5]),
        Some(DegenerateDotKind::OuterProduct)
    );
    // One scalar side stays the transpose-and-scale form.
    assert_eq!(
        degenerate_dot_kind(&empty, &[], &[2, 3]),
        Some(DegenerateDotKind::ScalarTranspose)
    );

    // The real Hadamard is the one whose operands are *entirely* batched, so the
    // output is the shared shape; the rewrite must keep answering that.
    let all_batch = DotGeneralConfig {
        lhs_contracting_dims: no_dims(),
        rhs_contracting_dims: no_dims(),
        lhs_batch_dims: [0, 1].as_slice().iter().copied().collect(),
        rhs_batch_dims: [0, 1].as_slice().iter().copied().collect(),
    };
    assert_eq!(
        degenerate_dot_kind(&all_batch, &[2, 2], &[2, 2]),
        Some(DegenerateDotKind::Hadamard)
    );

    // Every axis batched is not enough: batch axis `k` pairs `lhs_batch_dims[k]`
    // with `rhs_batch_dims[k]`, so a permuted list pairs different axes and a
    // plain multiply would transpose the result. Measured with `[2, 2]` operands
    // against `[1, 0]` on the right: `a[i, j] * b[j, i]`, which the general path
    // computes and a Hadamard would not.
    let permuted = DotGeneralConfig {
        lhs_contracting_dims: no_dims(),
        rhs_contracting_dims: no_dims(),
        lhs_batch_dims: [0, 1].as_slice().iter().copied().collect(),
        rhs_batch_dims: [1, 0].as_slice().iter().copied().collect(),
    };
    assert_eq!(degenerate_dot_kind(&permuted, &[2, 2], &[2, 2]), None);

    let permuted_left = DotGeneralConfig {
        lhs_contracting_dims: no_dims(),
        rhs_contracting_dims: no_dims(),
        lhs_batch_dims: [1, 0].as_slice().iter().copied().collect(),
        rhs_batch_dims: [0, 1].as_slice().iter().copied().collect(),
    };
    assert_eq!(degenerate_dot_kind(&permuted_left, &[2, 2], &[2, 2]), None);
}
