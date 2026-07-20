use super::{validate_axis_groups, validate_dot_general, validate_layout_metadata};
use tenferro_tensor::{
    ContractionScalar, DType, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorRead,
    TensorViewMut, TensorWrite, TypedTensorViewMut,
};

fn config(
    lhs_contracting_dims: &[usize],
    rhs_contracting_dims: &[usize],
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contracting_dims.to_vec(),
        rhs_contracting_dims: rhs_contracting_dims.to_vec(),
        lhs_batch_dims: lhs_batch_dims.to_vec(),
        rhs_batch_dims: rhs_batch_dims.to_vec(),
    }
}

#[test]
fn axis_groups_preserve_order_and_find_free_axes() {
    let config = config(&[1], &[0], &[2], &[2]);
    let groups = validate_axis_groups(4, 4, &config).unwrap();

    assert_eq!(groups.contracting_pairs().collect::<Vec<_>>(), vec![(1, 0)]);
    assert_eq!(groups.batch_pairs().collect::<Vec<_>>(), vec![(2, 2)]);
    assert_eq!(groups.lhs_free_axes().collect::<Vec<_>>(), vec![0, 3]);
    assert_eq!(groups.rhs_free_axes().collect::<Vec<_>>(), vec![1, 3]);
}

#[test]
fn axis_groups_match_existing_rank_validation_through_rank_seventy() {
    for rank in [0, 1, 2, 8, 63, 64, 65, 70] {
        let valid = if rank == 0 {
            config(&[], &[], &[], &[])
        } else if rank == 1 {
            config(&[0], &[0], &[], &[])
        } else {
            config(&[rank - 1], &[0], &[rank - 2], &[rank - 1])
        };
        let invalid = [
            config(&[rank], &[0], &[], &[]),
            config(&[0, 0], &[0, 1], &[], &[]),
            config(&[0], &[0], &[0], &[0]),
            config(&[0], &[], &[], &[]),
            config(&[], &[], &[0], &[]),
        ];

        assert_eq!(
            validate_axis_groups(rank, rank, &valid).is_ok(),
            valid.validate_dims_with_ranks(rank, rank).is_ok(),
            "valid parity failed at rank {rank}",
        );
        for candidate in invalid {
            assert_eq!(
                validate_axis_groups(rank, rank, &candidate).is_ok(),
                candidate.validate_dims_with_ranks(rank, rank).is_ok(),
                "invalid parity failed at rank {rank}: {candidate:?}",
            );
        }
    }
}

#[test]
fn axis_group_role_conflict_preserves_ordered_error_parity() {
    let config = config(&[5, 2], &[0, 1], &[2, 5], &[2, 3]);
    let current = config.validate_dims_with_ranks(6, 6).unwrap_err();
    let candidate = validate_axis_groups(6, 6, &config).unwrap_err();

    assert_eq!(candidate.to_string(), current.to_string());
}

#[test]
fn axis_group_competing_errors_match_existing_precedence_through_rank_seventy() {
    for rank in [2, 8, 63, 64, 65, 70] {
        let cases = [
            config(&[0, 0], &[rank], &[], &[]),
            config(&[0, 0], &[0, 0], &[1, 1], &[1, 1]),
            config(&[0], &[], &[0], &[]),
            config(&[rank], &[rank], &[0, 0], &[0, 0]),
        ];
        for candidate in cases {
            let current = candidate.validate_dims_with_ranks(rank, rank).unwrap_err();
            let replacement = validate_axis_groups(rank, rank, &candidate).unwrap_err();
            assert_eq!(
                replacement.to_string(),
                current.to_string(),
                "error precedence diverged at rank {rank} for {candidate:?}",
            );
        }
    }
}

#[test]
fn dot_general_validation_checks_extents_output_and_accumulation() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output = Tensor::from_vec_col_major(vec![2, 5, 4], vec![0.0_f64; 40]).unwrap();
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let mut output = TensorWrite::from_tensor(&mut output);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
    assert_eq!(
        validated.axes().lhs_free_axes().collect::<Vec<_>>(),
        vec![0]
    );

    let wrong_accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F32(1.0),
        beta: ContractionScalar::F32(0.0),
    };
    assert!(validate_dot_general(&lhs, &rhs, &output, &config, wrong_accumulation).is_err());

    let mut wrong_shape = Tensor::from_vec_col_major(vec![2, 5], vec![0.0_f64; 10]).unwrap();
    let wrong_shape = TensorWrite::from_tensor(&mut wrong_shape);
    assert!(validate_dot_general(&lhs, &rhs, &wrong_shape, &config, accumulation).is_err());

    let bad_rhs = Tensor::from_vec_col_major(vec![7, 5, 4], vec![1.0_f64; 140]).unwrap();
    let bad_rhs = TensorRead::from_tensor(&bad_rhs);
    assert!(validate_dot_general(&lhs, &bad_rhs, &output, &config, accumulation).is_err());

    let _ = &mut output;
}

#[test]
fn layout_validation_checks_strides_and_reachable_ranges() {
    assert!(validate_layout_metadata("output", &[2, 3], &[1], 0, 6).is_err());
    assert!(validate_layout_metadata("output", &[2], &[-1], 0, 2).is_err());
    assert!(validate_layout_metadata("output", &[2], &[isize::MAX], 1, 2).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[1, 2], 0, 5).is_err());
    assert!(validate_layout_metadata("output", &[2, 3], &[-1, 2], 1, 6).is_ok());
}

#[test]
fn dot_general_validation_accepts_checked_negative_stride_output() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 5, 4], vec![1.0_f64; 60]).unwrap();
    let mut output_storage = vec![0.0_f64; 40];
    let output =
        TypedTensorViewMut::from_slice(vec![2, 5, 4], vec![-1, 2, 10], 1, &mut output_storage)
            .unwrap();
    let output = TensorWrite::from_view(TensorViewMut::F64(output));
    let lhs = TensorRead::from_tensor(&lhs);
    let rhs = TensorRead::from_tensor(&rhs);
    let config = config(&[1], &[0], &[2], &[2]);
    let accumulation = DotGeneralAccumulation::overwrite(DType::F64).unwrap();

    let validated = validate_dot_general(&lhs, &rhs, &output, &config, accumulation).unwrap();
    assert_eq!(validated.output_element_count(), 40);
}
