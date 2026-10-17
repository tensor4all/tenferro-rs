use num_complex::{Complex32, Complex64};
use tenferro_tensor::{
    DefaultScalars, DefaultScalarsRef, Error, Host, StridedSliceSpec, TensorScalar, TypedTensor,
    TypedTensorView,
};

use tenferro_tensor_core::{
    col_major_strides, DType, DynRank, ErrorKind, IntoRankShape, Rank, ShapeMismatch, ShapeVec,
    SliceSpec, TensorLayout, TensorRank, ValidationError, ValidationKind,
};

/// Unwrap the validation source of a tensor error for exact-payload matches.
fn validation(err: Error) -> ValidationError {
    match err {
        Error::Validation { source, .. } => source,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn shape_mismatch_keeps_machine_readable_payload() {
    let err: ValidationError = ShapeMismatch::IncompatibleShapes {
        lhs: vec![2, 3].into_iter().collect::<ShapeVec>(),
        rhs: vec![2, 4].into_iter().collect::<ShapeVec>(),
    }
    .into();

    assert_eq!(err.kind(), ValidationKind::ShapeMismatch);
    assert!(matches!(
        &err,
        ValidationError::ShapeMismatch(mismatch)
            if matches!(mismatch.as_ref(), ShapeMismatch::IncompatibleShapes { lhs, rhs }
                if lhs.as_slice() == [2, 3] && rhs.as_slice() == [2, 4])
    ));

    let source = std::error::Error::source(&err).expect("shape mismatch source");
    assert!(source.downcast_ref::<Box<ShapeMismatch>>().is_some());
}

#[test]
fn public_error_kind_can_classify_validation() {
    assert_eq!(
        ErrorKind::Validation(ValidationKind::RankMismatch),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
}

#[test]
fn tensor_core_does_not_expose_row_major_compatibility_apis() {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(crate_dir.join("src/lib.rs"))
        .expect("tenferro-tensor-core source must be readable");

    assert!(
        !source.contains("from_vec_row_major") && !source.contains("into_vec_row_major"),
        "tensor-core public API must stay column-major only; row-major conversion belongs at external boundaries"
    );
}

#[test]
fn dynamic_rank_shape_roundtrips_vec() {
    let shape = <DynRank as TensorRank>::shape_from_vec(vec![2, 3].into_iter().collect()).unwrap();
    assert_eq!(shape.as_ref(), &[2, 3]);
    assert_eq!(
        <DynRank as TensorRank>::shape_into_vec(shape).as_slice(),
        &[2, 3]
    );
}

#[test]
fn dynamic_rank_strides_roundtrip_vec() {
    let strides =
        <DynRank as TensorRank>::strides_from_vec(vec![1, 2, -1].into_iter().collect()).unwrap();
    assert_eq!(strides.as_ref(), &[1, 2, -1]);
    assert_eq!(
        <DynRank as TensorRank>::strides_into_vec(strides).as_slice(),
        &[1, 2, -1]
    );
}

#[test]
fn static_rank_shape_and_strides_roundtrip_vecs() {
    let shape = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3].into_iter().collect()).unwrap();
    assert_eq!(
        <Rank<2> as TensorRank>::shape_into_vec(shape).as_slice(),
        &[2, 3]
    );

    let strides =
        <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2].into_iter().collect()).unwrap();
    assert_eq!(
        <Rank<2> as TensorRank>::strides_into_vec(strides).as_slice(),
        &[1, 2]
    );
}

#[test]
fn static_rank_rejects_wrong_shape_length() {
    let err =
        <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3, 4].into_iter().collect()).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn rank_shape_conversion_accepts_static_and_runtime_inputs() {
    let array = <[usize; 2] as IntoRankShape<Rank<2>>>::into_rank_shape([2, 3]).unwrap();
    assert_eq!(array, [2, 3]);

    let array_ref: &[usize; 2] = &[4, 5];
    let array_ref = <&[usize; 2] as IntoRankShape<Rank<2>>>::into_rank_shape(array_ref).unwrap();
    assert_eq!(array_ref, [4, 5]);

    let vector = <Vec<usize> as IntoRankShape<Rank<2>>>::into_rank_shape(vec![6, 7]).unwrap();
    assert_eq!(vector, [6, 7]);

    let slice = <&[usize] as IntoRankShape<Rank<2>>>::into_rank_shape(&[8, 9][..]).unwrap();
    assert_eq!(slice, [8, 9]);

    let shape_vec = <ShapeVec as IntoRankShape<Rank<2>>>::into_rank_shape(
        vec![10, 11].into_iter().collect::<ShapeVec>(),
    )
    .unwrap();
    assert_eq!(shape_vec, [10, 11]);

    let dynamic = <[usize; 9] as IntoRankShape<DynRank>>::into_rank_shape([0; 9]).unwrap();
    assert_eq!(dynamic.as_slice(), &[0; 9]);
}

#[test]
fn rank_shape_conversion_rejects_wrong_runtime_rank() {
    let vector = <Vec<usize> as IntoRankShape<Rank<2>>>::into_rank_shape(vec![1]).unwrap_err();
    assert!(matches!(
        vector,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));

    let slice = <&[usize] as IntoRankShape<Rank<2>>>::into_rank_shape(&[1, 2, 3][..]).unwrap_err();
    assert!(matches!(
        slice,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));

    let shape_vec = <ShapeVec as IntoRankShape<Rank<2>>>::into_rank_shape(
        vec![].into_iter().collect::<ShapeVec>(),
    )
    .unwrap_err();
    assert!(matches!(
        shape_vec,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 0
        }
    ));
}

#[test]
fn static_rank_rejects_wrong_stride_length() {
    let err =
        <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2, 3].into_iter().collect()).unwrap_err();
    assert!(matches!(
        err,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn compact_layout_for_static_rank_has_column_major_strides() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    assert_eq!(layout.shape(), &[2, 3]);
    assert_eq!(layout.strides(), &[1, 2]);
    assert_eq!(layout.offset(), 0);
    assert!(layout.is_compact_col_major().unwrap());
}

#[test]
fn transpose_view_permutes_layout_metadata() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    let transposed = layout.transpose_view([1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[2, 1]);
    assert_eq!(transposed.offset(), 0);
}

#[test]
fn slice_view_supports_negative_step() {
    let layout = TensorLayout::<Rank<1>>::compact([4]).unwrap();
    let sliced = layout
        .slice_view(
            [SliceSpec {
                start: 3,
                end: -1,
                step: -2,
            }],
            4,
        )
        .unwrap();
    assert_eq!(sliced.shape(), &[2]);
    assert_eq!(sliced.strides(), &[-2]);
    assert_eq!(sliced.offset(), 3);
}

#[test]
fn slice_view_normalizes_negative_end_for_negative_step() {
    let layout = TensorLayout::<Rank<1>>::compact([5]).unwrap();
    let sliced = layout
        .slice_view(
            [SliceSpec {
                start: 3,
                end: -3,
                step: -1,
            }],
            5,
        )
        .unwrap();

    assert_eq!(sliced.shape(), &[1]);
    assert_eq!(sliced.strides(), &[-1]);
    assert_eq!(sliced.offset(), 3);
}

#[test]
fn reshape_view_as_requires_compact_layout() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    let reshaped = layout.reshape_view_as::<Rank<1>>([6], 6).unwrap();
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.strides(), &[1]);

    let non_compact = TensorLayout::<Rank<2>>::from_parts([2, 3], [2, 1], 0, 6).unwrap();
    let err = non_compact.reshape_view_as::<Rank<1>>([6], 6).unwrap_err();
    assert!(matches!(err, ValidationError::NonContiguousViewAsSlice));
    assert_eq!(
        err.to_string(),
        "view is not slice-contiguous; materialize with to_contiguous before requesting a borrowed slice"
    );
}

#[test]
fn slice_view_rejects_zero_step_with_exact_error() {
    let layout = TensorLayout::<Rank<1>>::compact([4]).unwrap();
    let err = layout
        .slice_view(
            [SliceSpec {
                start: 0,
                end: 4,
                step: 0,
            }],
            4,
        )
        .unwrap_err();
    assert!(matches!(err, ValidationError::InvalidSliceStep { step: 0 }));
}

#[test]
fn slice_view_rejects_invalid_normalized_bounds_with_exact_error() {
    let layout = TensorLayout::<Rank<1>>::compact([4]).unwrap();
    let err = layout
        .slice_view(
            [SliceSpec {
                start: 3,
                end: -6,
                step: -1,
            }],
            4,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        ValidationError::InvalidSliceBounds {
            start: 3,
            end: -6,
            axis_len: 4
        }
    ));
}

#[test]
fn broadcast_in_dim_view_uses_zero_strides_for_broadcast_axes() {
    let layout = TensorLayout::<Rank<1>>::compact([3]).unwrap();
    let broadcast = layout
        .broadcast_in_dim_view::<Rank<2>>([2, 3], [1], 3)
        .unwrap();
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(broadcast.strides(), &[0, 1]);
}

#[test]
fn dynamic_layout_rejects_shape_stride_rank_mismatch() {
    let err = TensorLayout::<DynRank>::from_parts(
        vec![2, 3].into_iter().collect(),
        vec![1].into_iter().collect(),
        0,
        6,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ValidationError::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn scalar_layout_with_static_rank_zero_is_compact() {
    let layout = TensorLayout::<Rank<0>>::compact([]).unwrap();
    assert_eq!(layout.shape(), &[] as &[usize]);
    assert_eq!(layout.strides(), &[] as &[isize]);
    assert_eq!(layout.offset(), 0);
    assert!(layout.is_compact_col_major().unwrap());
}

#[test]
fn non_compact_layout_reports_false() {
    let layout = TensorLayout::<Rank<2>>::from_parts([2, 3], [2, 1], 0, 6).unwrap();
    assert_eq!(layout.shape(), &[2, 3]);
    assert_eq!(layout.strides(), &[2, 1]);
    assert!(!layout.is_compact_col_major().unwrap());
}

#[test]
fn compact_layout_reports_stride_overflow() {
    let err = TensorLayout::<Rank<2>>::compact([usize::MAX, 2]).unwrap_err();
    assert!(matches!(err, ValidationError::IntegerOverflow));
}

#[test]
fn layout_rejects_non_empty_broadcast_shape_product_overflow() {
    let huge_extent = usize::MAX / 2 + 1;
    let err = TensorLayout::<DynRank>::from_parts(
        vec![huge_extent, huge_extent].into_iter().collect(),
        vec![0, 0].into_iter().collect(),
        0,
        1,
    )
    .unwrap_err();
    assert!(matches!(err, ValidationError::IntegerOverflow));
}

#[test]
fn layout_from_parts_preserves_offset() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![3].into_iter().collect(),
        vec![1].into_iter().collect(),
        7,
        10,
    )
    .unwrap();
    assert_eq!(layout.shape(), &[3]);
    assert_eq!(layout.strides(), &[1]);
    assert_eq!(layout.offset(), 7);
}

#[test]
fn layout_accepts_negative_stride_when_reachable_range_is_in_bounds() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![3].into_iter().collect(),
        vec![-1].into_iter().collect(),
        2,
        3,
    )
    .unwrap();
    assert_eq!(layout.shape(), &[3]);
    assert_eq!(layout.strides(), &[-1]);
    assert_eq!(layout.offset(), 2);
}

#[test]
fn layout_rejects_negative_stride_when_reachable_range_is_out_of_bounds() {
    assert!(TensorLayout::<DynRank>::from_parts(
        vec![3].into_iter().collect(),
        vec![-1].into_iter().collect(),
        1,
        3
    )
    .is_err());
}

#[test]
fn layout_rejects_positive_stride_when_max_offset_exceeds_buffer_len() {
    assert!(TensorLayout::<DynRank>::from_parts(
        vec![3].into_iter().collect(),
        vec![2].into_iter().collect(),
        0,
        3
    )
    .is_err());
}

#[test]
fn layout_accepts_empty_shape_offsets_at_buffer_boundaries() {
    let at_start = TensorLayout::<DynRank>::from_parts(
        vec![0].into_iter().collect(),
        vec![1].into_iter().collect(),
        0,
        3,
    )
    .unwrap();
    assert_eq!(at_start.offset(), 0);

    let at_end = TensorLayout::<DynRank>::from_parts(
        vec![0].into_iter().collect(),
        vec![1].into_iter().collect(),
        3,
        3,
    )
    .unwrap();
    assert_eq!(at_end.offset(), 3);

    assert!(TensorLayout::<DynRank>::from_parts(
        vec![0].into_iter().collect(),
        vec![1].into_iter().collect(),
        4,
        3
    )
    .is_err());
}

#[test]
fn layout_reports_overflow_for_unreachable_offset_arithmetic() {
    assert!(matches!(
        TensorLayout::<DynRank>::from_parts(
            vec![3].into_iter().collect(),
            vec![isize::MAX].into_iter().collect(),
            0,
            3
        )
        .unwrap_err(),
        ValidationError::IntegerOverflow
    ));
}

#[test]
fn mutable_layout_rejects_zero_stride_broadcast() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![2].into_iter().collect(),
        vec![0].into_iter().collect(),
        0,
        1,
    )
    .unwrap();
    let err = layout.validate_mutable_no_overlap().unwrap_err();
    assert!(matches!(err, ValidationError::OverlappingMutableLayout));
    assert_eq!(
        err.to_string(),
        "mutable tensor layout may overlap physical elements; materialize a contiguous owner before requesting mutable access"
    );
}

#[test]
fn mutable_layout_rejects_overlapping_strides() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![2, 2].into_iter().collect(),
        vec![1, 1].into_iter().collect(),
        0,
        4,
    )
    .unwrap();
    assert!(matches!(
        layout.validate_mutable_no_overlap(),
        Err(ValidationError::OverlappingMutableLayout)
    ));
}

#[test]
fn mutable_layout_accepts_reversed_non_overlapping_vector() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![3].into_iter().collect(),
        vec![-1].into_iter().collect(),
        2,
        3,
    )
    .unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_accepts_empty_view_before_product_overflow() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![2, usize::MAX, 0].into_iter().collect(),
        vec![1, 1, 1].into_iter().collect(),
        0,
        0,
    )
    .unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_exact_fallback_accepts_small_ambiguous_non_overlapping_layout() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![3, 2].into_iter().collect(),
        vec![2, 3].into_iter().collect(),
        0,
        8,
    )
    .unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_rejects_large_ambiguous_layout_without_exact_fallback() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![4097, 2].into_iter().collect(),
        vec![2, 4097].into_iter().collect(),
        0,
        12_290,
    )
    .unwrap();
    assert!(matches!(
        layout.validate_mutable_no_overlap(),
        Err(ValidationError::OverlappingMutableLayout)
    ));
}

#[test]
fn layout_rejects_huge_non_empty_zero_stride_before_product_overflow() {
    let huge_extent = isize::MAX as usize + 1;
    assert!(matches!(
        TensorLayout::<DynRank>::from_parts(
            vec![huge_extent, 3].into_iter().collect(),
            vec![0, 1].into_iter().collect(),
            0,
            3
        ),
        Err(ValidationError::IntegerOverflow)
    ));
}

#[test]
fn constructs_contiguous_col_major_and_validates_count() {
    let tensor = TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let err = TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1.0_f64])
        .unwrap_err();
    assert!(matches!(
        validation(err),
        ValidationError::ShapeDataLengthMismatch {
            expected: 4,
            actual: 1
        }
    ));
}

#[test]
fn owned_view_has_expected_shape_stride_offset() {
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2, 3], vec![0_i64; 6])
            .unwrap();
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.offset(), 0);
    assert!(view.layout().is_compact_col_major().unwrap());
}

#[test]
fn reshape_view_requires_compact_col_major_but_allows_nonzero_offset() {
    let data = [10_i32, 11, 12, 13, 14, 15];
    let compact_nonzero = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![4].as_slice(),
        vec![1].as_slice(),
        1,
        &data,
    )
    .unwrap();
    let reshaped = compact_nonzero.reshape_view(&[2, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[2, 2]);
    assert_eq!(reshaped.offset(), 1);
    assert_eq!(reshaped.as_slice().unwrap(), &[11, 12, 13, 14]);

    let non_contiguous = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![2, 2].as_slice(),
        vec![1, 3].as_slice(),
        0,
        &data,
    )
    .unwrap();
    let err = non_contiguous.reshape_view(&[4]).unwrap_err();
    assert!(matches!(
        validation(err),
        ValidationError::NonContiguousViewAsSlice
    ));
}

#[test]
fn transpose_view_only_reorders_metadata() {
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2, 3], vec![0_i64; 6])
            .unwrap();
    let view = tensor.as_view().transpose_view([1, 0]).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.strides(), &[2, 1]);
    assert_eq!(view.offset(), 0);
    assert!(matches!(
        validation(tensor.as_view().transpose_view([0, 0]).unwrap_err()),
        ValidationError::DuplicateAxis {
            axis: 0,
            role: "permutation",
        }
    ));
}

#[test]
fn slice_view_positive_step_and_empty_slice() {
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![5], vec![1_i64, 2, 3, 4, 5])
            .unwrap();
    let view = tensor
        .as_view()
        .slice_view(&[StridedSliceSpec::new(1, Some(5), 2)])
        .unwrap();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(view.strides(), &[2]);
    assert_eq!(view.offset(), 1);

    let empty = tensor
        .as_view()
        .slice_view(&[StridedSliceSpec::new(4, Some(2), 1)])
        .unwrap();
    assert!(empty.shape().contains(&0));
    assert_eq!(empty.as_slice().unwrap(), &[] as &[i64]);
}

#[test]
fn slice_view_negative_step_empty_at_axis_end_is_not_rejected() {
    // An empty selection is step-sign independent: `start == axis_len` must
    // behave exactly like the positive-step empty case and the
    // `start < axis_len` negative-step case. See issue #2025.
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![3], vec![1_i64, 2, 3])
            .unwrap();
    for slice in [
        StridedSliceSpec::new(3, Some(3), -1),
        StridedSliceSpec::new(3, None, -1),
    ] {
        let empty = tensor.as_view().slice_view(&[slice]).unwrap();
        assert_eq!(empty.shape(), &[0]);
        assert_eq!(empty.n_elements(), 0);
    }
}

#[test]
fn invalid_slice_steps_and_negative_bounds_are_rejected() {
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![3], vec![1_i64, 2, 3])
            .unwrap();
    assert!(matches!(
        validation(
            tensor
                .as_view()
                .slice_view(&[StridedSliceSpec::new(0, Some(2), 0)])
                .unwrap_err()
        ),
        ValidationError::InvalidSliceStep { step: 0 }
    ));
    // Strided slice bounds count negative values from the axis end, so only a
    // bound that stays negative after normalization is rejected.
    assert!(matches!(
        validation(
            tensor
                .as_view()
                .slice_view(&[StridedSliceSpec::new(-4, Some(2), 1)])
                .unwrap_err()
        ),
        ValidationError::InvalidSliceBounds { .. }
    ));
}

#[test]
fn view_bounds_are_validated_eagerly_with_checked_arithmetic() {
    let data = [1_i32, 2, 3];
    assert!(matches!(
        validation(
            TypedTensorView::<_, DynRank, Host>::from_host_slice(
                vec![4].as_slice(),
                vec![1].as_slice(),
                0,
                &data
            )
            .unwrap_err()
        ),
        ValidationError::ViewOutOfBounds
    ));
    assert!(matches!(
        validation(
            TypedTensorView::<_, DynRank, Host>::from_host_slice(
                vec![usize::MAX, 2].as_slice(),
                vec![1, 2].as_slice(),
                0,
                &data
            )
            .unwrap_err()
        ),
        ValidationError::IntegerOverflow
    ));

    let reversed = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![3].as_slice(),
        vec![-1].as_slice(),
        2,
        &data,
    )
    .unwrap();
    assert_eq!(reversed.shape(), &[3]);
    assert_eq!(reversed.strides(), &[-1]);
    assert_eq!(reversed.offset(), 2);
    assert!(matches!(
        validation(
            TypedTensorView::<_, DynRank, Host>::from_host_slice(
                vec![3].as_slice(),
                vec![-1].as_slice(),
                1,
                &data
            )
            .unwrap_err()
        ),
        ValidationError::ViewOutOfBounds
    ));
}

#[test]
fn empty_view_offsets_may_point_one_past_the_borrowed_slice() {
    let data = [1_i32, 2, 3];
    let empty = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![0].as_slice(),
        vec![-1].as_slice(),
        3,
        &data,
    )
    .unwrap();
    assert!(empty.shape().contains(&0));
    assert_eq!(empty.as_slice().unwrap(), &[] as &[i32]);

    assert!(matches!(
        validation(
            TypedTensorView::<_, DynRank, Host>::from_host_slice(
                vec![0].as_slice(),
                vec![1].as_slice(),
                4,
                &data
            )
            .unwrap_err()
        ),
        ValidationError::ViewOutOfBounds
    ));
}

#[test]
fn empty_views_are_contiguous_even_with_degenerate_strides() {
    let data = [1_i32, 2, 3];
    let empty = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![0, 3].as_slice(),
        vec![1, 0].as_slice(),
        3,
        &data,
    )
    .unwrap();
    assert!(empty.shape().contains(&0));
    assert_eq!(empty.as_slice().unwrap(), &[] as &[i32]);
    assert_eq!(empty.reshape_view(&[0]).unwrap().shape(), &[0]);

    let layout = TensorLayout::<DynRank>::from_parts(
        vec![0, 3].into_iter().collect(),
        vec![99, -7].into_iter().collect(),
        3,
        data.len(),
    )
    .unwrap();
    assert!(layout.is_compact_col_major().unwrap());
    assert_eq!(
        layout
            .reshape_view_as::<DynRank>(vec![0].as_slice(), data.len())
            .unwrap()
            .shape(),
        &[0]
    );
}

#[test]
fn empty_axis_allows_negative_step_slices() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![0].into_iter().collect(),
        vec![1].into_iter().collect(),
        0,
        0,
    )
    .unwrap();
    let reversed = layout
        .slice_view(
            [SliceSpec {
                start: 0,
                end: 0,
                step: -1,
            }],
            0,
        )
        .unwrap();

    assert_eq!(reversed.shape(), &[0]);
    assert_eq!(reversed.offset(), 0);
}

#[test]
fn as_slice_accepts_nonzero_offset_and_rejects_non_contiguous_views() {
    let data = [1_i32, 2, 3, 4, 5];
    let contiguous = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![3].as_slice(),
        vec![1].as_slice(),
        1,
        &data,
    )
    .unwrap();
    assert_eq!(contiguous.as_slice().unwrap(), &[2, 3, 4]);

    let non_contiguous = TypedTensorView::<_, DynRank, Host>::from_host_slice(
        vec![2].as_slice(),
        vec![2].as_slice(),
        0,
        &data,
    )
    .unwrap();
    // The canonical view reports a non-contiguous layout with the same variant as
    // `reshape_view_as`.
    assert!(matches!(
        validation(non_contiguous.as_slice().unwrap_err()),
        ValidationError::NonContiguousViewAsSlice
    ));
}

#[test]
fn dynamic_tensor_and_view_report_dtype_mismatch() {
    let tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(tensor.dtype(), DType::F64);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert!(matches!(
        validation(tensor.as_slice::<f32>().unwrap_err()),
        ValidationError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F64
        }
    ));
    assert_eq!(
        tensor.as_view().reshape_view([1, 2]).unwrap().shape(),
        &[1, 2]
    );
}

#[test]
fn tensor_scalar_into_tensor_rejects_shape_data_length_mismatch() {
    let err = <f64 as TensorScalar>::into_default_scalars(vec![2, 3], vec![1.0, 2.0]).unwrap_err();

    assert!(matches!(
        validation(err),
        ValidationError::ShapeDataLengthMismatch {
            expected: 6,
            actual: 2
        }
    ));
}

#[test]
fn layout_helpers_and_col_major_roundtrip() {
    assert_eq!(col_major_strides(&[2, 3]).unwrap().as_slice(), &[1, 2]);
    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1_i64, 3, 2, 4])
            .unwrap();
    assert_eq!(tensor.as_slice(), &[1, 3, 2, 4]);
    assert_eq!(tensor.into_vec_col_major().1, vec![1, 3, 2, 4]);
}

#[test]
fn typed_owned_accessors_and_exports_cover_success_and_errors() {
    let mut tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    assert_eq!(tensor.rank(), 1);
    assert!(!tensor.shape().contains(&0));
    tensor.host_data_mut()[1] = 3;
    assert_eq!(tensor.as_slice(), &[1, 3]);

    // An owned reshape rebuilds the tensor from its column-major buffer.
    let reshaped = TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(
        vec![1, 2],
        tensor.clone().into_host_vec(),
    )
    .unwrap();
    assert_eq!(reshaped.shape(), &[1, 2]);
    assert!(matches!(
        validation(
            TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(
                vec![3],
                tensor.into_host_vec()
            )
            .unwrap_err()
        ),
        ValidationError::ShapeDataLengthMismatch {
            expected: 3,
            actual: 2
        }
    ));

    let (shape, data) = reshaped.into_vec_col_major();
    assert_eq!(shape.as_slice(), &[1, 2]);
    assert_eq!(data, vec![1, 3]);

    let empty =
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![0], vec![]).unwrap();
    assert!(empty.shape().contains(&0));
}

#[test]
fn scalar_col_major_helpers_cover_scalar_and_empty_shapes() {
    let scalar =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![], vec![7_i64]).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.into_vec_col_major().1, vec![7]);

    let empty =
        TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![0, 3], vec![]).unwrap();
    assert!(empty.shape().contains(&0));
    assert_eq!(empty.into_vec_col_major().1, Vec::<i64>::new());
}

#[test]
fn dynamic_tensor_accessors_cover_all_dtype_variants() {
    let tensors = [
        DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![3_i32]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![4_i64]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![true]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 2.0)]).unwrap(),
        DefaultScalars::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)]).unwrap(),
    ];
    let expected = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];

    for (tensor, dtype) in tensors.iter().zip(expected) {
        assert_eq!(tensor.dtype(), dtype);
        assert_eq!(tensor.shape(), &[1]);
        assert_eq!(tensor.rank(), 1);
        assert!(!tensor.is_empty());

        let view = tensor.as_view();
        assert_eq!(view.dtype(), dtype);
        assert_eq!(view.shape(), &[1]);
        assert_eq!(view.rank(), 1);
        assert!(!view.is_empty());
    }

    let empty = DefaultScalars::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();
    assert!(empty.is_empty());
    assert!(empty.as_view().is_empty());
}

#[test]
fn dynamic_tensor_mutation_and_owned_exports_validate_dtype() {
    let mut tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    tensor.as_mut_slice::<f64>().unwrap()[0] = 5.0;
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[5.0, 2.0]);
    assert!(matches!(
        validation(tensor.as_mut_slice::<f32>().unwrap_err()),
        ValidationError::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F64
        }
    ));

    let (shape, data) = tensor.into_vec_col_major::<f64>().unwrap();
    assert_eq!(shape.as_slice(), &[2]);
    assert_eq!(data, vec![5.0, 2.0]);

    let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1_i32]).unwrap();
    assert!(matches!(
        validation(tensor.into_vec_col_major::<i64>().unwrap_err()),
        ValidationError::DTypeMismatch {
            expected: DType::I64,
            actual: DType::I32
        }
    ));
}

#[test]
fn dynamic_col_major_and_view_metadata_ops_cover_all_variants() {
    let tensor = DefaultScalars::from_vec_col_major(vec![1, 2], vec![10_i64, 20]).unwrap();
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[10, 20]);

    let permuted = tensor.as_view().transpose_view(&[1, 0]).unwrap();
    assert_eq!(permuted.shape(), &[2, 1]);

    let sliced = tensor
        .as_view()
        .slice_view(&[
            StridedSliceSpec::new(0, Some(1), 1),
            StridedSliceSpec::new(1, Some(2), 1),
        ])
        .unwrap();
    assert_eq!(sliced.shape(), &[1, 1]);
}

#[test]
fn view_validation_reports_rank_permutation_and_slice_errors() {
    let data = [1_i32, 2, 3, 4];
    assert!(matches!(
        validation(
            TypedTensorView::<_, DynRank, Host>::from_host_slice(
                vec![2].as_slice(),
                vec![1, 2].as_slice(),
                0,
                &data
            )
            .unwrap_err()
        ),
        ValidationError::RankMismatch {
            expected: 1,
            actual: 2
        }
    ));

    let tensor =
        TypedTensor::<_, DynRank, Host>::from_host_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4])
            .unwrap();
    let view = tensor.as_view();
    assert!(view.layout().is_compact_col_major().unwrap());
    assert!(matches!(
        validation(view.transpose_view([0]).unwrap_err()),
        ValidationError::InvalidPermutationLength {
            expected: 2,
            actual: 1
        }
    ));
    assert!(matches!(
        validation(view.transpose_view([0, 2]).unwrap_err()),
        ValidationError::AxisOutOfBounds { axis: 2, rank: 2 }
    ));
    assert!(matches!(
        validation(view.reshape_view(&[3]).unwrap_err()),
        ValidationError::ShapeMismatch(ref mismatch)
            if matches!(mismatch.as_ref(), ShapeMismatch::ReshapeElementCount { from: 4, to: 3 })
    ));
    assert!(matches!(
        validation(
            view.slice_view(&[StridedSliceSpec::new(0, Some(1), 1)])
                .unwrap_err()
        ),
        ValidationError::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
    assert!(matches!(
        validation(
            view.slice_view(&[
                StridedSliceSpec::new(0, Some(3), 1),
                StridedSliceSpec::new(0, Some(1), 1),
            ])
            .unwrap_err()
        ),
        ValidationError::InvalidSliceBounds {
            start: 0,
            end: 3,
            axis_len: 2
        }
    ));
}

#[test]
fn tensor_ref_reports_tensor_and_view_metadata() {
    let tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let tensor_ref = DefaultScalarsRef::Tensor(&tensor);
    assert_eq!(tensor_ref.dtype(), DType::I64);
    assert_eq!(tensor_ref.shape(), &[2]);
    assert_eq!(tensor_ref.rank(), 1);
    assert!(!tensor_ref.is_empty());

    let view = tensor.as_view();
    let view_ref = DefaultScalarsRef::View(view);
    assert_eq!(view_ref.dtype(), DType::I64);
    assert_eq!(view_ref.shape(), &[2]);
    assert_eq!(view_ref.rank(), 1);
    assert!(!view_ref.is_empty());
}
