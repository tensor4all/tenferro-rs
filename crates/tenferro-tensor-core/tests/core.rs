use num_complex::{Complex32, Complex64};
use tenferro_tensor_core::{
    col_major_strides, DType, DynRank, Error, HostTensor, HostTensorView, Rank, SliceSpec, Tensor,
    TensorLayout, TensorRank, TensorRef,
};

#[test]
fn host_tensor_uses_host_specific_public_name() {
    let tensor = HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let view: HostTensorView<'_, f64> = tensor.as_view();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0]);
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
    let shape = <DynRank as TensorRank>::shape_from_vec(vec![2, 3].into()).unwrap();
    assert_eq!(shape.as_ref(), &[2, 3]);
    assert_eq!(
        <DynRank as TensorRank>::shape_into_vec(shape).as_slice(),
        &[2, 3]
    );
}

#[test]
fn dynamic_rank_strides_roundtrip_vec() {
    let strides = <DynRank as TensorRank>::strides_from_vec(vec![1, 2, -1].into()).unwrap();
    assert_eq!(strides.as_ref(), &[1, 2, -1]);
    assert_eq!(
        <DynRank as TensorRank>::strides_into_vec(strides).as_slice(),
        &[1, 2, -1]
    );
}

#[test]
fn static_rank_shape_and_strides_roundtrip_vecs() {
    let shape = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3].into()).unwrap();
    assert_eq!(
        <Rank<2> as TensorRank>::shape_into_vec(shape).as_slice(),
        &[2, 3]
    );

    let strides = <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2].into()).unwrap();
    assert_eq!(
        <Rank<2> as TensorRank>::strides_into_vec(strides).as_slice(),
        &[1, 2]
    );
}

#[test]
fn static_rank_rejects_wrong_shape_length() {
    let err = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3, 4].into()).unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

#[test]
fn static_rank_rejects_wrong_stride_length() {
    let err = <Rank<2> as TensorRank>::strides_from_vec(vec![1, 2, 3].into()).unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
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
fn reshape_view_as_requires_compact_layout() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    let reshaped = layout.reshape_view_as::<Rank<1>>([6], 6).unwrap();
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.strides(), &[1]);

    let non_compact = TensorLayout::<Rank<2>>::from_parts([2, 3], [2, 1], 0, 6).unwrap();
    let err = non_compact.reshape_view_as::<Rank<1>>([6], 6).unwrap_err();
    assert!(matches!(err, Error::NonContiguousViewAsSlice));
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
    assert!(matches!(err, Error::InvalidSliceStep { step: 0 }));
}

#[test]
fn slice_view_rejects_invalid_normalized_bounds_with_exact_error() {
    let layout = TensorLayout::<Rank<1>>::compact([4]).unwrap();
    let err = layout
        .slice_view(
            [SliceSpec {
                start: 3,
                end: -2,
                step: -1,
            }],
            4,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidSliceBounds {
            start: 3,
            end: -2,
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
    let err =
        TensorLayout::<DynRank>::from_parts(vec![2, 3].into(), vec![1].into(), 0, 6).unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn scalar_layout_with_static_rank_zero_is_compact() {
    let layout = TensorLayout::<Rank<0>>::compact([]).unwrap();
    assert_eq!(layout.shape(), &[]);
    assert_eq!(layout.strides(), &[]);
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
    assert!(matches!(err, Error::IntegerOverflow));
}

#[test]
fn layout_rejects_non_empty_broadcast_shape_product_overflow() {
    let huge_extent = usize::MAX / 2 + 1;
    let err = TensorLayout::<DynRank>::from_parts(
        vec![huge_extent, huge_extent].into(),
        vec![0, 0].into(),
        0,
        1,
    )
    .unwrap_err();
    assert!(matches!(err, Error::IntegerOverflow));
}

#[test]
fn compact_host_tensor_view_reuses_checked_col_major_stride_helper() {
    let source = include_str!("../src/lib.rs");
    let helper = source
        .split("fn compact_col_major_strides")
        .nth(1)
        .and_then(|rest| rest.split("/// Return compact column-major strides").next())
        .expect("compact_col_major_strides helper should exist");

    assert!(
        helper.contains("col_major_strides(shape)"),
        "compact host view strides must not duplicate unchecked stride arithmetic"
    );
    assert!(
        !helper.contains("stride *= extent as isize"),
        "compact host view strides must avoid unchecked stride multiplication"
    );
}

#[test]
fn layout_from_parts_preserves_offset() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![1].into(), 7, 10).unwrap();
    assert_eq!(layout.shape(), &[3]);
    assert_eq!(layout.strides(), &[1]);
    assert_eq!(layout.offset(), 7);
}

#[test]
fn layout_accepts_negative_stride_when_reachable_range_is_in_bounds() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 2, 3).unwrap();
    assert_eq!(layout.shape(), &[3]);
    assert_eq!(layout.strides(), &[-1]);
    assert_eq!(layout.offset(), 2);
}

#[test]
fn layout_rejects_negative_stride_when_reachable_range_is_out_of_bounds() {
    assert!(TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 1, 3).is_err());
}

#[test]
fn layout_rejects_positive_stride_when_max_offset_exceeds_buffer_len() {
    assert!(TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![2].into(), 0, 3).is_err());
}

#[test]
fn layout_accepts_empty_shape_offsets_at_buffer_boundaries() {
    let at_start =
        TensorLayout::<DynRank>::from_parts(vec![0].into(), vec![1].into(), 0, 3).unwrap();
    assert_eq!(at_start.offset(), 0);

    let at_end = TensorLayout::<DynRank>::from_parts(vec![0].into(), vec![1].into(), 3, 3).unwrap();
    assert_eq!(at_end.offset(), 3);

    assert!(TensorLayout::<DynRank>::from_parts(vec![0].into(), vec![1].into(), 4, 3).is_err());
}

#[test]
fn layout_reports_overflow_for_unreachable_offset_arithmetic() {
    assert!(matches!(
        TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![isize::MAX].into(), 0, 3)
            .unwrap_err(),
        Error::IntegerOverflow
    ));
}

#[test]
fn mutable_layout_rejects_zero_stride_broadcast() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![2].into(), vec![0].into(), 0, 1).unwrap();
    assert!(matches!(
        layout.validate_mutable_no_overlap(),
        Err(Error::OverlappingMutableLayout)
    ));
}

#[test]
fn mutable_layout_rejects_overlapping_strides() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![2, 2].into(), vec![1, 1].into(), 0, 4).unwrap();
    assert!(matches!(
        layout.validate_mutable_no_overlap(),
        Err(Error::OverlappingMutableLayout)
    ));
}

#[test]
fn mutable_layout_accepts_reversed_non_overlapping_vector() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 2, 3).unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_accepts_empty_view_before_product_overflow() {
    let layout = TensorLayout::<DynRank>::from_parts(
        vec![2, usize::MAX, 0].into(),
        vec![1, 1, 1].into(),
        0,
        0,
    )
    .unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_exact_fallback_accepts_small_ambiguous_non_overlapping_layout() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![3, 2].into(), vec![2, 3].into(), 0, 8).unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}

#[test]
fn mutable_layout_rejects_large_ambiguous_layout_without_exact_fallback() {
    let layout =
        TensorLayout::<DynRank>::from_parts(vec![4097, 2].into(), vec![2, 4097].into(), 0, 12_290)
            .unwrap();
    assert!(matches!(
        layout.validate_mutable_no_overlap(),
        Err(Error::OverlappingMutableLayout)
    ));
}

#[test]
fn layout_rejects_huge_non_empty_zero_stride_before_product_overflow() {
    let huge_extent = isize::MAX as usize + 1;
    assert!(matches!(
        TensorLayout::<DynRank>::from_parts(vec![huge_extent, 3].into(), vec![0, 1].into(), 0, 3),
        Err(Error::IntegerOverflow)
    ));
}

#[test]
fn constructs_contiguous_col_major_and_validates_count() {
    let tensor = HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let err = HostTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64]).unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeDataLengthMismatch {
            expected: 4,
            actual: 1
        }
    ));
}

#[test]
fn owned_view_has_expected_shape_stride_offset() {
    let tensor = HostTensor::from_vec_col_major(vec![2, 3], vec![0_i64; 6]).unwrap();
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.offset(), 0);
    assert!(view.is_compact_col_major().unwrap());
    assert!(view.is_zero_offset_col_major().unwrap());
}

#[test]
fn reshape_view_requires_compact_col_major_but_allows_nonzero_offset() {
    let data = [10_i32, 11, 12, 13, 14, 15];
    let compact_nonzero = HostTensorView::from_slice(vec![4], vec![1], 1, &data).unwrap();
    let reshaped = compact_nonzero.reshape_view(vec![2, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[2, 2]);
    assert_eq!(reshaped.offset(), 1);
    assert_eq!(reshaped.as_slice().unwrap(), &[11, 12, 13, 14]);

    let non_contiguous = HostTensorView::from_slice(vec![2, 2], vec![1, 3], 0, &data).unwrap();
    let err = non_contiguous.reshape_view(vec![4]).unwrap_err();
    assert!(matches!(err, Error::NonContiguousViewAsSlice));
}

#[test]
fn transpose_view_only_reorders_metadata() {
    let tensor = HostTensor::from_vec_col_major(vec![2, 3], vec![0_i64; 6]).unwrap();
    let view = tensor.as_view().transpose_view(&[1, 0]).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.strides(), &[2, 1]);
    assert_eq!(view.offset(), 0);
    assert!(matches!(
        tensor.as_view().transpose_view(&[0, 0]).unwrap_err(),
        Error::DuplicateAxis { axis: 0 }
    ));
}

#[test]
fn slice_view_positive_step_and_empty_slice() {
    let tensor = HostTensor::from_vec_col_major(vec![5], vec![1_i64, 2, 3, 4, 5]).unwrap();
    let view = tensor
        .as_view()
        .slice_view(&[SliceSpec {
            start: 1,
            end: 5,
            step: 2,
        }])
        .unwrap();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(view.strides(), &[2]);
    assert_eq!(view.offset(), 1);

    let empty = tensor
        .as_view()
        .slice_view(&[SliceSpec {
            start: 4,
            end: 2,
            step: 1,
        }])
        .unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.as_slice().unwrap(), &[]);
}

#[test]
fn invalid_slice_steps_and_negative_bounds_are_rejected() {
    let tensor = HostTensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]).unwrap();
    assert!(matches!(
        tensor
            .as_view()
            .slice_view(&[SliceSpec {
                start: 0,
                end: 2,
                step: 0
            }])
            .unwrap_err(),
        Error::InvalidSliceStep { step: 0 }
    ));
    assert!(matches!(
        tensor
            .as_view()
            .slice_view(&[SliceSpec {
                start: -1,
                end: 2,
                step: 1
            }])
            .unwrap_err(),
        Error::InvalidSliceBounds { .. }
    ));
}

#[test]
fn view_bounds_are_validated_eagerly_with_checked_arithmetic() {
    let data = [1_i32, 2, 3];
    assert!(matches!(
        HostTensorView::from_slice(vec![4], vec![1], 0, &data).unwrap_err(),
        Error::ViewOutOfBounds
    ));
    assert!(matches!(
        HostTensorView::from_slice(vec![usize::MAX, 2], vec![1, 2], 0, &data).unwrap_err(),
        Error::IntegerOverflow
    ));

    let reversed = HostTensorView::from_slice(vec![3], vec![-1], 2, &data).unwrap();
    assert_eq!(reversed.shape(), &[3]);
    assert_eq!(reversed.strides(), &[-1]);
    assert_eq!(reversed.offset(), 2);
    assert!(matches!(
        HostTensorView::from_slice(vec![3], vec![-1], 1, &data).unwrap_err(),
        Error::ViewOutOfBounds
    ));
}

#[test]
fn empty_view_offsets_may_point_one_past_the_borrowed_slice() {
    let data = [1_i32, 2, 3];
    let empty = HostTensorView::from_slice(vec![0], vec![-1], 3, &data).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.as_slice().unwrap(), &[]);

    assert!(matches!(
        HostTensorView::from_slice(vec![0], vec![1], 4, &data).unwrap_err(),
        Error::ViewOutOfBounds
    ));
}

#[test]
fn empty_views_are_contiguous_even_with_degenerate_strides() {
    let data = [1_i32, 2, 3];
    let empty = HostTensorView::from_slice(vec![0, 3], vec![1, 0], 3, &data).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.as_slice().unwrap(), &[]);
    assert_eq!(empty.reshape_view(vec![0]).unwrap().shape(), &[0]);

    let layout =
        TensorLayout::<DynRank>::from_parts(vec![0, 3].into(), vec![99, -7].into(), 3, data.len())
            .unwrap();
    assert!(layout.is_compact_col_major().unwrap());
    assert_eq!(
        layout
            .reshape_view_as::<DynRank>(vec![0].into(), data.len())
            .unwrap()
            .shape(),
        &[0]
    );
}

#[test]
fn empty_axis_allows_negative_step_slices() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![0].into(), vec![1].into(), 0, 0).unwrap();
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
    let contiguous = HostTensorView::from_slice(vec![3], vec![1], 1, &data).unwrap();
    assert_eq!(contiguous.as_slice().unwrap(), &[2, 3, 4]);

    let non_contiguous = HostTensorView::from_slice(vec![2], vec![2], 0, &data).unwrap();
    assert!(matches!(
        non_contiguous.as_slice().unwrap_err(),
        Error::NonContiguousViewAsSlice
    ));
}

#[test]
fn dynamic_tensor_and_view_report_dtype_mismatch() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    assert_eq!(tensor.dtype(), DType::F64);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    assert!(matches!(
        tensor.as_slice::<f32>().unwrap_err(),
        Error::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F64
        }
    ));
    assert_eq!(
        tensor.as_view().reshape_view(vec![1, 2]).unwrap().shape(),
        &[1, 2]
    );
}

#[test]
fn layout_helpers_and_col_major_roundtrip() {
    assert_eq!(col_major_strides(&[2, 3]).unwrap().as_slice(), &[1, 2]);
    let tensor = HostTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 3, 2, 4]).unwrap();
    assert_eq!(tensor.as_slice(), &[1, 3, 2, 4]);
    assert_eq!(tensor.into_vec_col_major().1, vec![1, 3, 2, 4]);
}

#[test]
fn typed_owned_accessors_and_exports_cover_success_and_errors() {
    let mut tensor = HostTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    assert_eq!(tensor.rank(), 1);
    assert!(!tensor.is_empty());
    tensor.as_mut_slice()[1] = 3;
    assert_eq!(tensor.as_slice(), &[1, 3]);

    let reshaped = tensor.clone().into_reshaped(vec![1, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[1, 2]);
    assert!(matches!(
        tensor.into_reshaped(vec![3]).unwrap_err(),
        Error::ReshapeElementCountMismatch { from: 2, to: 3 }
    ));

    let (shape, data) = reshaped.into_vec_col_major();
    assert_eq!(shape.as_slice(), &[1, 2]);
    assert_eq!(data, vec![1, 3]);

    let empty = HostTensor::<f64>::from_vec_col_major(vec![0], vec![]).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn scalar_col_major_helpers_cover_scalar_and_empty_shapes() {
    let scalar = HostTensor::from_vec_col_major(vec![], vec![7_i64]).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.into_vec_col_major().1, vec![7]);

    let empty = HostTensor::<i64>::from_vec_col_major(vec![0, 3], vec![]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.into_vec_col_major().1, Vec::<i64>::new());
}

#[test]
fn dynamic_tensor_accessors_cover_all_dtype_variants() {
    let tensors = [
        Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![3_i32]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![4_i64]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![true]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![Complex32::new(1.0, 2.0)]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)]).unwrap(),
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

    let empty = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();
    assert!(empty.is_empty());
    assert!(empty.as_view().is_empty());
}

#[test]
fn dynamic_tensor_mutation_and_owned_exports_validate_dtype() {
    let mut tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    tensor.as_mut_slice::<f64>().unwrap()[0] = 5.0;
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[5.0, 2.0]);
    assert!(matches!(
        tensor.as_mut_slice::<f32>().unwrap_err(),
        Error::DTypeMismatch {
            expected: DType::F32,
            actual: DType::F64
        }
    ));

    let (shape, data) = tensor.into_vec_col_major::<f64>().unwrap();
    assert_eq!(shape.as_slice(), &[2]);
    assert_eq!(data, vec![5.0, 2.0]);

    let tensor = Tensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap();
    assert!(matches!(
        tensor.into_vec_col_major::<i64>().unwrap_err(),
        Error::DTypeMismatch {
            expected: DType::I64,
            actual: DType::I32
        }
    ));
}

#[test]
fn dynamic_col_major_and_view_metadata_ops_cover_all_variants() {
    let tensor = Tensor::from_vec_col_major(vec![1, 2], vec![10_i64, 20]).unwrap();
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[10, 20]);

    let permuted = tensor.as_view().transpose_view(&[1, 0]).unwrap();
    assert_eq!(permuted.shape(), &[2, 1]);

    let sliced = tensor
        .as_view()
        .slice_view(&[
            SliceSpec {
                start: 0,
                end: 1,
                step: 1,
            },
            SliceSpec {
                start: 1,
                end: 2,
                step: 1,
            },
        ])
        .unwrap();
    assert_eq!(sliced.shape(), &[1, 1]);
}

#[test]
fn view_validation_reports_rank_permutation_and_slice_errors() {
    let data = [1_i32, 2, 3, 4];
    assert!(matches!(
        HostTensorView::from_slice(vec![2], vec![1, 2], 0, &data).unwrap_err(),
        Error::RankMismatch {
            expected: 1,
            actual: 2
        }
    ));

    let tensor = HostTensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let view = tensor.as_view();
    assert!(view.is_compact_col_major().unwrap());
    assert!(matches!(
        view.transpose_view(&[0]).unwrap_err(),
        Error::InvalidPermutationLength {
            expected: 2,
            actual: 1
        }
    ));
    assert!(matches!(
        view.transpose_view(&[0, 2]).unwrap_err(),
        Error::AxisOutOfBounds { axis: 2, rank: 2 }
    ));
    assert!(matches!(
        view.reshape_view(vec![3]).unwrap_err(),
        Error::ReshapeElementCountMismatch { from: 4, to: 3 }
    ));
    assert!(matches!(
        view.slice_view(&[SliceSpec {
            start: 0,
            end: 1,
            step: 1
        }])
        .unwrap_err(),
        Error::RankMismatch {
            expected: 2,
            actual: 1
        }
    ));
    assert!(matches!(
        view.slice_view(&[
            SliceSpec {
                start: 0,
                end: 3,
                step: 1,
            },
            SliceSpec {
                start: 0,
                end: 1,
                step: 1,
            },
        ])
        .unwrap_err(),
        Error::InvalidSliceBounds {
            start: 0,
            end: 3,
            axis_len: 2
        }
    ));
}

#[test]
fn tensor_ref_reports_tensor_and_view_metadata() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let tensor_ref = TensorRef::Tensor(&tensor);
    assert_eq!(tensor_ref.dtype(), DType::I64);
    assert_eq!(tensor_ref.shape(), &[2]);
    assert_eq!(tensor_ref.rank(), 1);
    assert!(!tensor_ref.is_empty());

    let view = tensor.as_view();
    let view_ref = TensorRef::View(view);
    assert_eq!(view_ref.dtype(), DType::I64);
    assert_eq!(view_ref.shape(), &[2]);
    assert_eq!(view_ref.rank(), 1);
    assert!(!view_ref.is_empty());
}
