use num_complex::{Complex32, Complex64};
use tenferro_tensor_core::{
    col_major_strides, DType, Error, SliceSpec, Tensor, TensorRef, TypedTensor, TypedTensorView,
};

#[test]
fn constructs_contiguous_col_major_and_validates_count() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

    let err = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64]).unwrap_err();
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
    let tensor = TypedTensor::from_vec_col_major(vec![2, 3], vec![0_i64; 6]).unwrap();
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.offset(), 0);
    assert!(view.is_compact_col_major());
    assert!(view.is_zero_offset_col_major());
}

#[test]
fn reshape_view_requires_compact_col_major_but_allows_nonzero_offset() {
    let data = [10_i32, 11, 12, 13, 14, 15];
    let compact_nonzero = TypedTensorView::from_slice(vec![4], vec![1], 1, &data).unwrap();
    let reshaped = compact_nonzero.reshape_view(vec![2, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[2, 2]);
    assert_eq!(reshaped.offset(), 1);
    assert_eq!(reshaped.as_slice().unwrap(), &[11, 12, 13, 14]);

    let non_contiguous = TypedTensorView::from_slice(vec![2, 2], vec![1, 3], 0, &data).unwrap();
    let err = non_contiguous.reshape_view(vec![4]).unwrap_err();
    assert!(matches!(err, Error::NonContiguousViewAsSlice));
}

#[test]
fn permute_view_only_reorders_metadata() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 3], vec![0_i64; 6]).unwrap();
    let view = tensor.as_view().permute_view(&[1, 0]).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.strides(), &[2, 1]);
    assert_eq!(view.offset(), 0);
    assert!(matches!(
        tensor.as_view().permute_view(&[0, 0]).unwrap_err(),
        Error::DuplicateAxis { axis: 0 }
    ));
}

#[test]
fn slice_view_positive_step_and_empty_slice() {
    let tensor = TypedTensor::from_vec_col_major(vec![5], vec![1_i64, 2, 3, 4, 5]).unwrap();
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
    let tensor = TypedTensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]).unwrap();
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
        TypedTensorView::from_slice(vec![4], vec![1], 0, &data).unwrap_err(),
        Error::ViewOutOfBounds
    ));
    assert!(matches!(
        TypedTensorView::from_slice(vec![usize::MAX, 2], vec![1, 2], 0, &data).unwrap_err(),
        Error::IntegerOverflow
    ));
    assert!(matches!(
        TypedTensorView::from_slice(vec![1], vec![-1], 0, &data).unwrap_err(),
        Error::ViewOutOfBounds
    ));
}

#[test]
fn as_slice_accepts_nonzero_offset_and_rejects_non_contiguous_views() {
    let data = [1_i32, 2, 3, 4, 5];
    let contiguous = TypedTensorView::from_slice(vec![3], vec![1], 1, &data).unwrap();
    assert_eq!(contiguous.as_slice().unwrap(), &[2, 3, 4]);

    let non_contiguous = TypedTensorView::from_slice(vec![2], vec![2], 0, &data).unwrap();
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
fn layout_helpers_and_row_major_roundtrip() {
    assert_eq!(col_major_strides(&[2, 3]).unwrap().as_slice(), &[1, 2]);
    let tensor = TypedTensor::from_vec_row_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap();
    assert_eq!(tensor.as_slice(), &[1, 3, 2, 4]);
    assert_eq!(tensor.into_vec_row_major().unwrap().1, vec![1, 2, 3, 4]);
}

#[test]
fn typed_owned_accessors_and_exports_cover_success_and_errors() {
    let mut tensor = TypedTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
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

    let empty = TypedTensor::<f64>::from_vec_col_major(vec![0], vec![]).unwrap();
    assert!(empty.is_empty());
}

#[test]
fn scalar_row_major_helpers_cover_scalar_and_empty_shapes() {
    let scalar = TypedTensor::from_vec_row_major(vec![], vec![7_i64]).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.into_vec_row_major().unwrap().1, vec![7]);

    let empty = TypedTensor::<i64>::from_vec_row_major(vec![0, 3], vec![]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.into_vec_row_major().unwrap().1, Vec::<i64>::new());
}

#[test]
fn dynamic_tensor_accessors_cover_all_dtype_variants() {
    let tensors = vec![
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
fn dynamic_row_major_and_view_metadata_ops_cover_all_variants() {
    let tensor = Tensor::from_vec_row_major(vec![1, 2], vec![10_i64, 20]).unwrap();
    assert_eq!(tensor.shape(), &[1, 2]);
    assert_eq!(tensor.as_slice::<i64>().unwrap(), &[10, 20]);

    let permuted = tensor.as_view().permute_view(&[1, 0]).unwrap();
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
        TypedTensorView::from_slice(vec![2], vec![1, 2], 0, &data).unwrap_err(),
        Error::RankMismatch {
            expected: 1,
            actual: 2
        }
    ));

    let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
    let view = tensor.as_view();
    assert!(view.is_contiguous_col_major());
    assert!(matches!(
        view.permute_view(&[0]).unwrap_err(),
        Error::InvalidPermutationLength {
            expected: 2,
            actual: 1
        }
    ));
    assert!(matches!(
        view.permute_view(&[0, 2]).unwrap_err(),
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
