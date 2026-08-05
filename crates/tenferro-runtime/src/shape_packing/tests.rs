use std::ops::Range;

use tenferro_tensor::{DType, ErrorKind, ValidationError, ValidationKind};

use super::{
    apply_slice_axis_config, concrete_shape_for_axis_slice, index_select_config,
    normalize_existing_axis, normalize_insert_axis, validate_axis_selection, validate_stack_shapes,
    AxisSelection,
};
use crate::TracedTensor;

#[test]
fn axis_normalization_handles_ranks_larger_than_isize_max() {
    assert_eq!(normalize_existing_axis("test", 0, usize::MAX).unwrap(), 0);
    assert_eq!(
        normalize_existing_axis("test", -1, usize::MAX).unwrap(),
        usize::MAX - 1
    );
    assert_eq!(
        normalize_insert_axis("test", -1, usize::MAX - 1).unwrap(),
        usize::MAX - 1
    );
    assert!(normalize_insert_axis("test", -1, usize::MAX).is_err());
}

#[test]
fn axis_normalization_reports_positive_and_negative_bounds() {
    assert_eq!(normalize_existing_axis("test", 1, 3).unwrap(), 1);
    assert_eq!(normalize_existing_axis("test", -1, 3).unwrap(), 2);
    assert_eq!(normalize_insert_axis("test", 3, 3).unwrap(), 3);
    assert_eq!(normalize_insert_axis("test", -1, 3).unwrap(), 3);

    for error in [
        normalize_existing_axis("test", 3, 3).unwrap_err(),
        normalize_existing_axis("test", -4, 3).unwrap_err(),
        normalize_insert_axis("test", 4, 3).unwrap_err(),
        normalize_insert_axis("test", -5, 3).unwrap_err(),
    ] {
        assert_eq!(
            error.kind(),
            ErrorKind::Validation(ValidationKind::AxisOutOfBounds)
        );
    }
}

#[test]
fn index_select_config_validates_positions_and_builds_gather_metadata() {
    let (indices, config, output_shape) = index_select_config(&[3, 4], -1, &[3, 1]).unwrap();
    assert_eq!(indices.as_slice::<i64>().unwrap(), &[3, 1]);
    assert_eq!(output_shape, vec![3, 2]);
    assert_eq!(config.collapsed_slice_dims, vec![1]);
    assert_eq!(config.offset_dims, vec![0]);
    assert_eq!(config.start_index_map, vec![1]);
    assert_eq!(config.slice_sizes, vec![3, 1]);

    let error = index_select_config(&[3], 0, &[3]).unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "positions",
                ..
            },
            ..
        }
    ));
}

#[test]
fn stack_shape_and_selection_helpers_cover_validation_contracts() {
    let first = [2usize, 3];
    let same = [2usize, 3];
    let different = [2usize, 4];
    assert!(validate_stack_shapes("stack", &[&first, &same]).is_ok());
    assert!(matches!(
        validate_stack_shapes("stack", &[]).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "tensors",
                ..
            },
            ..
        }
    ));
    assert!(matches!(
        validate_stack_shapes("stack", &[&first, &different]).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::ShapeMismatch(_),
            ..
        }
    ));

    let mut seen = vec![false; 2];
    validate_axis_selection("slice", 2, &mut seen, 1).unwrap();
    assert!(matches!(
        validate_axis_selection("slice", 2, &mut seen, 1).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::DuplicateAxis { axis: 1, .. },
            ..
        }
    ));
    assert!(matches!(
        validate_axis_selection("slice", 2, &mut seen, 2).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
            ..
        }
    ));
}

#[test]
fn slice_config_distinguishes_take_only_from_ranges_and_rejects_bad_ranges() {
    let take_only = [AxisSelection::Take {
        axis: 0,
        indices: vec![1],
    }];
    assert!(apply_slice_axis_config("slice", &[3], &take_only)
        .unwrap()
        .is_none());

    let ranges = [AxisSelection::Slice {
        axis: 0,
        range: 1..3,
        step: 2,
    }];
    assert_eq!(
        apply_slice_axis_config("slice", &[4], &ranges)
            .unwrap()
            .unwrap()
            .strides,
        vec![2]
    );

    let zero_step = [AxisSelection::Slice {
        axis: 0,
        range: 0..1,
        step: 0,
    }];
    assert!(matches!(
        apply_slice_axis_config("slice", &[2], &zero_step).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::InvalidSliceStep { step: 0 },
            ..
        }
    ));

    let bad_bounds = [AxisSelection::Slice {
        axis: 0,
        range: Range { start: 2, end: 4 },
        step: 1,
    }];
    assert!(matches!(
        apply_slice_axis_config("slice", &[3], &bad_bounds).unwrap_err(),
        crate::Error::Validation {
            source: ValidationError::InvalidSliceBounds { axis_len: 3, .. },
            ..
        }
    ));
}

#[test]
fn public_slice_stack_and_concatenate_paths_preserve_structured_errors() {
    let x =
        TracedTensor::from_vec_col_major(vec![4, 5], (0..20).map(|value| value as f64).collect())
            .unwrap();
    let sliced = x
        .slice_builder()
        .axis(0, 1..4)
        .take_axis(1, &[4, 0])
        .apply()
        .unwrap();
    assert_eq!(sliced.try_concrete_shape(), Some(vec![3, 2]));

    assert!(matches!(
        x.slice_builder()
            .axis(0, 0..2)
            .axis_step(0, 0..2, 1)
            .apply(),
        Err(crate::Error::Validation {
            source: ValidationError::DuplicateAxis { .. },
            ..
        })
    ));

    let symbolic = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    assert!(matches!(
        concrete_shape_for_axis_slice(&symbolic, "slice_builder"),
        Err(crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "shape",
                ..
            },
            ..
        })
    ));
    assert!(matches!(
        symbolic.index_select(0, &[0]),
        Err(crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "shape",
                ..
            },
            ..
        })
    ));

    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let stacked = TracedTensor::stack(&[&a, &b], -1).unwrap();
    assert_eq!(stacked.try_concrete_shape(), Some(vec![2, 2]));
    assert!(matches!(
        TracedTensor::stack(&[], 0),
        Err(crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "tensors",
                ..
            },
            ..
        })
    ));
    assert!(matches!(
        TracedTensor::stack(&[&a, &x], 0),
        Err(crate::Error::Validation {
            source: ValidationError::ShapeMismatch(_),
            ..
        })
    ));
    assert!(matches!(
        TracedTensor::stack(&[&a, &b], 2),
        Err(crate::Error::Validation {
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        })
    ));

    let c = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f32, 4.0]).unwrap();
    let concatenated = TracedTensor::concatenate(&[&a, &c], 0).unwrap();
    assert_eq!(concatenated.dtype, DType::F64);
    assert_eq!(concatenated.try_concrete_shape(), Some(vec![4]));
    assert!(matches!(
        TracedTensor::concatenate(&[], 0),
        Err(crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "tensors",
                ..
            },
            ..
        })
    ));
    assert!(matches!(
        TracedTensor::concatenate(&[&a, &b], 1),
        Err(crate::Error::Validation {
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        })
    ));
    let rank_one = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
    assert!(matches!(
        TracedTensor::concatenate(&[&a, &rank_one], 0),
        Err(crate::Error::Validation {
            source: ValidationError::RankMismatch { .. },
            ..
        })
    ));
}
