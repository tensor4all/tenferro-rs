use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{
    dynamic_slice, f32_index_to_i64, f64_index_to_i64, inline_col_major_strides, typed_concatenate,
    BufferPool,
};
use strided_kernel::ExecContext;
use tenferro_tensor::{Error, ErrorKind, Tensor, TypedTensor, ValidationError, ValidationKind};

#[test]
fn typed_concatenate_rejects_empty_typed_inputs_without_panicking() {
    let mut buffers = BufferPool::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        typed_concatenate::<f64>(&mut buffers, &ExecContext::serial(), &[], 0)
    }));

    assert!(result.is_ok(), "empty typed concatenate should return Err");
    assert!(matches!(
        result.unwrap().unwrap_err(),
        Error::Validation {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn typed_concatenate_accepts_nonempty_typed_inputs() {
    let mut buffers = BufferPool::new();
    let a = TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let b = TypedTensor::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let inputs = vec![&a, &b];

    let out = typed_concatenate(&mut buffers, &ExecContext::serial(), &inputs, 0).unwrap();

    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.host_data().unwrap(), &[1.0, 2.0, 3.0]);
}

#[test]
fn inline_strides_cover_zero_extent_and_spill_without_overflow() {
    let dims = [2, 0, usize::MAX, 3, 5, 7, 11, 13, 17];
    let strides = inline_col_major_strides("test_static_indexing", &dims).unwrap();

    assert_eq!(strides.len(), 9);
    assert_eq!(&strides[..4], &[1, 2, 0, 0]);
}

#[test]
fn inline_strides_report_isize_overflow_as_typed_error() {
    let error = inline_col_major_strides("test_static_indexing", &[usize::MAX, 2])
        .expect_err("oversized dimensions must not panic or wrap");

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
}

#[test]
fn float_index_validation_identifies_the_index_argument() {
    for error in [
        f32_index_to_i64(1.5).unwrap_err(),
        f64_index_to_i64(f64::NAN).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            Error::Validation {
                op: "index_tensor",
                source: ValidationError::InvalidArgument {
                    argument: "index",
                    ..
                },
            }
        ));
    }
}

#[test]
fn dynamic_slice_validation_identifies_the_starts_argument() {
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let wrong_rank = Tensor::from_vec_col_major(vec![], vec![0_i64]).unwrap();
    let wrong_length = Tensor::from_vec_col_major(vec![2], vec![0_i64, 1]).unwrap();

    for starts in [&wrong_rank, &wrong_length] {
        let error = dynamic_slice(&input, starts, &[1]).unwrap_err();
        assert!(matches!(
            error,
            Error::Validation {
                op: "dynamic_slice",
                source: ValidationError::InvalidArgument {
                    argument: "starts",
                    ..
                },
            }
        ));
    }
}
