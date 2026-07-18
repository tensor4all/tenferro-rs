use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_einsum::Subscripts;
use tenferro_ops::ext_op::HostReference;
use tenferro_tensor::{
    core::DType as CoreDType, Error, ErrorKind, ShapeMismatch, Tensor, ValidationError,
    ValidationKind,
};

use super::{TropicalEinsumJvpOp, TropicalEinsumVjpOp};
use crate::TropicalKind;

fn matrix(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap()
}

fn host_error(
    result: std::thread::Result<tenferro_tensor::Result<Vec<Tensor>>>,
) -> Error {
    result
        .expect("host-reference validation must not panic")
        .expect_err("malformed host-reference inputs must be rejected")
}

fn assert_invalid_argument(error: &Error, op: &str) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
        } if *actual_op == op
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_dtype_mismatch(error: &Error, op: &str, expected: CoreDType, actual: CoreDType) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::DTypeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::DTypeMismatch {
                expected: actual_expected,
                actual: actual_actual,
            },
        } if *actual_op == op && *actual_expected == expected && *actual_actual == actual
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_shape_mismatch(error: &Error, op: &str, expected: &[usize], actual: &[usize]) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: actual_op,
            source: ValidationError::ShapeMismatch(payload),
        } if *actual_op == op && matches!(
            payload.as_ref(),
            ShapeMismatch::IncompatibleShapes { lhs, rhs }
                if lhs.as_slice() == expected && rhs.as_slice() == actual
        )
    ));
    let validation_source =
        std::error::Error::source(error).expect("shape mismatch must retain its validation source");
    assert!(
        std::error::Error::source(validation_source).is_some(),
        "shape mismatch must retain its typed payload source"
    );
}

#[test]
fn tropical_jvp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let op = TropicalEinsumJvpOp::new(
        TropicalKind::MaxPlus,
        Subscripts::parse("ij,jk->ik").unwrap(),
        vec![0, 1],
    );
    let lhs = matrix(vec![2, 3]);
    let rhs = matrix(vec![3, 2]);
    let valid_lhs_tangent = matrix(vec![2, 3]);
    let wrong_dtype = Tensor::from_vec_col_major(vec![3, 2], vec![1_i64; 6]).unwrap();
    let wrong_shape = matrix(vec![2, 3]);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs])
    })));
    assert_invalid_argument(&error, "tropical_einsum_jvp");

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_dtype])
    })));
    assert_dtype_mismatch(
        &error,
        "tropical_einsum_jvp",
        CoreDType::F64,
        CoreDType::I64,
    );

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_shape])
    })));
    assert_shape_mismatch(&error, "tropical_einsum_jvp", &[3, 2], &[2, 3]);
}

#[test]
fn tropical_vjp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let op = TropicalEinsumVjpOp::new(
        TropicalKind::MaxPlus,
        Subscripts::parse("ij,jk->ik").unwrap(),
        0,
    );
    let lhs = matrix(vec![2, 3]);
    let rhs = matrix(vec![3, 2]);
    let wrong_dtype = Tensor::from_vec_col_major(vec![2, 2], vec![1_i64; 4]).unwrap();
    let wrong_shape = matrix(vec![4]);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs])
    })));
    assert_invalid_argument(&error, "tropical_einsum_vjp");

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_dtype])
    })));
    assert_dtype_mismatch(
        &error,
        "tropical_einsum_vjp",
        CoreDType::F64,
        CoreDType::I64,
    );

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_shape])
    })));
    assert_shape_mismatch(&error, "tropical_einsum_vjp", &[2, 2], &[4]);
}
