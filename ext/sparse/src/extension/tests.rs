use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ops::{ext_op::HostReference, SymDim};
use tenferro_tensor::{
    core::DType as CoreDType, DType, Error, ErrorKind, ShapeMismatch, Tensor, ValidationError,
    ValidationKind,
};

use super::{SparseMatmulJvpOp, SparseMatmulPlan, SparseMatmulVjpOp};

fn plan() -> SparseMatmulPlan {
    SparseMatmulPlan::new(
        &[2, 2],
        &[[0, 0], [0, 1], [1, 0]],
        &[2, 2],
        &[[0, 0], [1, 0], [0, 1]],
    )
    .unwrap()
}

fn f64_values(len: usize) -> Tensor {
    Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap()
}

fn host_error(
    result: std::thread::Result<tenferro_tensor::Result<Vec<Tensor>>>,
) -> Error {
    result
        .expect("host-reference validation must not panic")
        .expect_err("malformed host-reference inputs must be rejected")
}

fn assert_invalid_argument(error: &Error) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
        }
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_dtype_mismatch(error: &Error, expected: CoreDType, actual: CoreDType) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::DTypeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::DTypeMismatch {
                expected: actual_expected,
                actual: actual_actual,
            },
        } if *actual_expected == expected && *actual_actual == actual
    ));
    assert!(std::error::Error::source(error).is_some());
}

fn assert_shape_mismatch(error: &Error, expected: &[usize], actual: &[usize]) {
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::ShapeMismatch(payload),
        } if matches!(
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
fn sparse_jvp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let plan = plan();
    let op = SparseMatmulJvpOp {
        plan,
        active_inputs: vec![0, 1],
    };
    let lhs = f64_values(3);
    let rhs = f64_values(3);
    let valid_lhs_tangent = f64_values(3);
    let wrong_dtype = Tensor::from_vec_col_major(vec![3], vec![1_i64; 3]).unwrap();
    let wrong_shape = f64_values(4);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs])
    })));
    assert_invalid_argument(&error);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_dtype])
    })));
    assert_dtype_mismatch(&error, CoreDType::F64, CoreDType::I64);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_shape])
    })));
    assert_shape_mismatch(&error, &[3], &[4]);
}

#[test]
fn sparse_vjp_host_boundary_rejects_count_dtype_and_exact_shape() {
    let plan = plan();
    let output_nnz = plan.output_nnz();
    let op = SparseMatmulVjpOp {
        plan,
        active_input: 0,
    };
    let lhs = f64_values(3);
    let rhs = f64_values(3);
    let wrong_dtype =
        Tensor::from_vec_col_major(vec![output_nnz], vec![1_i64; output_nnz]).unwrap();
    let wrong_shape = f64_values(output_nnz + 1);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs])
    })));
    assert_invalid_argument(&error);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_dtype])
    })));
    assert_dtype_mismatch(&error, CoreDType::F64, CoreDType::I64);

    let error = host_error(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_shape])
    })));
    assert_shape_mismatch(&error, &[output_nnz], &[output_nnz + 1]);
}

#[test]
fn sparse_metadata_validation_preserves_dtype_and_rank_payloads() {
    let shape = [SymDim::from(3_usize)];
    let dtype_error = super::validate_primal_meta(
        &[DType::I64, DType::F64],
        &[&shape[..], &shape[..]],
    )
    .unwrap_err();
    assert_dtype_mismatch(&dtype_error, CoreDType::F64, CoreDType::I64);

    let rank_shape = [SymDim::from(3_usize), SymDim::from(1_usize)];
    let rank_error = super::validate_primal_meta(
        &[DType::F64, DType::F64],
        &[&rank_shape[..], &shape[..]],
    )
    .unwrap_err();
    assert_eq!(
        rank_error.kind(),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
    assert!(matches!(
        rank_error,
        Error::Validation {
            op: "tenferro-ext-sparse",
            source: ValidationError::RankMismatch {
                expected: 1,
                actual: 2,
            },
        }
    ));
}
