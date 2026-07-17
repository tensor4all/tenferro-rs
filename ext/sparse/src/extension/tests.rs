use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ops::ext_op::HostReference;
use tenferro_tensor::{Error, ErrorKind, Tensor, ValidationKind};

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

fn assert_invalid_without_panic(result: std::thread::Result<tenferro_tensor::Result<Vec<Tensor>>>) {
    let error = result
        .expect("host-reference validation must not panic")
        .expect_err("malformed host-reference inputs must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation(ValidationKind::InvalidArgument));
    assert!(matches!(error, Error::Validation { .. }));
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

    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs])
    })));
    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_dtype])
    })));
    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &valid_lhs_tangent, &wrong_shape])
    })));
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

    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs])
    })));
    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_dtype])
    })));
    assert_invalid_without_panic(catch_unwind(AssertUnwindSafe(|| {
        HostReference::execute(&op, &[&lhs, &rhs, &wrong_shape])
    })));
}
