use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_einsum::Subscripts;
use tenferro_ops::ext_op::HostReference;
use tenferro_tensor::{Error, Tensor};

use super::{TropicalEinsumJvpOp, TropicalEinsumVjpOp};
use crate::TropicalKind;

fn matrix(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap()
}

fn assert_invalid_without_panic(result: std::thread::Result<tenferro_tensor::Result<Vec<Tensor>>>) {
    let error = result
        .expect("host-reference validation must not panic")
        .expect_err("malformed host-reference inputs must be rejected");
    assert!(matches!(
        error,
        Error::Validation {
            source: tenferro_tensor::ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
            ..
        }
    ));
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
