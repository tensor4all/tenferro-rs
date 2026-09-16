use tenferro_tensor_core::HostTensor;

use super::{scalar_binary_into, scalar_fold, AddOp, SubOp};

fn tensor(values: &[f64]) -> HostTensor<f64> {
    HostTensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap()
}

#[test]
fn elementwise_rejects_a_destination_of_the_wrong_shape() {
    let lhs = tensor(&[1.0, 2.0]);
    let rhs = tensor(&[1.0, 2.0]);
    let mut destination = tensor(&[0.0]);

    let error = scalar_binary_into::<f64, AddOp>("add", &mut destination, &lhs, &rhs)
        .expect_err("destination shape must match");

    assert_eq!(destination.as_slice(), &[0.0]);
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Validation(tenferro_tensor_core::ValidationKind::ShapeMismatch)
    );
}

#[test]
fn elementwise_rejects_operands_of_different_shapes() {
    let lhs = tensor(&[1.0, 2.0]);
    let rhs = tensor(&[1.0]);
    let mut destination = tensor(&[0.0, 0.0]);

    let error = scalar_binary_into::<f64, AddOp>("add", &mut destination, &lhs, &rhs)
        .expect_err("operand shapes must match");

    assert_eq!(destination.as_slice(), &[0.0, 0.0]);
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Validation(tenferro_tensor_core::ValidationKind::ShapeMismatch)
    );
}

#[test]
fn elementwise_handles_an_empty_destination() {
    let lhs: HostTensor<f64> = HostTensor::from_vec_col_major(vec![0], Vec::new()).unwrap();
    let rhs: HostTensor<f64> = HostTensor::from_vec_col_major(vec![0], Vec::new()).unwrap();
    let mut destination: HostTensor<f64> =
        HostTensor::from_vec_col_major(vec![0], Vec::new()).unwrap();

    scalar_binary_into::<f64, AddOp>("add", &mut destination, &lhs, &rhs).unwrap();

    assert!(destination.as_slice().is_empty());
}

#[test]
fn fold_uses_the_caller_operation_and_initial_value() {
    let values = tensor(&[1.0, 2.0, 3.0]);

    assert_eq!(scalar_fold::<f64, AddOp>("sum", &values, 0.0).unwrap(), 6.0);
    assert_eq!(
        scalar_fold::<f64, AddOp>("sum", &values, 10.0).unwrap(),
        16.0
    );
    assert_eq!(
        scalar_fold::<f64, SubOp>("diff", &values, 0.0).unwrap(),
        -6.0
    );
}

#[test]
fn integer_arithmetic_through_the_shared_entry_point_wraps() {
    use crate::scalar_ops::{scalar_binary_into, scalar_fold, AddOp, MulOp, SubOp};
    use tenferro_tensor_core::HostTensor;

    // The preset path wraps for integers, and the shared entry points have to agree with it in
    // every build: the operator would panic on overflow in a debug build instead.
    let maximum = HostTensor::from_vec_col_major(vec![1], vec![i32::MAX]).expect("shape");
    let one = HostTensor::from_vec_col_major(vec![1], vec![1]).expect("shape");
    let mut destination = HostTensor::from_vec_col_major(vec![1], vec![0]).expect("shape");
    scalar_binary_into::<i32, AddOp>("add", &mut destination, &maximum, &one).expect("addition");
    assert_eq!(destination.as_slice(), &[i32::MIN]);

    let minimum = HostTensor::from_vec_col_major(vec![1], vec![i32::MIN]).expect("shape");
    scalar_binary_into::<i32, SubOp>("sub", &mut destination, &minimum, &one).expect("subtraction");
    assert_eq!(destination.as_slice(), &[i32::MAX]);

    let large = HostTensor::from_vec_col_major(vec![1], vec![i64::MAX]).expect("shape");
    let two = HostTensor::from_vec_col_major(vec![1], vec![2]).expect("shape");
    let mut destination = HostTensor::from_vec_col_major(vec![1], vec![0]).expect("shape");
    scalar_binary_into::<i64, MulOp>("mul", &mut destination, &large, &two).expect("product");
    assert_eq!(destination.as_slice(), &[i64::MAX.wrapping_mul(2)]);

    // The reduction folds through the same contract.
    let total = scalar_fold::<i32, AddOp>(
        "sum",
        &HostTensor::from_vec_col_major(vec![2], vec![i32::MAX, 1]).expect("shape"),
        0,
    )
    .expect("reduction");
    assert_eq!(total, i32::MIN);
}
