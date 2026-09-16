use tenferro_tensor_core::HostTensor;

use super::{scalar_binary_into, scalar_fold};

fn tensor(values: &[f64]) -> HostTensor<f64> {
    HostTensor::from_vec_col_major(vec![values.len()], values.to_vec()).unwrap()
}

#[test]
fn elementwise_rejects_a_destination_of_the_wrong_shape() {
    let lhs = tensor(&[1.0, 2.0]);
    let rhs = tensor(&[1.0, 2.0]);
    let mut destination = tensor(&[0.0]);

    let error = scalar_binary_into("add", &mut destination, &lhs, &rhs, |a, b| a + b)
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

    let error = scalar_binary_into("add", &mut destination, &lhs, &rhs, |a, b| a + b)
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

    scalar_binary_into("add", &mut destination, &lhs, &rhs, |a, b| a + b).unwrap();

    assert!(destination.as_slice().is_empty());
}

#[test]
fn fold_uses_the_caller_operation_and_initial_value() {
    let values = tensor(&[1.0, 2.0, 3.0]);

    assert_eq!(scalar_fold("sum", &values, 0.0, |a, b| a + b).unwrap(), 6.0);
    assert_eq!(
        scalar_fold("sum", &values, 10.0, |a, b| a + b).unwrap(),
        16.0
    );
    assert_eq!(
        scalar_fold("max", &values, f64::NEG_INFINITY, f64::max).unwrap(),
        3.0
    );
}
