use super::*;
use tenferro_tensor::MemoryOrder;

fn scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix(values: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[rows, cols], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn from_diagonal_vector_rejects_invalid_payload_rank_and_zero_logical_rank() {
    let err =
        StructuredTensor::from_diagonal_vector(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2), 2).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let err = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 0).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn scalar_layout_is_not_diag_and_as_ref_exposes_payload() {
    let scalar_layout = StructuredTensor::from_dense(scalar(3.0));
    assert!(!scalar_layout.is_diag());

    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    assert_eq!(diag.as_ref().dims(), &[2]);
}

#[test]
fn validate_layout_covers_empty_case_and_error_paths() {
    assert!(validate_layout::<f64>(&[], &[], &scalar(1.0)).is_ok());

    let err = validate_layout::<f64>(&[2], &[0, 1], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let err = validate_layout::<f64>(&[2], &[1], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let err = validate_layout::<f64>(&[2, 3], &[0, 0], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));

    let bad_payload = vector(&[1.0, 2.0, 3.0]);
    let err = validate_layout::<f64>(&[2, 2], &[0, 0], &bad_payload).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn canonicalize_axis_classes_is_first_appearance_stable() {
    assert_eq!(
        canonicalize_axis_classes(&[4, 9, 4, 7, 9]),
        vec![0, 1, 0, 2, 1]
    );
}
