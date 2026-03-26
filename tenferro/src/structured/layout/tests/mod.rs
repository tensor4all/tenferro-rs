use super::StructuredTensor;
use tenferro_device::Error as DeviceError;
use tenferro_tensor::{
    structured_tensor::{canonicalize_axis_classes, validate_layout},
    MemoryOrder, StructuredTensor as InnerStructuredTensor, Tensor as DenseTensor,
};

fn scalar(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix(values: &[f64], rows: usize, cols: usize) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[rows, cols], MemoryOrder::ColumnMajor).unwrap()
}

fn dense_layout(payload: DenseTensor<f64>) -> StructuredTensor<f64> {
    StructuredTensor(InnerStructuredTensor::from_dense(payload))
}

fn diag_layout(
    payload: DenseTensor<f64>,
    logical_rank: usize,
) -> Result<StructuredTensor<f64>, DeviceError> {
    InnerStructuredTensor::from_diagonal_vector(payload, logical_rank).map(StructuredTensor)
}

#[test]
fn from_diagonal_vector_rejects_invalid_payload_rank_and_zero_logical_rank() {
    let err = diag_layout(matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2), 2).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));

    let err = diag_layout(vector(&[1.0, 2.0]), 0).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));
}

#[test]
fn scalar_layout_is_not_diag_and_as_ref_exposes_payload() {
    let scalar_layout = dense_layout(scalar(3.0));
    assert!(!scalar_layout.is_diag());

    let diag = diag_layout(vector(&[1.0, 2.0]), 2).unwrap();
    assert_eq!(diag.as_ref().dims(), &[2]);
}

#[test]
fn validate_layout_covers_empty_case_and_error_paths() {
    assert!(validate_layout::<f64>(&[], &[], &scalar(1.0)).is_ok());

    let err = validate_layout::<f64>(&[2], &[0, 1], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));

    let err = validate_layout::<f64>(&[2], &[1], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));

    let err = validate_layout::<f64>(&[2, 3], &[0, 0], &vector(&[1.0, 2.0])).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));

    let bad_payload = vector(&[1.0, 2.0, 3.0]);
    let err = validate_layout::<f64>(&[2, 2], &[0, 0], &bad_payload).unwrap_err();
    assert!(matches!(err, DeviceError::InvalidArgument(_)));
}

#[test]
fn canonicalize_axis_classes_is_first_appearance_stable() {
    assert_eq!(
        canonicalize_axis_classes(&[4, 9, 4, 7, 9]),
        vec![0, 1, 0, 2, 1]
    );
}

#[test]
fn wrapper_newtype_exposes_inner_layout_methods() {
    let wrapped = StructuredTensor(
        InnerStructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );

    assert!(wrapped.is_diag());
    assert_eq!(wrapped.logical_dims(), &[2, 2]);
    assert_eq!(wrapped.as_ref().dims(), &[2]);
}
