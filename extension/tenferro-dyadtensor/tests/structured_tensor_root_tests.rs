use tenferro_dyadtensor::{AdTensor, DynAdTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor};

fn vector(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn tensor2(values: &[f64], d0: usize, d1: usize) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[d0, d1], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn root_structured_tensor_supports_dense_and_diag_layouts() {
    let dense = StructuredTensor::from_dense(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    assert_eq!(dense.logical_dims(), &[2, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert!(dense.is_dense());

    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    assert_eq!(diag.logical_dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert!(diag.is_diag());
}

#[test]
fn ad_tensor_wraps_structured_payload_and_reports_logical_dims() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x = AdTensor::new_primal(diag);
    assert_eq!(x.dims(), &[2, 2]);
    assert!(x.structured_primal().is_diag());
    assert_eq!(x.primal().dims(), &[2]);
}

#[test]
fn dyn_ad_tensor_carries_diag_payload_without_dense_materialization() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x: DynAdTensor = AdTensor::new_primal(diag).into();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(x.axis_classes(), &[0, 0]);
    assert!(x.is_diag());
    assert_eq!(x.as_f64().unwrap().primal().dims(), &[2]);
}
