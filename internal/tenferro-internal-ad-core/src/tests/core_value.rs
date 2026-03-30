use tenferro_internal_error::Error;
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::expert::Tape;

use crate::{AdMode, AdTensor, AdTensorSnapshot, AdValue};

fn dense_matrix(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn diag2(values: &[f64; 2]) -> StructuredTensor<f64> {
    StructuredTensor::from(
        tenferro_tensor::StructuredTensor::from_diagonal_vector(
            DenseTensor::<f64>::from_slice(values, &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        )
        .unwrap(),
    )
}

fn dense_layout(values: &[f64; 4]) -> StructuredTensor<f64> {
    StructuredTensor::from(dense_matrix(values))
}

#[test]
fn ad_value_map_preserving_metadata_preserves_mode() {
    let x = AdValue::forward(2_i32, 3_i32);
    let y = x.map_preserving_metadata(|v| v as f64);
    assert_eq!(y.mode(), AdMode::Forward);
    assert_eq!(y.primal_ref(), &2.0_f64);
    assert_eq!(y.tangent_ref(), Some(&3.0_f64));
}

#[test]
fn ad_tensor_metadata() {
    let tensor =
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let ad = AdTensor::new_primal(tensor);
    assert_eq!(ad.mode(), AdMode::Primal);
    assert_eq!(ad.dims(), &[2]);
    assert_eq!(ad.ndim(), 1);
    assert_eq!(ad.len(), 2);
}

#[test]
fn ad_tensor_new_forward_rejects_tangent_layout_mismatch() {
    let err = match AdTensor::new_forward(dense_matrix(&[1.0, 2.0, 3.0, 4.0]), diag2(&[5.0, 6.0])) {
        Ok(_) => panic!("expected tangent layout mismatch"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn ad_tensor_new_reverse_rejects_tangent_layout_mismatch() {
    let tape = Tape::<DynTensor>::new();
    let err = match AdTensor::new_reverse_leaf_with_tangent(
        dense_matrix(&[1.0, 2.0, 3.0, 4.0]),
        diag2(&[5.0, 6.0]),
        &tape,
    ) {
        Ok(_) => panic!("expected reverse tangent layout mismatch"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn ad_tensor_try_from_structured_value_rejects_tangent_layout_mismatch() {
    let value = AdTensorSnapshot::Forward {
        primal: dense_layout(&[1.0, 2.0, 3.0, 4.0]),
        tangent: diag2(&[5.0, 6.0]),
    };
    let err = match AdTensor::try_from(value) {
        Ok(_) => panic!("expected structured tangent layout mismatch"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}
