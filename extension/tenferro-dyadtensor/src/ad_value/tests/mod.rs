mod organization;

use super::*;
use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{Error, StructuredTensor};

fn dense_matrix(values: &[f64; 4]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn diag2(values: &[f64; 2]) -> StructuredTensor<f64> {
    StructuredTensor::from_diagonal_vector(
        Tensor::<f64>::from_slice(values, &[2], MemoryOrder::ColumnMajor).unwrap(),
        2,
    )
    .unwrap()
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
    let tensor = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let ad = AdTensor::new_primal(tensor);
    assert_eq!(ad.mode(), AdMode::Primal);
    assert_eq!(ad.dims(), &[2]);
    assert_eq!(ad.ndim(), 1);
    assert_eq!(ad.len(), 2);
}

#[test]
fn ad_tensor_new_forward_rejects_tangent_layout_mismatch() {
    let err =
        AdTensor::new_forward(dense_matrix(&[1.0, 2.0, 3.0, 4.0]), diag2(&[5.0, 6.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn ad_tensor_new_reverse_rejects_tangent_layout_mismatch() {
    let err = AdTensor::new_reverse(
        dense_matrix(&[1.0, 2.0, 3.0, 4.0]),
        NodeId(1),
        TapeId(7),
        Some(diag2(&[5.0, 6.0])),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn ad_tensor_try_from_structured_value_rejects_tangent_layout_mismatch() {
    let value = AdValue::Forward {
        primal: StructuredTensor::from_dense(dense_matrix(&[1.0, 2.0, 3.0, 4.0])),
        tangent: diag2(&[5.0, 6.0]),
    };
    let err = AdTensor::try_from(value).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn ad_scalar_into_primal_drops_metadata() {
    let x = AdScalar::new_forward(2.5_f64, 0.1_f64);
    assert_eq!(x.into_primal(), 2.5_f64);
}

#[test]
fn ad_scalar_sqrt_forward_propagates_tangent() {
    let x = AdScalar::new_forward(9.0_f64, 1.0_f64);
    let y = x.sqrt();
    assert!((*y.primal() - 3.0).abs() < 1e-12);
    assert!((*y.tangent().unwrap() - (1.0 / 6.0)).abs() < 1e-12);
}

#[test]
fn ad_scalar_powi_forward_propagates_tangent() {
    let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    let y = x.powi(4);
    assert_eq!(*y.primal(), 16.0);
    assert_eq!(*y.tangent().unwrap(), 32.0);
}

#[test]
fn ad_scalar_mul_forward_propagates_tangent() {
    let x = AdScalar::new_forward(2.0_f64, 0.5_f64);
    let y = AdScalar::new_forward(4.0_f64, 0.25_f64);
    let z = (x * y).unwrap();
    assert_eq!(*z.primal(), 8.0_f64);
    assert_eq!(*z.tangent().unwrap(), 2.5_f64);
}

#[test]
fn ad_scalar_div_forward_propagates_tangent() {
    let x = AdScalar::new_forward(8.0_f64, 0.5_f64);
    let y = AdScalar::new_forward(2.0_f64, 0.25_f64);
    let z = (x / y).unwrap();
    assert_eq!(*z.primal(), 4.0_f64);
    assert_eq!(*z.tangent().unwrap(), -0.25_f64);
}

#[test]
fn ad_scalar_conj_reverse_allocates_fresh_output_node() {
    let x = AdScalar::new_reverse(
        Complex64::new(1.0, 2.0),
        NodeId(11),
        TapeId(7),
        Some(Complex64::new(-1.0, 0.5)),
    );
    let y = x.conj();
    assert_eq!(y.mode(), AdMode::Reverse);
    assert_eq!(y.as_value().tape_id(), Some(TapeId(7)));
    assert_ne!(y.as_value().node_id(), Some(NodeId(11)));
    assert_eq!(*y.primal(), Complex64::new(1.0, -2.0));
    assert_eq!(*y.tangent().unwrap(), Complex64::new(-1.0, -0.5));
}

#[test]
fn ad_scalar_sqrt_reverse_registers_pullback_chain() {
    let x = AdScalar::new_reverse(4.0_f64, NodeId(21), TapeId(17), None);
    let y = x.sqrt();
    let grads = crate::reverse_tape::pullback_scalar::<f64>(
        TapeId(17),
        y.as_value().node_id().unwrap(),
        &3.0_f64,
    )
    .unwrap();
    assert_eq!(grads.get(&NodeId(21)).copied(), Some(0.75));
}

#[test]
fn ad_scalar_binary_op_returns_error_on_mixed_reverse_tapes() {
    let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(7), None);
    let y = AdScalar::new_reverse(3.0_f64, NodeId(2), TapeId(8), None);
    let err = (x * y).unwrap_err();
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: 7,
            found: 8
        }
    ));
}

#[test]
fn ad_scalar_try_binary_op_returns_error_on_mixed_reverse_tapes() {
    let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(7), None);
    let y = AdScalar::new_reverse(3.0_f64, NodeId(2), TapeId(8), None);
    let err = x.try_mul(y).unwrap_err();
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: 7,
            found: 8
        }
    ));
}
