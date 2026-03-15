use chainrules::Tape;
mod organization;
mod reverse_api;

use super::*;
use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::{core::AdTensorSnapshot, Error, StructuredTensor};

fn dense_matrix(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn diag2(values: &[f64; 2]) -> StructuredTensor<f64> {
    StructuredTensor::from_diagonal_vector(
        DenseTensor::<f64>::from_slice(values, &[2], MemoryOrder::ColumnMajor).unwrap(),
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
    let tape = Tape::<crate::DynTensor>::new();
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
        primal: StructuredTensor::from_dense(dense_matrix(&[1.0, 2.0, 3.0, 4.0])),
        tangent: diag2(&[5.0, 6.0]),
    };
    let err = match AdTensor::try_from(value) {
        Ok(_) => panic!("expected structured tangent layout mismatch"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn rank0_reverse_tensor_scale_allocates_fresh_output_node() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf_with_tangent(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(-1.0, 0.5)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    )
    .unwrap();
    let alpha: crate::Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(0.0, 1.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    )
    .unwrap()
    .into();
    let y = crate::Tensor::from(x.clone()).scale(&alpha).unwrap();
    let y = y.as_c64().unwrap();
    assert_eq!(y.mode(), AdMode::Reverse);
    assert!(y
        .tape()
        .expect("reverse output should expose a tape")
        .same_tape(&tape));
    assert_ne!(y.node_id(), x.node_id());
    assert_eq!(
        y.primal().buffer().as_slice().unwrap()[0],
        Complex64::new(-2.0, 1.0)
    );
}

#[test]
fn rank0_reverse_tensor_sqrt_registers_pullback_chain() {
    let _guard = crate::set_default_runtime(crate::RuntimeContext::Cpu(
        tenferro_prims::CpuContext::new(1),
    ));
    let tape = Tape::<crate::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[4.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape,
    )
    .unwrap();
    let y = crate::ops::ad::sqrt(&x).unwrap();
    let cotangent =
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap();
    let grads = crate::tape::pullback(&y, &cotangent).unwrap();
    assert_eq!(
        grads
            .get(&x.node_id().unwrap())
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.75]
    );
}

#[test]
fn rank0_reverse_tensor_scale_returns_error_on_mixed_reverse_tapes() {
    let tape_a = Tape::<crate::DynTensor>::new();
    let tape_b = Tape::<crate::DynTensor>::new();
    let x: crate::Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_a,
    )
    .unwrap()
    .into();
    let y: crate::Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_b,
    )
    .unwrap()
    .into();
    let err = match x.scale(&y) {
        Ok(_) => panic!("mixed reverse tapes should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}

#[test]
fn rank0_reverse_tensor_div_scalar_returns_error_on_mixed_reverse_tapes() {
    let tape_a = Tape::<crate::DynTensor>::new();
    let tape_b = Tape::<crate::DynTensor>::new();
    let x: crate::Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_a,
    )
    .unwrap()
    .into();
    let y: crate::Tensor = AdTensor::new_reverse_leaf(
        DenseTensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
        &tape_b,
    )
    .unwrap()
    .into();
    let err = match x.div_scalar(&y) {
        Ok(_) => panic!("mixed reverse tapes should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}
