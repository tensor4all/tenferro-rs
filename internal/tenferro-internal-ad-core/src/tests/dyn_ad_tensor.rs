use num_complex::Complex64;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};
use tenferro_internal_frontend_core::{DynTensor, ScalarType, ScalarValue};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::expert::Tape;

use crate::{AdMode, AdTensor, DynAdTensor};

fn dense_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn dense_c64(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix_f64(values: &[f64], dims: &[usize]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn dyn_ad_tensor_reports_scalar_type_and_mode() {
    let primal = DynAdTensor::from(AdTensor::new_primal(dense_f64(&[1.0, 2.0])));
    assert_eq!(primal.scalar_type(), ScalarType::F64);
    assert_eq!(primal.mode(), AdMode::Primal);
    assert_eq!(primal.dims(), &[2]);

    let forward = DynAdTensor::from(
        AdTensor::new_forward(
            dense_c64(&[Complex64::new(1.0, 0.0)]),
            dense_c64(&[Complex64::new(0.5, 0.0)]),
        )
        .unwrap(),
    );
    assert_eq!(forward.scalar_type(), ScalarType::C64);
    assert_eq!(forward.mode(), AdMode::Forward);
    assert_eq!(forward.dims(), &[1]);
}

#[test]
fn dyn_ad_tensor_owned_constructors_preserve_scalar_type_and_mode() {
    let primal = DynAdTensor::new_primal(dense_f64(&[1.0, 2.0]));
    assert_eq!(primal.scalar_type(), ScalarType::F64);
    assert_eq!(primal.mode(), AdMode::Primal);
    assert_eq!(primal.dims(), &[2]);

    let forward = DynAdTensor::new_forward(
        dense_c64(&[Complex64::new(1.0, 0.0)]),
        dense_c64(&[Complex64::new(0.5, 0.0)]),
    )
    .unwrap();
    assert_eq!(forward.scalar_type(), ScalarType::C64);
    assert_eq!(forward.mode(), AdMode::Forward);
    assert_eq!(forward.dims(), &[1]);
}

#[test]
fn dyn_ad_tensor_owned_reverse_constructors_preserve_tape_and_tangent() {
    let tape = Tape::<DynTensor>::new();
    let reverse = DynAdTensor::new_reverse_leaf(dense_f64(&[5.0, 6.0]), &tape).unwrap();
    assert_eq!(reverse.mode(), AdMode::Reverse);
    assert!(reverse.tape().unwrap().same_tape(&tape));

    let tangent_tape = Tape::<DynTensor>::new();
    let reverse_with_tangent = DynAdTensor::new_reverse_leaf_with_tangent(
        dense_f64(&[7.0, 8.0]),
        dense_f64(&[0.1, 0.2]),
        &tangent_tape,
    )
    .unwrap();
    assert_eq!(reverse_with_tangent.mode(), AdMode::Reverse);
    assert!(reverse_with_tangent
        .tape()
        .unwrap()
        .same_tape(&tangent_tape));
    assert!(reverse_with_tangent.has_tangent());
}

#[test]
fn dyn_ad_tensor_typed_accessors_match_variant() {
    let value = DynAdTensor::from(AdTensor::new_primal(dense_f64(&[3.0, 4.0])));

    assert!(value.as_f64().is_some());
    assert!(value.as_c64().is_none());
}

#[test]
fn dyn_ad_tensor_preserves_reverse_metadata() {
    let tape = Tape::<DynTensor>::new();
    let value =
        DynAdTensor::from(AdTensor::new_reverse_leaf(dense_f64(&[5.0, 6.0]), &tape).unwrap());

    assert_eq!(value.mode(), AdMode::Reverse);
    assert!(value.node_id().is_some());
    assert!(value
        .tape()
        .expect("reverse dyn tensor should expose its tape")
        .same_tape(&tape));
}

#[test]
fn dyn_ad_tensor_ref_exposes_reverse_read_only_helpers() {
    let tape = Tape::<DynTensor>::new();
    let value =
        DynAdTensor::from(AdTensor::new_reverse_leaf(dense_f64(&[7.0, 8.0]), &tape).unwrap());
    let value_ref = crate::DynAdTensorRef::from(&value);

    assert_eq!(value_ref.scalar_type(), ScalarType::F64);
    assert!(value_ref.reverse_handle().is_some());
    assert!(value_ref.as_tracked().is_some());
}

#[test]
fn dyn_ad_tensor_ref_delegates_shape_and_placement_helpers() {
    let mut value = DynAdTensor::from(AdTensor::new_primal(matrix_f64(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
    )));
    value.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));

    let value_ref = crate::DynAdTensorRef::from(&value);
    assert_eq!(value_ref.axis_classes(), &[0, 1]);
    assert!(value_ref.is_dense());
    assert!(!value_ref.is_diag());
    assert_eq!(value_ref.memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(
        value_ref.preferred_compute_device(),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
    assert!(value_ref.is_ready());
    value_ref.wait();

    let moved = value_ref
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(moved.scalar_type(), ScalarType::F64);
    assert_eq!(moved.memory_space(), LogicalMemorySpace::MainMemory);
}

#[test]
fn dyn_ad_tensor_ref_extracts_rank0_scalar_value() {
    let value = DynAdTensor::from(AdTensor::new_primal(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(2.0, -3.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    ));

    let value_ref = crate::DynAdTensorRef::from(&value);
    assert_eq!(
        value_ref.try_scalar_value().unwrap(),
        ScalarValue::C64(Complex64::new(2.0, -3.0))
    );
}

#[test]
fn dyn_ad_tensor_ref_exposes_typed_erased_borrows_for_matching_dtype() {
    let value = DynAdTensor::from(
        AdTensor::new_forward(
            matrix_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            matrix_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2]),
        )
        .unwrap(),
    );
    let value_ref = crate::DynAdTensorRef::from(&value);

    assert_eq!(value_ref.primal_as::<f64>().unwrap().dims(), &[2, 2]);
    assert_eq!(
        value_ref.primal_as::<f64>().unwrap().to_vec(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        value_ref.tangent_as::<f64>().unwrap().to_vec(),
        vec![10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        value_ref
            .structured_primal_as::<f64>()
            .unwrap()
            .logical_dims(),
        &[2, 2]
    );
    assert_eq!(
        value_ref
            .structured_tangent_as::<f64>()
            .unwrap()
            .logical_dims(),
        &[2, 2]
    );
}

#[test]
fn dyn_ad_tensor_ref_typed_erased_borrows_reject_mismatched_dtype() {
    let value = DynAdTensor::from(AdTensor::new_primal(dense_f64(&[1.0, 2.0])));
    let value_ref = crate::DynAdTensorRef::from(&value);

    assert!(value_ref.primal_as::<f32>().is_none());
    assert!(value_ref.structured_primal_as::<f32>().is_none());
    assert!(value_ref.tangent_as::<f32>().is_none());
    assert!(value_ref.structured_tangent_as::<f32>().is_none());
}

#[test]
fn dyn_ad_tensor_ref_typed_erased_borrows_preserve_absent_tangent() {
    let value = DynAdTensor::from(AdTensor::new_primal(dense_c64(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ])));
    let value_ref = crate::DynAdTensorRef::from(&value);

    assert!(value_ref.primal_as::<Complex64>().is_some());
    assert!(value_ref.tangent_as::<Complex64>().is_none());
    assert!(value_ref.structured_tangent_as::<Complex64>().is_none());
}

#[test]
fn dyn_ad_tensor_mut_ref_updates_preferred_compute_device() {
    let mut value = DynAdTensor::from(AdTensor::new_primal(dense_f64(&[1.0, 2.0])));

    let value_mut = crate::DynAdTensorMutRef::from(&mut value);
    value_mut.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 1 }));

    assert_eq!(
        value.preferred_compute_device(),
        Some(ComputeDevice::Cpu { device_id: 1 })
    );
}
