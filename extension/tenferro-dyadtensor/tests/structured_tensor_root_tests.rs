use num_complex::{Complex32, Complex64};
use tenferro_dyadtensor::{
    plan_axis_classes_for_subscripts, AdMode, DynAdTensor, OperandAxisClasses, StructuredTensor,
};
use tenferro_einsum::Subscripts;
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
fn dynadtensor_wraps_structured_payload_and_reports_logical_dims() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x = DynAdTensor::new_primal(diag);
    assert_eq!(x.dims(), &[2, 2]);
    assert!(x.is_diag());
    assert_eq!(x.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn dyn_ad_tensor_carries_diag_payload_without_dense_materialization() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x = DynAdTensor::new_primal(diag);
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(x.axis_classes(), &[0, 0]);
    assert!(x.is_diag());
    assert_eq!(x.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn detach_preserves_dense_and_structured_payloads() {
    let dense = DynAdTensor::new_primal(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    let diag = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap(),
    );

    let dense_detached = dense.detach();
    assert_eq!(dense_detached.mode(), AdMode::Primal);
    assert!(dense_detached.is_dense());
    assert_eq!(dense_detached.dims(), &[2, 2]);

    let diag_detached = diag.detach();
    assert_eq!(diag_detached.mode(), AdMode::Primal);
    assert!(diag_detached.is_diag());
    assert_eq!(diag_detached.axis_classes(), &[0, 0]);
    assert_eq!(diag_detached.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn detach_preserves_general_axis_classes() {
    let structured = StructuredTensor::new(
        vec![2, 2, 2],
        vec![0, 1, 1],
        tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2),
    )
    .unwrap();
    let x = DynAdTensor::new_primal(structured);

    let detached = x.detach();
    assert_eq!(detached.mode(), AdMode::Primal);
    assert_eq!(detached.dims(), &[2, 2, 2]);
    assert_eq!(detached.axis_classes(), &[0, 1, 1]);
    assert_eq!(detached.as_f64().unwrap().primal().dims(), &[2, 2]);
}

#[test]
fn detach_covers_all_runtime_variants() {
    let f32_value = DynAdTensor::new_primal(
        Tensor::<f32>::from_slice(&[1.0_f32], &[1], MemoryOrder::ColumnMajor).unwrap(),
    );
    let c32_value = DynAdTensor::new_primal(
        Tensor::<Complex32>::from_slice(
            &[Complex32::new(1.0, -2.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let c64_value = DynAdTensor::new_primal(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(2.0, 3.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let f32_detached = f32_value.detach();
    let c32_detached = c32_value.detach();
    let c64_detached = c64_value.detach();

    assert_eq!(f32_detached.mode(), AdMode::Primal);
    assert_eq!(c32_detached.mode(), AdMode::Primal);
    assert_eq!(c64_detached.mode(), AdMode::Primal);
    assert_eq!(
        f32_detached.scalar_type(),
        tenferro_dyadtensor::ScalarType::F32
    );
    assert_eq!(
        c32_detached.scalar_type(),
        tenferro_dyadtensor::ScalarType::C32
    );
    assert_eq!(
        c64_detached.scalar_type(),
        tenferro_dyadtensor::ScalarType::C64
    );
}

#[test]
fn root_metadata_planning_api_is_exposed_from_crate_root() {
    let operands = vec![
        OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
        OperandAxisClasses::new(vec![3, 3], vec![0, 0]).unwrap(),
    ];
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let plan = plan_axis_classes_for_subscripts(&operands, &subs).unwrap();
    assert_eq!(plan.output_axis_classes, vec![0, 0]);
    assert_eq!(plan.output_dims, vec![3, 3]);
}
