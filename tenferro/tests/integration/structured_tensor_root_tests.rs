use num_complex::{Complex32, Complex64};
use tenferro::{ScalarType, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn tensor2(values: &[f64], d0: usize, d1: usize) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[d0, d1], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn frontend_structured_constructors_cover_dense_diag_and_multi_class_layouts() {
    let dense = Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    assert_eq!(dense.dims(), &[2, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert!(dense.is_dense());

    let diag = Tensor::diag(&Tensor::from_tensor(vector(&[1.0, 2.0]))).unwrap();
    assert_eq!(diag.dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert!(diag.is_diag());

    let structured = Tensor::with_axis_classes(
        Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2)),
        &[0, 1, 1],
    )
    .unwrap();
    assert_eq!(structured.dims(), &[2, 2, 2]);
    assert_eq!(structured.axis_classes(), &[0, 1, 1]);
    assert!(!structured.is_dense());
    assert!(!structured.is_diag());
}

#[test]
fn tensor_wraps_structured_payloads_without_dense_materialization() {
    let diag = Tensor::diag(&Tensor::from_tensor(vector(&[1.0, 2.0]))).unwrap();
    assert_eq!(diag.dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert_eq!(diag.as_f64().unwrap().primal().dims(), &[2]);

    let structured = Tensor::with_axis_classes(
        Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2)),
        &[0, 1, 1],
    )
    .unwrap();
    assert_eq!(structured.as_f64().unwrap().primal().dims(), &[2, 2]);
}

#[test]
fn detach_preserves_dense_and_structured_payloads() {
    let dense = Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    let diag = Tensor::diag(&Tensor::from_tensor(vector(&[1.0, 2.0]))).unwrap();

    let dense_detached = dense.detach();
    assert!(dense_detached.is_dense());
    assert_eq!(dense_detached.dims(), &[2, 2]);

    let diag_detached = diag.detach();
    assert!(diag_detached.is_diag());
    assert_eq!(diag_detached.axis_classes(), &[0, 0]);
    assert_eq!(diag_detached.as_f64().unwrap().primal().dims(), &[2]);
}

#[test]
fn with_axis_classes_requires_canonical_axis_class_order() {
    let err = match Tensor::with_axis_classes(
        Tensor::from_tensor(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2)),
        &[1, 0, 0],
    ) {
        Ok(_) => panic!("expected non-canonical axis classes to be rejected"),
        Err(err) => err,
    };

    let message = match err {
        tenferro::Error::InvalidAdTensor { message } => message,
        other => panic!("expected InvalidAdTensor, got {other:?}"),
    };
    assert!(message.contains("canonical axis classes"));
}

#[test]
fn detach_covers_non_f64_runtime_variants() {
    let f32_value = Tensor::from_tensor(
        DenseTensor::<f32>::from_slice(&[1.0_f32], &[1], MemoryOrder::ColumnMajor).unwrap(),
    );
    let c32_value = Tensor::from_tensor(
        DenseTensor::<Complex32>::from_slice(
            &[Complex32::new(1.0, -2.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let c64_value = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(2.0, 3.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let f32_detached = f32_value.detach();
    let c32_detached = c32_value.detach();
    let c64_detached = c64_value.detach();

    assert_eq!(f32_detached.scalar_type(), ScalarType::F32);
    assert_eq!(c32_detached.scalar_type(), ScalarType::C32);
    assert_eq!(c64_detached.scalar_type(), ScalarType::C64);
}
