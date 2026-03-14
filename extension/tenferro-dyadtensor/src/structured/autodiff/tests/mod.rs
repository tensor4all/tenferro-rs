use chainrules_core::Differentiable;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::structured::StructuredTensor;

fn diag(values: &[f64], dim: usize) -> StructuredTensor<f64> {
    StructuredTensor::from_diagonal_vector(
        Tensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap(),
        dim,
    )
    .unwrap()
}

#[test]
fn zero_tangent_preserves_structured_layout() {
    let x = diag(&[1.0, 2.0], 2);
    let tangent = x.zero_tangent();

    assert_eq!(tangent.logical_dims(), x.logical_dims());
    assert_eq!(tangent.axis_classes(), x.axis_classes());
    assert_eq!(tangent.payload().dims(), x.payload().dims());
    assert!(tangent
        .payload()
        .buffer()
        .as_slice()
        .unwrap()
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn seed_cotangent_preserves_structured_layout() {
    let x = diag(&[1.0, 2.0], 2);
    let seed = x.seed_cotangent();

    assert_eq!(seed.logical_dims(), x.logical_dims());
    assert_eq!(seed.axis_classes(), x.axis_classes());
    assert_eq!(seed.payload().dims(), x.payload().dims());
    assert!(seed
        .payload()
        .buffer()
        .as_slice()
        .unwrap()
        .iter()
        .all(|v| *v == 1.0));
}

#[test]
fn accumulate_tangent_and_num_elements_follow_structured_layout() {
    let x = diag(&[1.0, 2.0], 2);
    let a = x.seed_cotangent();
    let b = x.zero_tangent();
    let sum = StructuredTensor::accumulate_tangent(a.clone(), &b);

    assert_eq!(sum.logical_dims(), x.logical_dims());
    assert_eq!(sum.axis_classes(), x.axis_classes());
    assert_eq!(sum.payload().dims(), x.payload().dims());
    assert_eq!(sum.payload().buffer().as_slice().unwrap(), &[1.0, 1.0]);
    assert_eq!(x.num_elements(), 4);
}

#[test]
fn dense_zero_and_seed_tangents_preserve_dense_layout() {
    let dense = StructuredTensor::from_dense(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let zero = dense.zero_tangent();
    let seed = dense.seed_cotangent();

    assert!(zero.is_dense());
    assert!(seed.is_dense());
    assert_eq!(zero.logical_dims(), &[2, 2]);
    assert_eq!(seed.logical_dims(), &[2, 2]);
    assert_eq!(
        zero.payload().buffer().as_slice().unwrap(),
        &[0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        seed.payload().buffer().as_slice().unwrap(),
        &[1.0, 1.0, 1.0, 1.0]
    );
}
