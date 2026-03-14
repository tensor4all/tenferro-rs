use num_complex::Complex64;
use tenferro_dyadtensor::{AdMode, DynAdTensor, DynTape, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor};

fn scalar_f64(value: f64) -> Tensor<f64> {
    Tensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f64(values: &[f64]) -> Tensor<f64> {
    Tensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn dynadtensor_public_primal_constructor_handles_dense_and_diag() {
    let dense = DynAdTensor::new_primal(vector_f64(&[1.0, 2.0]));
    assert_eq!(dense.mode(), AdMode::Primal);
    assert!(dense.is_dense());
    assert_eq!(dense.dims(), &[2]);

    let diag = DynAdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(vector_f64(&[3.0, 4.0]), 2).unwrap(),
    );
    assert_eq!(diag.mode(), AdMode::Primal);
    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
}

#[test]
fn dynadtensor_public_forward_constructor_preserves_tangent() {
    let x = DynAdTensor::new_forward(vector_f64(&[1.0, 2.0]), vector_f64(&[0.5, -0.5])).unwrap();
    assert_eq!(x.mode(), AdMode::Forward);
    assert_eq!(x.dims(), &[2]);
    assert_eq!(
        x.as_f64()
            .unwrap()
            .tangent()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5, -0.5]
    );
}

#[test]
fn dynadtensor_public_reverse_constructor_uses_dyntape() {
    let tape = DynTape::new();
    let x = DynAdTensor::new_reverse_leaf(scalar_f64(2.0), &tape).unwrap();

    assert_eq!(x.mode(), AdMode::Reverse);
    assert_eq!(x.tape_id(), Some(tape.id() as u64));
    assert!(x.node_id().is_some());
}

#[test]
fn dynadtensor_public_rank0_complex_scale_does_not_require_adtensor() {
    let x = DynAdTensor::new_primal(scalar_f64(2.0));
    let alpha = DynAdTensor::new_primal(
        Tensor::from_slice(&[Complex64::new(0.0, 3.0)], &[], MemoryOrder::ColumnMajor).unwrap(),
    );

    let y = x.scale(&alpha).unwrap();
    assert_eq!(y.mode(), AdMode::Primal);
    assert_eq!(y.dims(), &[]);
    assert_eq!(
        y.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(0.0, 6.0)]
    );
}

#[test]
fn dynadtensor_public_to_scalar_type_supports_cross_precision_cast() {
    let x = DynAdTensor::new_primal(scalar_f64(2.0));
    let y = x
        .to_scalar_type(tenferro_dyadtensor::ScalarType::F32)
        .unwrap();
    assert_eq!(y.scalar_type(), tenferro_dyadtensor::ScalarType::F32);
    assert_eq!(
        y.as_f32().unwrap().primal().buffer().as_slice().unwrap(),
        &[2.0]
    );
}
