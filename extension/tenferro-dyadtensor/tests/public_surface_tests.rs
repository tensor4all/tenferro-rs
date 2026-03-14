use num_complex::Complex64;
use tenferro_dyadtensor::{
    set_default_runtime, AdMode, DynAdTensor, DynTape, RuntimeContext, StructuredTensor,
};
use tenferro_prims::CpuContext;
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

#[test]
fn dynadtensor_public_scalar_eager_methods_do_not_require_typed_api() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = DynAdTensor::new_primal(vector_f64(&[0.0, 1.0]));
    let y = x.exp().unwrap();
    assert_eq!(y.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
    let y_vals = y.as_f64().unwrap().primal().buffer().as_slice().unwrap();
    assert!((y_vals[0] - 1.0).abs() < 1e-12);
    assert!((y_vals[1] - std::f64::consts::E).abs() < 1e-12);

    let a = DynAdTensor::new_primal(scalar_f64(2.0));
    let b = DynAdTensor::new_primal(scalar_f64(3.0));
    let c = a.add(&b).unwrap();
    assert_eq!(
        c.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[5.0]
    );

    let m = x.mean().unwrap();
    assert_eq!(m.dims(), &[]);
    assert_eq!(
        m.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[0.5]
    );
}

#[test]
fn dynadtensor_public_pullback_wrt_does_not_require_typed_api() {
    let tape = DynTape::new();
    let x = DynAdTensor::new_reverse_leaf(vector_f64(&[1.0, 2.0]), &tape).unwrap();
    let a = DynAdTensor::new_reverse_leaf(scalar_f64(3.0), &tape).unwrap();
    let out = x.scale(&a).unwrap();
    let cotangent = DynAdTensor::new_primal(vector_f64(&[0.5, 1.25]));

    let grads = out.pullback_wrt(&cotangent, &[&x, &a]).unwrap();
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.5, 3.75]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0]
    );
}
