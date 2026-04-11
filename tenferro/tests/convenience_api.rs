use tenferro::{CpuBackend, Engine, TensorScalar, TracedTensor};

#[test]
fn traced_tensor_new_and_tensor_as_slice_cover_common_f64_flow() {
    let a = TracedTensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = TracedTensor::new(vec![2, 3], vec![6.0_f64, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let mut sum = &a + &b;
    let mut engine = Engine::new(CpuBackend::new());
    let result = sum.eval(&mut engine).unwrap();

    assert_eq!(
        result.as_slice::<f64>(),
        Some([7.0, 7.0, 7.0, 7.0, 7.0, 7.0].as_slice())
    );
    assert_eq!(result.as_slice::<f32>(), None);
}

#[test]
fn tensor_scalar_is_reexported_from_tenferro() {
    let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]);

    assert_eq!(tensor.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
}
