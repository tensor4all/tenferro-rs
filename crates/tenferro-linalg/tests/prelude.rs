use tenferro_cpu::CpuBackend;
use tenferro_linalg::prelude::*;

#[test]
fn prelude_calls_concrete_linalg_operation() {
    let input = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();
    let (_u, singular_values, _vt) = backend
        .with_backend_session(|session| input.svd(session))
        .unwrap();
    assert_eq!(singular_values.shape(), &[2]);
}

#[cfg(feature = "autodiff")]
#[test]
fn prelude_calls_eager_linalg_operation() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap(),
        runtime,
    )
    .unwrap();
    let (_u, singular_values, _vt) = input.svd().unwrap();
    assert_eq!(singular_values.shape(), &[2]);
}
