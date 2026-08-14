use tenferro_cpu::CpuBackend;
use tenferro_runtime::prelude::*;
use tenferro_tensor::BackendSessionHost;

#[test]
fn prelude_calls_backend_explicit_tensor_operation() {
    let lhs = Tensor::from_vec_col_major([2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major([2], vec![3.0_f64, 4.0]).unwrap();
    let mut backend = CpuBackend::new();
    let sum = backend
        .with_backend_session(|session| lhs.add(&rhs, session))
        .unwrap();
    assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}
