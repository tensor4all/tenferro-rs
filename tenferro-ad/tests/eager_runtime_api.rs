use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::Tensor;

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        runtime.clone(),
    );
    let loss = (&x * &x).reduce_sum(&[0]).unwrap();

    loss.backward().unwrap();

    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    runtime.clear_grads();
    assert!(x.grad().is_none());
}
