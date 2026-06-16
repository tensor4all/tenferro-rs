use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::WebGpuBackend;
use tenferro_runtime::Tensor;

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        runtime.clone(),
    );
    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();

    loss.backward().unwrap();

    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    runtime.clear_grads();
    assert!(x.grad().is_none());
}

#[test]
fn eager_runtime_synchronize_is_available_and_cpu_noop() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());

    runtime.synchronize().unwrap();
}

#[cfg(feature = "webgpu")]
#[test]
fn eager_runtime_accepts_webgpu_backend_constructor() {
    let _ctor: fn(WebGpuBackend) -> std::sync::Arc<EagerRuntime> =
        EagerRuntime::with_webgpu_backend;
}
