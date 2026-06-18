use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::WebGpuBackend;
use tenferro_runtime::Tensor;

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        runtime.clone(),
    );
    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();

    loss.backward().unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
    runtime.clear_grads().unwrap();
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_synchronize_is_available_and_cpu_noop() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());

    runtime.synchronize().unwrap();
}

#[test]
fn eager_runtime_grad_accumulation_keeps_slot_locked_through_update() {
    let source = include_str!("../src/eager.rs");

    assert!(
        !source.contains("let mut staged = Vec::new();"),
        "gradient accumulation must not stage slot updates after releasing the per-slot lock"
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn eager_runtime_accepts_webgpu_backend_constructor() {
    let _ctor: fn(WebGpuBackend) -> std::sync::Arc<EagerRuntime> =
        EagerRuntime::with_webgpu_backend;
}
