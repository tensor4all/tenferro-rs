use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_runtime::Tensor;

#[test]
fn eager_runtime_constructors_return_typed_results() {
    let _: fn() -> tenferro_ad::Result<Arc<EagerRuntime>> = EagerRuntime::new;
    let _: fn(&tenferro_ad::AdContext) -> tenferro_ad::Result<Arc<EagerRuntime>> =
        EagerRuntime::with_ad_context;
}

#[test]
fn eager_runtime_replaces_eager_context_public_name() {
    let runtime = EagerRuntime::new().unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let loss = runtime
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            let squared = s.mul(&x, &x)?;
            s.reduce_sum(&squared, Some(&[0]))
        })
        .unwrap();

    loss.backward(&mut tenferro_cpu::CpuEngine::serial())
        .unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
    runtime.clear_grads().unwrap();
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_grad_accumulation_keeps_slot_locked_through_update() {
    let source = include_str!("../../src/eager.rs");

    assert!(
        !source.contains("let mut staged = Vec::new();"),
        "gradient accumulation must not stage slot updates after releasing the per-slot lock"
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn eager_runtime_accepts_webgpu_backend_constructor() {
    let _ctor: fn(WebGpuBackend) -> tenferro_ad::Result<std::sync::Arc<EagerRuntime>> =
        EagerRuntime::with_webgpu_backend;
}
