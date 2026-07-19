#![cfg(feature = "webgpu")]

use tenferro_gpu::{
    download_webgpu_tensor, upload_webgpu_tensor, webgpu_interop, WebGpuBackend, WebGpuRuntime,
};
use tenferro_tensor::{
    DotGeneralConfig, Error, Result, Tensor, TensorBackend, TensorDeviceTransfer, TensorDot,
};

fn assert_tensor_backend<B: TensorBackend>() {}

#[test]
fn webgpu_backend_implements_tensor_backend_contract() {
    assert_tensor_backend::<WebGpuBackend>();

    let _upload: fn(&mut WebGpuBackend, &Tensor) -> Result<Tensor> =
        <WebGpuBackend as TensorDeviceTransfer>::upload_host_tensor;
    let _download: fn(&mut WebGpuBackend, &Tensor) -> Result<Tensor> =
        <WebGpuBackend as TensorDeviceTransfer>::download_to_host;
    let _dot: fn(&mut WebGpuBackend, &Tensor, &Tensor, &DotGeneralConfig) -> Result<Tensor> =
        <WebGpuBackend as TensorDot>::dot_general;
}

#[test]
fn webgpu_transfer_helpers_are_provider_specific() {
    let _upload: fn(&WebGpuRuntime, &Tensor) -> Result<Tensor> = upload_webgpu_tensor;
    let _download: fn(&WebGpuRuntime, &Tensor) -> Result<Tensor> = download_webgpu_tensor;
}

#[test]
fn webgpu_download_checks_runtime_residency_before_reading_backend_handle() {
    let source = include_str!("../../src/webgpu/memory.rs");

    let residency_check = source
        .find("ensure_resident_on_runtime(rt, typed, \"webgpu_download\")?;")
        .expect("download helpers must validate runtime residency");
    let backend_read = source
        .find(".read_one(handle)")
        .expect("download helpers should read through the WebGPU client");

    assert!(
        residency_check < backend_read,
        "WebGPU download must reject non-resident buffers before reading from a runtime handle"
    );
}

#[test]
fn webgpu_output_completion_rejects_undersized_f32_and_c32_ranges() {
    let Ok(runtime) = WebGpuRuntime::new_default() else {
        return;
    };
    let backend = WebGpuBackend::from_runtime(runtime);
    let f32_handle = webgpu_interop::allocate_raw(&backend, 4);
    let error =
        webgpu_interop::finish_f32(&backend, vec![2], f32_handle, "test_finish_f32").unwrap_err();
    assert!(matches!(error, Error::RuntimeState { .. }));

    let c32_handle = webgpu_interop::allocate_raw(&backend, 4);
    let error =
        webgpu_interop::finish_c32(&backend, vec![1], c32_handle, "test_finish_c32").unwrap_err();
    assert!(matches!(error, Error::RuntimeState { .. }));
}

#[test]
fn webgpu_output_completion_rejects_surviving_raw_alias() {
    let Ok(runtime) = WebGpuRuntime::new_default() else {
        return;
    };
    let backend = WebGpuBackend::from_runtime(runtime);
    let handle = webgpu_interop::allocate_raw(&backend, 8);
    let alias = handle.clone();
    let error =
        webgpu_interop::finish_f32(&backend, vec![2], handle, "test_finish_alias").unwrap_err();
    assert!(matches!(error, Error::RuntimeState { .. }));
    drop(alias);
}
