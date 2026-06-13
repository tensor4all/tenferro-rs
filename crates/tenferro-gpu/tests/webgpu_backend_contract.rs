#![cfg(feature = "webgpu")]

use tenferro_gpu::{download_webgpu_tensor, upload_webgpu_tensor, WebGpuBackend, WebGpuRuntime};
use tenferro_tensor::{
    DotGeneralConfig, Result, Tensor, TensorBackend, TensorDeviceTransfer, TensorDot,
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
