#![cfg(feature = "webgpu")]

use tenferro_gpu::{webgpu::upload_webgpu_tensor, webgpu::WebGpuBackend};
use tenferro_tensor::{AllocationId, DType, Tensor, TensorRead, TensorStructural};

fn allocation_id(tensor: &Tensor) -> Option<AllocationId> {
    match tensor {
        Tensor::F32(tensor) => tensor.allocation_id(),
        Tensor::F64(tensor) => tensor.allocation_id(),
        Tensor::I32(tensor) => tensor.allocation_id(),
        Tensor::I64(tensor) => tensor.allocation_id(),
        Tensor::Bool(tensor) => tensor.allocation_id(),
        Tensor::C32(tensor) => tensor.allocation_id(),
        Tensor::C64(tensor) => tensor.allocation_id(),
    }
}

fn webgpu_backend() -> Option<WebGpuBackend> {
    match WebGpuBackend::new_default() {
        Ok(backend) => Some(backend),
        Err(error) => {
            eprintln!("skipping WebGPU provider contract test: {error}");
            None
        }
    }
}

#[test]
fn uploaded_storage_is_root_owned_and_prepares_once_at_the_descriptor_boundary() {
    let Some(mut backend) = webgpu_backend() else {
        return;
    };
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let tensor = upload_webgpu_tensor(backend.runtime(), &host).unwrap();
    let Tensor::F32(typed) = &tensor else {
        panic!("provider contract uses f32")
    };

    assert!(typed.allocation_domain().is_some());
    assert!(typed.allocation_id().is_some());
    typed
        .prepare_device_read("storage_provider_webgpu")
        .expect("root preparation should accept the checked descriptor");

    let duplicate = backend
        .to_contiguous_read(TensorRead::from_tensor(&tensor))
        .unwrap();
    assert_eq!(duplicate.dtype(), DType::F32);
    assert_ne!(allocation_id(&duplicate), allocation_id(&tensor));
}

#[test]
fn device_local_host_mapping_is_rejected_without_an_implicit_download() {
    let Some(backend) = webgpu_backend() else {
        return;
    };
    let host = Tensor::from_vec_col_major(vec![1], vec![3.0_f32]).unwrap();
    let tensor = upload_webgpu_tensor(backend.runtime(), &host).unwrap();
    let Tensor::F32(typed) = &tensor else {
        panic!("provider contract uses f32")
    };

    let error = typed.with_host_read(|_| ()).unwrap_err();
    assert!(error.to_string().contains("unsupported") || error.to_string().contains("host"));
}

#[test]
fn empty_upload_keeps_a_zero_logical_root_span() {
    let Some(backend) = webgpu_backend() else {
        return;
    };
    let host = Tensor::from_vec_col_major(vec![0], Vec::<f32>::new()).unwrap();
    let tensor = upload_webgpu_tensor(backend.runtime(), &host).unwrap();
    let Tensor::F32(typed) = tensor else {
        panic!("provider contract uses f32")
    };
    assert_eq!(typed.n_elements(), 0);
    typed
        .prepare_device_read("storage_provider_webgpu_empty")
        .expect("empty roots remain valid prepared descriptors");
}
