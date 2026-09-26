#![cfg(feature = "webgpu")]

use tenferro_gpu::{webgpu::upload_webgpu_tensor, webgpu::WebGpuBackend};
use tenferro_tensor::{AllocationId, BackendSessionHost, DType, Tensor, TensorRead};

fn allocation_id(tensor: &Tensor) -> Option<AllocationId> {
    fn typed<T: tenferro_tensor::TensorScalar>(tensor: &Tensor) -> Option<AllocationId> {
        tensor.as_typed::<T>().and_then(|t| t.allocation_id())
    }

    match tensor.dtype() {
        DType::F32 => typed::<f32>(tensor),
        DType::F64 => typed::<f64>(tensor),
        DType::I32 => typed::<i32>(tensor),
        DType::I64 => typed::<i64>(tensor),
        DType::Bool => typed::<bool>(tensor),
        DType::C32 => typed::<num_complex::Complex32>(tensor),
        DType::C64 => typed::<num_complex::Complex64>(tensor),
        // The fixture covers the preset dtypes; an externally defined payload has
        // no fixture and would change what this test asserts.
        _ => None,
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
    let Some(typed) = tensor.as_typed::<f32>() else {
        panic!("provider contract uses f32")
    };

    assert!(typed.allocation_domain().is_some());
    assert!(typed.allocation_id().is_some());
    typed
        .prepare_device_read("storage_provider_webgpu")
        .expect("root preparation should accept the checked descriptor");

    let duplicate = backend
        .with_backend_session(|session| {
            session.to_contiguous_read(TensorRead::from_tensor(&tensor))
        })
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
    let Some(typed) = tensor.as_typed::<f32>() else {
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
    let Some(typed) = tensor.as_typed::<f32>() else {
        panic!("provider contract uses f32")
    };
    assert_eq!(typed.n_elements(), 0);
    typed
        .prepare_device_read("storage_provider_webgpu_empty")
        .expect("empty roots remain valid prepared descriptors");
}
