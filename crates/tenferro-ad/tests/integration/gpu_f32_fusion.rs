#![cfg(feature = "cuda")]

// Run with: cargo test --features cuda -- --ignored

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_gpu::{gpu_available, upload_tensor, CudaBackend, CudaDeviceId};
use tenferro_runtime::{Tensor, TypedTensor};

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_f32_gpu_fusion_chain_e2e() {
    if !gpu_available() {
        eprintln!("skipping test_f32_gpu_fusion_chain_e2e — no CUDA device found");
        return;
    }
    let a_host = f32_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b_host = f32_tensor(vec![3], vec![0.5, -1.0, 2.0]);
    let c_host = f32_tensor(vec![3], vec![0.1, 0.1, 0.1]);

    let upload_backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let a_device = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_device = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let c_device = upload_tensor(upload_backend.runtime(), &c_host).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend).unwrap();
    let a = EagerTensor::from_tensor_in(a_device, ctx.clone()).unwrap();
    let b = EagerTensor::from_tensor_in(b_device, ctx.clone()).unwrap();
    let c = EagerTensor::from_tensor_in(c_device, ctx.clone()).unwrap();
    let sum = a.add(&b).unwrap();
    let result = sum.mul(&c).unwrap().materialized().unwrap();

    let result = ctx
        .with_execution_session(|session| session.download_to_host(result.as_ref()))
        .unwrap()
        .unwrap();
    let values = result
        .as_slice::<f32>()
        .expect("expected downloaded F32 tensor");

    let expected = [0.15_f32, 0.1, 0.5];
    assert_eq!(values.len(), expected.len());
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1.0e-6, "got {got}, want {want}");
    }
}
