#![cfg(feature = "cuda")]

use tenferro_tensor::TensorDeviceTransfer;

// Run with: cargo test --features cuda -- --ignored

use tenferro_ad::EagerRuntime;
use tenferro_gpu::{
    cuda::gpu_available, cuda::upload_tensor, cuda::CudaDeviceId, cuda::CudaEngine,
};
use tenferro_runtime::{Tensor, TensorRead, TypedTensor};

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_f32_gpu_fusion_chain_e2e() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let a_host = f32_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b_host = f32_tensor(vec![3], vec![0.5, -1.0, 2.0]);
    let c_host = f32_tensor(vec![3], vec![0.1, 0.1, 0.1]);

    let mut upload_backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let a_device = upload_tensor(upload_backend.runtime(), &a_host).unwrap();
    let b_device = upload_tensor(upload_backend.runtime(), &b_host).unwrap();
    let c_device = upload_tensor(upload_backend.runtime(), &c_host).unwrap();
    let ctx = EagerRuntime::new().unwrap();
    let a = ctx
        .with_eager_session(&mut upload_backend, |s| s.constant_from(a_device))
        .unwrap();
    let b = ctx
        .with_eager_session(&mut upload_backend, |s| s.constant_from(b_device))
        .unwrap();
    let c = ctx
        .with_eager_session(&mut upload_backend, |s| s.constant_from(c_device))
        .unwrap();
    let result = ctx
        .with_eager_session(&mut upload_backend, |s| {
            let sum = s.add(&a, &b)?;
            s.mul(&sum, &c)
        })
        .unwrap()
        .to_tensor(&mut upload_backend)
        .unwrap();

    let result = upload_backend
        .download_to_host(TensorRead::from_tensor(&result))
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
