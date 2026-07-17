#![cfg(feature = "cuda")]

// Run with: cargo test --features cuda -- --ignored

use crate::support;
use support::RunTraced;
use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn upload_traced(backend: &CudaBackend, tensor: &Tensor) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(upload_tensor(backend.runtime(), tensor).unwrap())
        .unwrap()
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

    let gpu_backend = CudaBackend::new(0).unwrap();
    let a = upload_traced(&gpu_backend, &a_host);
    let b = upload_traced(&gpu_backend, &b_host);
    let c = upload_traced(&gpu_backend, &c_host);
    let mut engine = GraphExecutor::new(gpu_backend);

    let sum = a.add(&b).unwrap();
    let result_traced = sum.mul(&c).unwrap();

    let result = result_traced.run_with(&mut engine).unwrap();
    let result = download_tensor(engine.backend().runtime(), &result).unwrap();
    let values = result
        .as_slice::<f32>()
        .expect("expected downloaded F32 tensor");

    let expected = [0.15_f32, 0.1, 0.5];
    assert_eq!(values.len(), expected.len());
    for (got, want) in values.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1.0e-6, "got {got}, want {want}");
    }
}
