#![cfg(feature = "cuda")]

use tenferro_ad::TracedTensorAdExt;
mod support;
use support::RunTraced;
use tenferro_gpu::cubecl::{download_tensor, upload_tensor, CubeclBackend};
use tenferro_runtime::{
    CpuBackend, DotGeneralConfig, GraphExecutor, Tensor, TracedTensor, TypedTensor,
};
use tenferro_tensor::Buffer;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn assert_f64_tensor_close(actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) {
    match (actual, expected) {
        (Tensor::F64(actual), Tensor::F64(expected)) => {
            assert_eq!(actual.shape(), expected.shape());
            for (idx, (&actual, &expected)) in actual
                .host_data()
                .iter()
                .zip(expected.host_data().iter())
                .enumerate()
            {
                let tol = atol + rtol * expected.abs();
                assert!(
                    (actual - expected).abs() <= tol,
                    "index {idx}: actual={actual}, expected={expected}, tol={tol}"
                );
            }
        }
        _ => panic!("expected f64 tensors, got actual={actual:?} expected={expected:?}"),
    }
}

fn assert_device_backed(tensor: &Tensor) {
    fn is_cubecl<T: 'static>(buffer: &Buffer<T>) -> bool {
        matches!(buffer, Buffer::Backend(buffer) if buffer.backend_family() == "cubecl")
    }

    match tensor {
        Tensor::F32(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::F64(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::I64(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::C32(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::C64(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::I32(inner) => assert!(is_cubecl(&inner.buffer)),
        Tensor::Bool(inner) => assert!(is_cubecl(&inner.buffer)),
    }
}

fn eval_cpu_tensor(engine: &mut GraphExecutor<CpuBackend>, tensor: &mut TracedTensor) -> Tensor {
    tensor.run_with(engine).unwrap().clone()
}

fn eval_gpu_tensor(engine: &mut GraphExecutor<CubeclBackend>, tensor: &mut TracedTensor) -> Tensor {
    let evaluated = tensor.run_with(engine).unwrap();
    assert_device_backed(&evaluated);
    download_tensor(engine.backend().runtime(), &evaluated).unwrap()
}

fn upload_traced(backend: &CubeclBackend, tensor: &Tensor) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(upload_tensor(backend.runtime(), tensor).unwrap())
}

fn matmul(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )
}

#[test]
fn test_gpu_matmul_vjp() {
    let a_host = f64_tensor(
        vec![3, 4],
        vec![
            1.0, -2.0, 0.5, //
            3.0, 1.5, -0.25, //
            -1.0, 0.75, 2.5, //
            4.0, -3.0, 1.25,
        ],
    );
    let b_host = f64_tensor(
        vec![4, 5],
        vec![
            0.5, -1.5, 2.0, 0.25, //
            1.25, 0.75, -0.5, 3.0, //
            -2.0, 1.0, 0.5, -1.25, //
            0.75, -0.25, 1.5, 2.25, //
            -1.0, 0.5, 2.75, -0.75,
        ],
    );
    let cotangent_host = f64_tensor(
        vec![3, 5],
        vec![
            1.0, -0.5, 0.25, //
            2.0, 1.5, -1.0, //
            -0.75, 0.125, 0.5, //
            0.25, -1.5, 2.5, //
            -0.5, 1.0, 0.75,
        ],
    );

    let a_cpu = TracedTensor::from_tensor_concrete_shape(a_host.clone());
    let b_cpu = TracedTensor::from_tensor_concrete_shape(b_host.clone());
    let cotangent_cpu = TracedTensor::from_tensor_concrete_shape(cotangent_host.clone());
    let mut cpu_engine = GraphExecutor::new(CpuBackend::new());
    let y_cpu = matmul(&a_cpu, &b_cpu);
    let mut grad_a_cpu = y_cpu.vjp(&a_cpu, &cotangent_cpu);
    let mut grad_b_cpu = y_cpu.vjp(&b_cpu, &cotangent_cpu);
    let cpu_grad_a = eval_cpu_tensor(&mut cpu_engine, &mut grad_a_cpu);
    let cpu_grad_b = eval_cpu_tensor(&mut cpu_engine, &mut grad_b_cpu);

    let gpu_backend = CubeclBackend::new(0).unwrap();
    let a_gpu = upload_traced(&gpu_backend, &a_host);
    let b_gpu = upload_traced(&gpu_backend, &b_host);
    let cotangent_gpu = upload_traced(&gpu_backend, &cotangent_host);
    let mut gpu_engine = GraphExecutor::new(gpu_backend);
    let y_gpu = matmul(&a_gpu, &b_gpu);
    let mut grad_a_gpu = y_gpu.vjp(&a_gpu, &cotangent_gpu);
    let mut grad_b_gpu = y_gpu.vjp(&b_gpu, &cotangent_gpu);
    let gpu_grad_a = eval_gpu_tensor(&mut gpu_engine, &mut grad_a_gpu);
    let gpu_grad_b = eval_gpu_tensor(&mut gpu_engine, &mut grad_b_gpu);

    assert_f64_tensor_close(&gpu_grad_a, &cpu_grad_a, 1.0e-10, 1.0e-10);
    assert_f64_tensor_close(&gpu_grad_b, &cpu_grad_b, 1.0e-10, 1.0e-10);
}
