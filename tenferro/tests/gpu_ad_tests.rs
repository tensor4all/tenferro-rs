#![cfg(feature = "cubecl")]

use tenferro::traced_tensor::einsum;
use tenferro::traced_tensor::svd;
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};
use tenferro_tensor::cubecl::{download_tensor, upload_tensor, CubeclBackend};
use tenferro_tensor::Buffer;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn assert_f64_tensor_close(actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) {
    match (actual, expected) {
        (Tensor::F64(actual), Tensor::F64(expected)) => {
            assert_eq!(actual.shape.as_slice(), expected.shape.as_slice());
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
    match tensor {
        Tensor::F32(inner) => assert!(matches!(&inner.buffer, Buffer::Cubecl(_))),
        Tensor::F64(inner) => assert!(matches!(&inner.buffer, Buffer::Cubecl(_))),
        Tensor::I64(inner) => assert!(matches!(&inner.buffer, Buffer::Cubecl(_))),
        Tensor::C32(inner) => assert!(matches!(&inner.buffer, Buffer::Cubecl(_))),
        Tensor::C64(inner) => assert!(matches!(&inner.buffer, Buffer::Cubecl(_))),
    }
}

fn eval_cpu_tensor(engine: &mut Engine<CpuBackend>, tensor: &mut TracedTensor) -> Tensor {
    tensor.eval(engine).unwrap().clone()
}

fn eval_gpu_tensor(engine: &mut Engine<CubeclBackend>, tensor: &mut TracedTensor) -> Tensor {
    let evaluated = tensor.eval(engine).unwrap();
    assert_device_backed(evaluated);
    download_tensor(engine.backend().runtime(), evaluated).unwrap()
}

fn upload_traced(backend: &CubeclBackend, tensor: &Tensor) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(upload_tensor(backend.runtime(), tensor).unwrap())
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
    let mut cpu_engine = Engine::new(CpuBackend::new());
    let y_cpu = einsum(&mut cpu_engine, &[&a_cpu, &b_cpu], "ij,jk->ik").unwrap();
    let mut grad_a_cpu = y_cpu.vjp(&a_cpu, &cotangent_cpu);
    let mut grad_b_cpu = y_cpu.vjp(&b_cpu, &cotangent_cpu);
    let cpu_grad_a = eval_cpu_tensor(&mut cpu_engine, &mut grad_a_cpu);
    let cpu_grad_b = eval_cpu_tensor(&mut cpu_engine, &mut grad_b_cpu);

    let gpu_backend = CubeclBackend::new(0).unwrap();
    let a_gpu = upload_traced(&gpu_backend, &a_host);
    let b_gpu = upload_traced(&gpu_backend, &b_host);
    let cotangent_gpu = upload_traced(&gpu_backend, &cotangent_host);
    let mut gpu_engine = Engine::new(gpu_backend);
    let y_gpu = einsum(&mut gpu_engine, &[&a_gpu, &b_gpu], "ij,jk->ik").unwrap();
    let mut grad_a_gpu = y_gpu.vjp(&a_gpu, &cotangent_gpu);
    let mut grad_b_gpu = y_gpu.vjp(&b_gpu, &cotangent_gpu);
    let gpu_grad_a = eval_gpu_tensor(&mut gpu_engine, &mut grad_a_gpu);
    let gpu_grad_b = eval_gpu_tensor(&mut gpu_engine, &mut grad_b_gpu);

    assert_f64_tensor_close(&gpu_grad_a, &cpu_grad_a, 1.0e-10, 1.0e-10);
    assert_f64_tensor_close(&gpu_grad_b, &cpu_grad_b, 1.0e-10, 1.0e-10);
}

#[test]
fn test_gpu_svd_vjp() {
    let a_host = f64_tensor(
        vec![4, 3],
        vec![
            4.0, 0.0, 0.0, 0.3, //
            0.2, 3.0, 0.0, -0.1, //
            0.1, -0.2, 1.5, 0.5,
        ],
    );
    let cotangent_host = f64_tensor(vec![3], vec![1.0, -0.5, 0.25]);

    let a_cpu = TracedTensor::from_tensor_concrete_shape(a_host.clone());
    let cotangent_cpu = TracedTensor::from_tensor_concrete_shape(cotangent_host.clone());
    let mut cpu_engine = Engine::new(CpuBackend::new());
    let (_u_cpu, s_cpu, _vh_cpu) = svd(&a_cpu);
    let mut grad_cpu = s_cpu.vjp(&a_cpu, &cotangent_cpu);
    let cpu_grad = eval_cpu_tensor(&mut cpu_engine, &mut grad_cpu);

    let gpu_backend = CubeclBackend::new(0).unwrap();
    let a_gpu = upload_traced(&gpu_backend, &a_host);
    let cotangent_gpu = upload_traced(&gpu_backend, &cotangent_host);
    let mut gpu_engine = Engine::new(gpu_backend);
    let (_u_gpu, s_gpu, _vh_gpu) = svd(&a_gpu);
    let mut grad_gpu = s_gpu.vjp(&a_gpu, &cotangent_gpu);
    let gpu_grad = eval_gpu_tensor(&mut gpu_engine, &mut grad_gpu);

    assert_f64_tensor_close(&gpu_grad, &cpu_grad, 1.0e-5, 1.0e-6);
}
