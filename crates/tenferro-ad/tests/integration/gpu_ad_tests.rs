#![cfg(feature = "cuda")]

use crate::support;
use support::{cpu_runtime, RunTraced};
use tenferro_ad::{EagerRuntime, EagerTensor, TracedTensorAdExt};
use tenferro_gpu::{download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_runtime::{DotGeneralConfig, Tensor, TracedTensor, TypedTensor};
use tenferro_tensor::{Buffer, TensorDeviceTransfer};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn assert_f64_tensor_close(actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) {
    match (actual, expected) {
        (Tensor::F64(actual), Tensor::F64(expected)) => {
            assert_eq!(actual.shape(), expected.shape());
            for (idx, (&actual, &expected)) in actual
                .host_data()
                .unwrap()
                .iter()
                .zip(expected.host_data().unwrap().iter())
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
        Tensor::F32(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::F64(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::I64(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::C32(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::C64(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::I32(inner) => assert!(is_cubecl(inner.buffer())),
        Tensor::Bool(inner) => assert!(is_cubecl(inner.buffer())),
    }
}

fn eval_cpu_tensor(runtime: &tenferro_runtime::Runtime, tensor: &TracedTensor) -> Tensor {
    tensor.run_with(runtime).unwrap()
}

fn upload_traced(backend: &CudaBackend, tensor: &Tensor) -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(upload_tensor(backend.runtime(), tensor).unwrap())
        .unwrap()
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
    .unwrap()
}

#[test]
fn test_gpu_eager_backward_smoke() {
    if !gpu_available() {
        return;
    }

    let upload_backend = CudaBackend::new(0).unwrap();
    let x_gpu = upload_tensor(
        upload_backend.runtime(),
        &f64_tensor(vec![2], vec![2.0_f64, 3.0]),
    )
    .unwrap();
    let seed_gpu = upload_tensor(
        upload_backend.runtime(),
        &f64_tensor(vec![2], vec![1.0_f64, 1.0]),
    )
    .unwrap();
    let ctx = EagerRuntime::with_cuda_backend(upload_backend);
    let x = EagerTensor::requires_grad_in(x_gpu, ctx.clone()).unwrap();
    let seed = EagerTensor::from_tensor_in(seed_gpu, ctx.clone()).unwrap();
    let y = x.mul(&x).unwrap();

    y.backward_with(&seed).unwrap();
    let grad = x.grad().unwrap().unwrap();
    let grad_host = ctx
        .with_backend_mut(|backend| backend.download_to_host(grad.as_ref()))
        .unwrap()
        .unwrap();

    assert_f64_tensor_close(
        &grad_host,
        &f64_tensor(vec![2], vec![4.0_f64, 6.0]),
        1.0e-10,
        1.0e-10,
    );
}

#[test]
fn test_gpu_matmul_vjp() {
    if !gpu_available() {
        return;
    }

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

    let a_cpu = TracedTensor::from_tensor_concrete_shape(a_host.clone()).unwrap();
    let b_cpu = TracedTensor::from_tensor_concrete_shape(b_host.clone()).unwrap();
    let cotangent_cpu = TracedTensor::from_tensor_concrete_shape(cotangent_host.clone()).unwrap();
    let cpu_engine = cpu_runtime();
    let y_cpu = matmul(&a_cpu, &b_cpu);
    let grad_a_cpu = y_cpu.vjp(&a_cpu, &cotangent_cpu).unwrap();
    let grad_b_cpu = y_cpu.vjp(&b_cpu, &cotangent_cpu).unwrap();
    let cpu_grad_a = eval_cpu_tensor(&cpu_engine, &grad_a_cpu);
    let cpu_grad_b = eval_cpu_tensor(&cpu_engine, &grad_b_cpu);

    let gpu_backend = CudaBackend::new(0).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(gpu_backend);
    let a_gpu = EagerTensor::from_tensor_in(
        upload_tensor(ctx.backend_runtime(), &a_host).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b_gpu = EagerTensor::from_tensor_in(
        upload_tensor(ctx.backend_runtime(), &b_host).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let cotangent_gpu = EagerTensor::from_tensor_in(
        upload_tensor(ctx.backend_runtime(), &cotangent_host).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y_gpu = a_gpu
        .dot_general(
            &b_gpu,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    let gpu_grad_a = ctx
        .vjp(&y_gpu, &a_gpu, &cotangent_gpu)
        .unwrap()
        .materialized()
        .unwrap();
    let gpu_grad_b = ctx
        .vjp(&y_gpu, &b_gpu, &cotangent_gpu)
        .unwrap()
        .materialized()
        .unwrap();
    assert_device_backed(&gpu_grad_a);
    assert_device_backed(&gpu_grad_b);
    let gpu_grad_a = download_tensor(ctx.backend_runtime(), &gpu_grad_a).unwrap();
    let gpu_grad_b = download_tensor(ctx.backend_runtime(), &gpu_grad_b).unwrap();

    assert_f64_tensor_close(&gpu_grad_a, &cpu_grad_a, 1.0e-10, 1.0e-10);
    assert_f64_tensor_close(&gpu_grad_b, &cpu_grad_b, 1.0e-10, 1.0e-10);
}
