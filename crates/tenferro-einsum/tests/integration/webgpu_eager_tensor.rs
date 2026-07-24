#![cfg(all(feature = "autodiff", feature = "webgpu"))]

use num_complex::Complex32;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_einsum::{EagerEinsumExt, TraceContextEinsumExt};
use tenferro_gpu::{download_webgpu_tensor, upload_webgpu_tensor, webgpu_available};
use tenferro_gpu::{WebGpuBackend, WebGpuRuntime};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{CompiledGraph, DType, GraphCompiler, GraphExecutor, Tensor, TraceContext};

fn matmul2_col_major(lhs: &[Complex32], rhs: &[Complex32]) -> [Complex32; 4] {
    let a00 = lhs[0];
    let a10 = lhs[1];
    let a01 = lhs[2];
    let a11 = lhs[3];
    let b00 = rhs[0];
    let b10 = rhs[1];
    let b01 = rhs[2];
    let b11 = rhs[3];
    [
        a00 * b00 + a01 * b10,
        a10 * b00 + a11 * b10,
        a00 * b01 + a01 * b11,
        a10 * b01 + a11 * b11,
    ]
}

fn assert_complex_close(actual: &[Complex32], expected: &[Complex32]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual.re - expected.re).abs() <= 1e-4,
            "real mismatch: actual={actual:?} expected={expected:?}"
        );
        assert!(
            (actual.im - expected.im).abs() <= 1e-4,
            "imag mismatch: actual={actual:?} expected={expected:?}"
        );
    }
}

fn batched_matmul_f32_reference() -> Vec<f32> {
    vec![58.0, 139.0, 64.0, 154.0, 5800.0, 13900.0, 6400.0, 15400.0]
}

fn assert_f32_close(actual: &[f32], expected: &[f32]) {
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "f32 mismatch: actual={actual} expected={expected}"
        );
    }
}

fn compile_einsum(dtype: DType, shapes: &[&[usize]], subscripts: &str) -> CompiledGraph {
    let mut trace = TraceContext::new();
    let inputs = shapes
        .iter()
        .map(|shape| {
            trace
                .input(ProgramInputSpec::new(dtype, DimExpr::from_concrete(shape)))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let output = trace.einsum(&inputs, subscripts).unwrap();
    let graph = trace.finish(&[output]).unwrap();
    GraphCompiler::new().compile_traced_graph(&graph).unwrap()
}

#[test]
fn eager_tensor_einsum_runs_rank2_f32_matmul_on_webgpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let runtime = WebGpuRuntime::new_default().unwrap();
    let ctx = EagerRuntime::with_webgpu_backend(WebGpuBackend::from_runtime(runtime.clone()));
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let lhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &lhs).unwrap(), ctx.clone())
            .unwrap();
    let rhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &rhs).unwrap(), ctx).unwrap();

    let out = [&lhs, &rhs].einsum("ij,jk->ik").unwrap();
    let host = download_webgpu_tensor(&runtime, out.materialized().unwrap().as_ref()).unwrap();

    assert_eq!(host.shape(), &[2, 2]);
    let actual = host.as_slice::<f32>().unwrap();
    let expected = [58.0_f32, 139.0, 64.0, 154.0];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
}

#[test]
fn eager_tensor_einsum_runs_batched_f32_matmul_on_webgpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let runtime = WebGpuRuntime::new_default().unwrap();
    let ctx = EagerRuntime::with_webgpu_backend(WebGpuBackend::from_runtime(runtime.clone()));
    let lhs = Tensor::from_vec_col_major(
        vec![2, 3, 2],
        vec![
            1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0, 10.0, 40.0, 20.0, 50.0, 30.0, 60.0,
        ],
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![3, 2, 2],
        vec![
            7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0, 70.0, 90.0, 110.0, 80.0, 100.0, 120.0,
        ],
    )
    .unwrap();
    let lhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &lhs).unwrap(), ctx.clone())
            .unwrap();
    let rhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &rhs).unwrap(), ctx).unwrap();

    let out = [&lhs, &rhs].einsum("ikb,kjb->ijb").unwrap();
    let host = download_webgpu_tensor(&runtime, out.materialized().unwrap().as_ref()).unwrap();

    assert_eq!(host.shape(), &[2, 2, 2]);
    assert_f32_close(
        host.as_slice::<f32>().unwrap(),
        &batched_matmul_f32_reference(),
    );
}

#[test]
fn eager_tensor_einsum_runs_rank2_c32_matmul_on_webgpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let runtime = WebGpuRuntime::new_default().unwrap();
    let ctx = EagerRuntime::with_webgpu_backend(WebGpuBackend::from_runtime(runtime.clone()));
    let lhs_data = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -1.0),
        Complex32::new(3.0, 0.25),
        Complex32::new(4.0, 1.0),
    ];
    let rhs_data = vec![
        Complex32::new(5.0, -0.5),
        Complex32::new(6.0, 0.25),
        Complex32::new(7.0, 1.0),
        Complex32::new(8.0, -0.75),
    ];
    let lhs = Tensor::from_vec_col_major(vec![2, 2], lhs_data.clone()).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], rhs_data.clone()).unwrap();
    let lhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &lhs).unwrap(), ctx.clone())
            .unwrap();
    let rhs =
        EagerTensor::from_tensor_in(upload_webgpu_tensor(&runtime, &rhs).unwrap(), ctx).unwrap();

    let out = [&lhs, &rhs].einsum("ij,jk->ik").unwrap();
    let host = download_webgpu_tensor(&runtime, out.materialized().unwrap().as_ref()).unwrap();

    assert_eq!(host.shape(), &[2, 2]);
    let actual = host.as_slice::<Complex32>().unwrap();
    let expected = matmul2_col_major(&lhs_data, &rhs_data);
    assert_complex_close(actual, &expected);
}

#[test]
fn traced_einsum_runs_rank2_f32_matmul_on_webgpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let program = compile_einsum(DType::F32, &[&[2, 3], &[3, 2]], "ij,jk->ik");
    let runtime = WebGpuRuntime::new_default().unwrap();
    let lhs_host =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    let rhs_host =
        Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0]).unwrap();
    let lhs_gpu = upload_webgpu_tensor(&runtime, &lhs_host).unwrap();
    let rhs_gpu = upload_webgpu_tensor(&runtime, &rhs_host).unwrap();
    let mut executor = GraphExecutor::new(WebGpuBackend::from_runtime(runtime.clone()));
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    let out = executor
        .run_with_inputs(&program, &[&lhs_gpu, &rhs_gpu])
        .unwrap();
    executor.backend().synchronize().unwrap();
    let host = download_webgpu_tensor(executor.backend().runtime(), &out).unwrap();

    assert_eq!(host.shape(), &[2, 2]);
    let actual = host.as_slice::<f32>().unwrap();
    let expected = [58.0_f32, 139.0, 64.0, 154.0];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
}

#[test]
fn traced_einsum_runs_batched_f32_matmul_on_webgpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let program = compile_einsum(DType::F32, &[&[2, 3, 2], &[3, 2, 2]], "ikb,kjb->ijb");
    let runtime = WebGpuRuntime::new_default().unwrap();
    let lhs_host = Tensor::from_vec_col_major(
        vec![2, 3, 2],
        vec![
            1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0, 10.0, 40.0, 20.0, 50.0, 30.0, 60.0,
        ],
    )
    .unwrap();
    let rhs_host = Tensor::from_vec_col_major(
        vec![3, 2, 2],
        vec![
            7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0, 70.0, 90.0, 110.0, 80.0, 100.0, 120.0,
        ],
    )
    .unwrap();
    let lhs_gpu = upload_webgpu_tensor(&runtime, &lhs_host).unwrap();
    let rhs_gpu = upload_webgpu_tensor(&runtime, &rhs_host).unwrap();
    let mut executor = GraphExecutor::new(WebGpuBackend::from_runtime(runtime.clone()));
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();

    let out = executor
        .run_with_inputs(&program, &[&lhs_gpu, &rhs_gpu])
        .unwrap();
    executor.backend().synchronize().unwrap();
    let host = download_webgpu_tensor(executor.backend().runtime(), &out).unwrap();

    assert_eq!(host.shape(), &[2, 2, 2]);
    assert_f32_close(
        host.as_slice::<f32>().unwrap(),
        &batched_matmul_f32_reference(),
    );
}
