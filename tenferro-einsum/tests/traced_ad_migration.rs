#![cfg(feature = "autodiff")]

use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TracedTensorAdExt};
use tenferro_einsum::einsum;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data)
}

fn run_traced(tensor: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
    executor.run(&program).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn grad_einsum_matmul_real_uses_extension_ad_rule() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, -2.0, 0.5, 3.0, 1.25, -0.75],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![2.0, 0.25, -1.5, 4.0, 0.75, -0.5],
    ));
    let mut compiler = GraphCompiler::new();

    let y = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    let grad_a = y.reduce_sum(&[0, 1]).grad(&a).unwrap();
    let result = run_traced(&grad_a);

    assert_eq!(result.shape(), &[2, 3]);
    assert_close(
        result.as_slice::<f64>().unwrap(),
        &[6.0, 6.0, 1.0, 1.0, -2.0, -2.0],
    );
}
