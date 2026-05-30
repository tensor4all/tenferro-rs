#![cfg(feature = "autodiff")]

use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::einsum;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

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

fn run_symbolic_chain_grad(
    grad: &TracedTensor,
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
    a_value: &Tensor,
    b_value: &Tensor,
    c_value: &Tensor,
) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            grad,
            &[
                (a, DType::F64, &[2, 3]),
                (b, DType::F64, &[3, 4]),
                (c, DType::F64, &[4, 2]),
            ],
        )
        .unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_einsum::register_runtime)
        .unwrap();
    executor
        .run_with_inputs(&program, &[(a, a_value), (b, b_value), (c, c_value)])
        .unwrap()
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

#[test]
fn symbolic_grad_einsum_with_explicit_path_uses_extension_ad_rule() {
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let c = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut compiler = GraphCompiler::new();

    let y = tenferro_einsum::einsum_with(
        &mut compiler,
        &[&a, &b, &c],
        "ij,jk,kl->il",
        tenferro_einsum::EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
    )
    .unwrap();
    let loss = y.reduce_sum(&[0, 1]);
    let grad_a = loss.grad(&a).unwrap();
    let grad_b = loss.grad(&b).unwrap();
    let grad_c = loss.grad(&c).unwrap();

    let a_value = f64_tensor(vec![2, 3], vec![1.0, -2.0, 0.5, 3.0, 1.25, -0.75]);
    let b_value = f64_tensor(
        vec![3, 4],
        vec![
            2.0, 0.25, -1.5, 4.0, 0.75, -0.5, 1.0, -2.0, 0.5, 1.25, -1.0, 3.0,
        ],
    );
    let c_value = f64_tensor(vec![4, 2], vec![1.0, 2.0, -0.5, 0.75, 1.5, -1.0, 2.5, 0.25]);
    let out_a = run_symbolic_chain_grad(&grad_a, &a, &b, &c, &a_value, &b_value, &c_value);
    let out_b = run_symbolic_chain_grad(&grad_b, &a, &b, &c, &a_value, &b_value, &c_value);
    let out_c = run_symbolic_chain_grad(&grad_c, &a, &b, &c, &a_value, &b_value, &c_value);

    assert_eq!(out_a.shape(), &[2, 3]);
    assert_close(
        out_a.as_slice::<f64>().unwrap(),
        &[12.25, 12.25, -3.625, -3.625, -0.25, -0.25],
    );
    assert_eq!(out_b.shape(), &[3, 4]);
    assert_close(
        out_b.as_slice::<f64>().unwrap(),
        &[
            -2.5, 8.75, 1.25, -1.0, 3.5, 0.5, -2.0, 7.0, 1.0, -1.0, 3.5, 0.5,
        ],
    );
    assert_eq!(out_c.shape(), &[4, 2]);
    assert_close(
        out_c.as_slice::<f64>().unwrap(),
        &[-1.875, -1.625, -7.75, -3.25, -1.875, -1.625, -7.75, -3.25],
    );
}
