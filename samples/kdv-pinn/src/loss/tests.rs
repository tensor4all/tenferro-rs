use super::*;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn mean_square_computes_correctly() {
    let pred = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 3.0]).unwrap();
    let target = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 1.0]).unwrap();
    let loss = mean_square(&pred, &target, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&loss).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();
    assert!((out.as_slice::<f64>().unwrap()[0] - 2.5).abs() < 1e-6);
}

#[test]
fn mean_square_single_computes_correctly() {
    let tensor = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 3.0]).unwrap();
    let loss = mean_square_single(&tensor, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&loss).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();
    assert!((out.as_slice::<f64>().unwrap()[0] - 5.0).abs() < 1e-6);
}

#[test]
fn total_loss_weights_pde_only() {
    let residual = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 3.0]).unwrap();
    let zeros = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 0.0]).unwrap();
    let loss = total_loss(
        &residual, &zeros, &zeros, &zeros, &zeros, 2, 2, 2, 1.0, 0.0, 0.0,
    )
    .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&loss).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();
    assert!((out.as_slice::<f64>().unwrap()[0] - 5.0).abs() < 1e-6);
}

#[test]
fn total_loss_applies_pde_weight_once_after_mean_square() {
    let residual = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 3.0]).unwrap();
    let zeros = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 0.0]).unwrap();
    let loss = total_loss(
        &residual, &zeros, &zeros, &zeros, &zeros, 2, 2, 2, 2.0, 0.0, 0.0,
    )
    .unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&loss).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();
    assert!((out.as_slice::<f64>().unwrap()[0] - 10.0).abs() < 1e-6);
}
