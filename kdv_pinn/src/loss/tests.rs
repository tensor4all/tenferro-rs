use super::*;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

#[test]
fn mean_square_computes_correctly() {
    let pred = TracedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 3.0]);
    let target = TracedTensor::from_vec_col_major(vec![2, 1], vec![0.0_f64, 1.0]);
    let loss = mean_square(&pred, &target, 2);
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&loss).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run(&program).unwrap();
    assert!((out.as_slice::<f64>().unwrap()[0] - 2.5).abs() < 1e-6);
}
