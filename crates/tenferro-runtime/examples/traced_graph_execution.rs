use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    let sum = (&a + &b)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&sum)?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let result = executor.run(&program)?;

    assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    Ok(())
}
