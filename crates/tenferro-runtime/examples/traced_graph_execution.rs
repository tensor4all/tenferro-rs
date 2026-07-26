use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime() -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(
        &CpuBackend::new(),
    )?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    let sum = (&a + &b)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&sum)?;
    let runtime = cpu_runtime()?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.len(), 1);
    let result = outputs.pop().unwrap();

    assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    Ok(())
}
