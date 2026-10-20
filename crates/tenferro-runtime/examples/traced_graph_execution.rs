use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime(engine: &CpuEngine) -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(engine)?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    let sum = (&a + &b)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&sum)?;
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime(&engine)?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
    assert_eq!(outputs.len(), 1);
    let result = outputs.pop().unwrap().into_tensor(&mut engine)?;

    assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    Ok(())
}
