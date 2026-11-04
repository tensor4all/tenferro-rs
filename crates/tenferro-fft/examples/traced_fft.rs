use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime_with_fft(engine: &CpuEngine) -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(engine)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft(None, -1, FftNorm::Backward)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y)?;
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime_with_fft(&engine)?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
    assert_eq!(outputs.len(), 1);
    let out = outputs.remove(0).into_tensor(&mut engine)?;
    assert_eq!(out.shape(), &[4]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );

    Ok(())
}
