use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime_with_fft() -> Result<Runtime, Box<dyn std::error::Error>> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(
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
    let runtime = cpu_runtime_with_fft()?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.len(), 1);
    let out = outputs.remove(0);
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
