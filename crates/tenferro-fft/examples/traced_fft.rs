use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_fft::register_runtime)?;
    let out = executor.run(&program)?;
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
