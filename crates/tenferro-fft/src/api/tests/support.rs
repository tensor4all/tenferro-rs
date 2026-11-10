//! Shared helpers of the FFT registry tests: sample inputs, bit-level
//! flattening and the traced-graph runner.

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{DType, Tensor};

pub(super) type TestResult<T> = Result<T, Box<dyn std::error::Error>>;
/// Bits of the plain call and bits of the `_with(default options)` call.
pub(super) type Pair = (Vec<u64>, Vec<u64>);

/// A real 3 x 4 matrix.
pub(super) fn real() -> tenferro_tensor::Result<Tensor> {
    Tensor::from_vec_col_major(
        vec![3, 4],
        (0..12)
            .map(|i| 1.0 + 0.5 * f64::from(i) - 0.125 * f64::from(i * i % 5))
            .collect::<Vec<_>>(),
    )
}

/// A complex 3 x 4 matrix.
pub(super) fn complex() -> tenferro_tensor::Result<Tensor> {
    Tensor::from_vec_col_major(
        vec![3, 4],
        (0..12)
            .map(|i| Complex64::new(f64::from(i) - 2.0, 0.25 * f64::from(i % 3) - 0.5))
            .collect::<Vec<_>>(),
    )
}

/// Shape, dtype tag and every element as raw bits.
pub(super) fn tensor_bits(t: &Tensor) -> TestResult<Vec<u64>> {
    let mut out: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
    out.push(u64::MAX);
    match t.dtype() {
        DType::F64 => out.extend(t.as_slice::<f64>()?.iter().map(|x| x.to_bits())),
        DType::F32 => out.extend(t.as_slice::<f32>()?.iter().map(|x| u64::from(x.to_bits()))),
        DType::C64 => out.extend(
            t.as_slice::<Complex64>()?
                .iter()
                .flat_map(|x| [x.re.to_bits(), x.im.to_bits()]),
        ),
        DType::C32 => out.extend(
            t.as_slice::<Complex32>()?
                .iter()
                .flat_map(|x| [u64::from(x.re.to_bits()), u64::from(x.im.to_bits())]),
        ),
        other => return Err(format!("unexpected result dtype {other:?}").into()),
    }
    Ok(out)
}

/// Compiles and runs a traced graph without inputs; one tensor per output.
pub(super) fn run_traced(outputs: &[&TracedTensor]) -> TestResult<Vec<Tensor>> {
    let program = GraphCompiler::new().compile_many(outputs)?;
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(crate::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let results = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
        runtime.run(&mut engines, &prepared, &[])?
    };
    results
        .into_iter()
        .map(|r| r.into_tensor(&mut engine).map_err(Into::into))
        .collect()
}

/// Runs two traced results in one graph and returns both bit sets.
pub(super) fn traced_pair(plain: &TracedTensor, with: &TracedTensor) -> TestResult<Pair> {
    let tensors = run_traced(&[plain, with])?;
    Ok((tensor_bits(&tensors[0])?, tensor_bits(&tensors[1])?))
}
