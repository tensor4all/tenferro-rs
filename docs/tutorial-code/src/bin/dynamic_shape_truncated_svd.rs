use tenferro_cpu::CpuBackend;
use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_linalg::{SvdOptions, TracedTensorLinalgExt};
use tenferro_runtime::{
    CompareDir, CompiledGraph, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor,
};

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "value {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }
}

fn diagonal_matrix(diagonal: &[f64]) -> Result<Tensor, tenferro_runtime::Error> {
    let n = diagonal.len();
    let mut values = vec![0.0_f64; n * n];
    for (index, value) in diagonal.iter().enumerate() {
        values[index + index * n] = *value;
    }
    Ok(Tensor::from_vec_col_major(vec![n, n], values)?)
}

fn truncated_expected(diagonal: &[f64], threshold: f64) -> Vec<f64> {
    let n = diagonal.len();
    let mut values = vec![0.0_f64; n * n];
    for (index, value) in diagonal.iter().enumerate() {
        if value.abs() > threshold {
            values[index + index * n] = *value;
        }
    }
    values
}

fn run_case(
    executor: &mut GraphExecutor<CpuBackend>,
    reconstructed_program: &CompiledGraph,
    singular_values_program: &CompiledGraph,
    x: &TracedTensor,
    input: &Tensor,
    expected_rank: usize,
    expected_values: &[f64],
) -> Result<(), tenferro_runtime::Error> {
    let reconstructed = executor.run_with_inputs(reconstructed_program, &[(x, input)])?;
    let singular_values = executor.run_with_inputs(singular_values_program, &[(x, input)])?;

    assert_eq!(singular_values.shape(), &[expected_rank]);
    assert_eq!(reconstructed.shape(), &[4, 4]);
    assert_close(
        reconstructed.as_slice::<f64>().unwrap(),
        expected_values,
        1.0e-10,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4, 4])?;
    let (u, s, vt) = x.svd_with_options(SvdOptions::default().derivative_eps(1.0e-12))?;

    let threshold = TracedTensor::from_vec_col_major(vec![], vec![0.5_f64])?;
    let keep_count = s
        .compare(&threshold, CompareDir::Gt)?
        .convert(DType::F64)?
        .reduce_sum(Some(&[0]))?;

    let u_truncated = u.dynamic_truncate(&keep_count, 1)?;
    let s_truncated = s.dynamic_truncate(&keep_count, 0)?;
    let vt_truncated = vt.dynamic_truncate(&keep_count, 0)?;

    let mut compiler = GraphCompiler::new();
    let reconstructed = compiler.einsum_with(
        &[&u_truncated, &s_truncated, &vt_truncated],
        "ik,k,kj->ij",
        tenferro_einsum::EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )?;
    let input_specs = [(&x, DType::F64, &[4, 4][..])];
    let reconstructed_program = compiler.compile_with_input_specs(&reconstructed, &input_specs)?;
    let singular_values_program = compiler.compile_with_input_specs(&s_truncated, &input_specs)?;

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_linalg::register_runtime)?;
    executor.register_extension(tenferro_einsum::register_runtime)?;

    let rank2 = diagonal_matrix(&[4.0, 3.0, 0.1, 0.01])?;
    run_case(
        &mut executor,
        &reconstructed_program,
        &singular_values_program,
        &x,
        &rank2,
        2,
        &truncated_expected(&[4.0, 3.0, 0.1, 0.01], 0.5),
    )?;

    let rank3 = diagonal_matrix(&[4.0, 3.0, 2.0, 0.01])?;
    run_case(
        &mut executor,
        &reconstructed_program,
        &singular_values_program,
        &x,
        &rank3,
        3,
        &truncated_expected(&[4.0, 3.0, 2.0, 0.01], 0.5),
    )?;

    Ok(())
}
