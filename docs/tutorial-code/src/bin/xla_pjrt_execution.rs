use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{DType, GraphCompiler, Tensor, TracedTensor};
use tenferro_xla::{XlaExecutor, TENFERRO_PJRT_PLUGIN_ENV};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let residual = (actual - expected).abs();
        assert!(
            residual <= 1.0e-3,
            "value {index} differs: actual={actual}, expected={expected}, residual={residual}"
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lhs = TracedTensor::input_symbolic_shape(DType::F32, 2)?;
    let mid = TracedTensor::input_symbolic_shape(DType::F32, 2)?;
    let rhs = TracedTensor::input_symbolic_shape(DType::F32, 2)?;

    let mut compiler = GraphCompiler::new();
    let product = compiler.einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")?;
    let y = product.abs()?.sqrt()?.log1p()?.exp()?;
    let program = compiler.compile_with_input_specs(
        &y,
        &[
            (&lhs, DType::F32, &[2, 3]),
            (&mid, DType::F32, &[3, 4]),
            (&rhs, DType::F32, &[4, 2]),
        ],
    )?;

    let module = XlaExecutor::default().lower_to_stablehlo(program.semantic_program())?;
    let stablehlo = module.as_str();
    assert!(stablehlo.contains("stablehlo.dot_general"));
    assert!(stablehlo.contains("stablehlo.abs"));
    assert!(stablehlo.contains("stablehlo.sqrt"));
    assert!(stablehlo.contains("stablehlo.log_plus_one"));
    assert!(stablehlo.contains("stablehlo.exponential"));

    if std::env::var_os(TENFERRO_PJRT_PLUGIN_ENV).is_none() {
        return Ok(());
    }

    let lhs_values = vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mid_values = vec![
        1.0_f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let rhs_values = vec![1.0_f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let lhs_input = Tensor::from_vec_col_major(vec![2, 3], lhs_values.clone())?;
    let mid_input = Tensor::from_vec_col_major(vec![3, 4], mid_values.clone())?;
    let rhs_input = Tensor::from_vec_col_major(vec![4, 2], rhs_values.clone())?;

    let output = XlaExecutor::from_env()?.run_with_inputs(
        program.semantic_program(),
        &[&lhs_input, &mid_input, &rhs_input],
    )?;
    assert_eq!(output.shape(), &[2, 2]);
    assert_close(
        output.as_slice::<f32>().unwrap(),
        &[29.495_613, 43.871_902, 32.622_776, 48.539_455],
    );

    Ok(())
}
