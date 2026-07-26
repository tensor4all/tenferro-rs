# Dynamic Shapes: Truncated SVD

Use traced dynamic-shape operations when an output size is known only after
execution starts. This tutorial builds compiled programs that run an SVD, count
singular values above a threshold, truncate `u`, `s`, and `vt` with
`dynamic_truncate`, and reconstruct the thresholded matrix.

The same traced graph and explicit placeholder spec are compiled once, then the
programs run twice below: once with two singular values above the threshold and
once with three. No re-trace or recompile is needed between the two executions.

<!-- snippet-source: docs/tutorial-code/src/bin/dynamic_shape_truncated_svd.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{SvdOptions, TracedTensorLinalgExt};
use tenferro_runtime::{
    CompareDir, CompiledGraph, DType, DotGeneralConfig, GraphCompiler, Runtime, Tensor,
    TracedTensor,
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
    runtime: &Runtime,
    reconstructed_program: &CompiledGraph,
    singular_values_program: &CompiledGraph,
    input: &Tensor,
    threshold: &Tensor,
    expected_rank: usize,
    expected_values: &[f64],
) -> Result<(), tenferro_runtime::Error> {
    let mut reconstructed_outputs =
        runtime.run_compiled(reconstructed_program, &[input, threshold])?;
    assert_eq!(reconstructed_outputs.len(), 1);
    let reconstructed = reconstructed_outputs.remove(0);
    let mut singular_value_outputs =
        runtime.run_compiled(singular_values_program, &[input, threshold])?;
    assert_eq!(singular_value_outputs.len(), 1);
    let singular_values = singular_value_outputs.remove(0);

    assert_eq!(singular_values.shape(), &[expected_rank]);
    assert_eq!(reconstructed.shape(), &[4, 4]);
    assert_close(
        reconstructed.as_slice::<f64>().unwrap(),
        expected_values,
        1.0e-10,
    );
    Ok(())
}

fn cpu_runtime_with_linalg_and_einsum() -> Result<Runtime, Box<dyn std::error::Error>> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    let engine_id = tenferro_cpu::runtime_engine_id()?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(
        engine_id.clone(),
    )?)?;
    builder
        .install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(engine_id)?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4, 4])?;
    let (u, s, vt) = x.svd_with_options(SvdOptions::default().derivative_eps(1.0e-12))?;

    let threshold = TracedTensor::from_vec_col_major(vec![], vec![0.5_f64])?;
    let keep_count = s
        .compare(&threshold, CompareDir::Gt)?
        .convert(DType::F64)?
        .reduce_sum(Some(&[0]))?;

    let s_truncated = s.dynamic_truncate(&keep_count, 0)?;

    let mut compiler = GraphCompiler::new();
    let keep_mask = s.compare(&threshold, CompareDir::Gt)?.convert(DType::F64)?;
    let masked_s = (&s * &keep_mask)?;
    let scaled_u = (&u * &masked_s.broadcast_in_dim(&[4, 4], &[1])?)?;
    let reconstructed = scaled_u.dot_general(
        &vt,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )?;
    let input_specs = [(&x, DType::F64, &[4, 4][..])];
    let reconstructed_program = compiler.compile_with_input_specs(&reconstructed, &input_specs)?;
    let singular_values_program = compiler.compile_with_input_specs(&s_truncated, &input_specs)?;

    let runtime = cpu_runtime_with_linalg_and_einsum()?;

    let threshold_input = Tensor::from_vec_col_major(vec![], vec![0.5_f64])?;
    let rank2 = diagonal_matrix(&[4.0, 3.0, 0.1, 0.01])?;
    run_case(
        &runtime,
        &reconstructed_program,
        &singular_values_program,
        &rank2,
        &threshold_input,
        2,
        &truncated_expected(&[4.0, 3.0, 0.1, 0.01], 0.5),
    )?;

    let rank3 = diagonal_matrix(&[4.0, 3.0, 2.0, 0.01])?;
    run_case(
        &runtime,
        &reconstructed_program,
        &singular_values_program,
        &rank3,
        &threshold_input,
        3,
        &truncated_expected(&[4.0, 3.0, 2.0, 0.01], 0.5),
    )?;

    Ok(())
}
```
<!-- end-snippet-source -->

The shape metadata for the truncated axis is an upper bound in the compiled
program. The concrete extent is resolved at dispatch from the runtime scalar
`keep_count`, then later operations consume the resulting dynamic extent.

For the implementation contract, see
[Dynamic and Symbolic Shape Metadata](../design/dynamic-symbolic-shapes.md).
For the broader eager/traced split, see the
[execution models guide](../guides/execution-models.md).
