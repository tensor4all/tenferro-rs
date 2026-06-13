# Dynamic Shapes: Truncated SVD

Use traced dynamic-shape operations when an output size is known only after
execution starts. This tutorial builds one compiled program that runs an SVD,
counts singular values above a threshold, truncates `u`, `s`, and `vt` with
`dynamic_truncate`, and reconstructs the thresholded matrix.

The same `GraphProgram` runs twice below: once with two singular values above
the threshold and once with three. No re-trace or recompile is needed between
the two executions.

<!-- snippet-source: docs/tutorial-code/src/bin/dynamic_shape_truncated_svd.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{CompareDir, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

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

fn diagonal_matrix(diagonal: &[f64]) -> Tensor {
    let n = diagonal.len();
    let mut values = vec![0.0_f64; n * n];
    for (index, value) in diagonal.iter().enumerate() {
        values[index + index * n] = *value;
    }
    Tensor::from_vec_col_major(vec![n, n], values)
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
    program: &tenferro_runtime::GraphProgram,
    x: &TracedTensor,
    input: &Tensor,
    expected_rank: usize,
    expected_values: &[f64],
) -> Result<(), tenferro_runtime::Error> {
    let outputs = executor.run_many_with_inputs(program, &[(x, input)])?;
    let reconstructed = &outputs[0];
    let singular_values = &outputs[1];

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
    let x = TracedTensor::input_concrete_shape(DType::F64, &[4, 4]);
    let (u, s, vt) = tenferro_linalg::traced_tensor::svd_with_eps(&x, 1.0e-12)?;

    let threshold = TracedTensor::from_vec_col_major(vec![], vec![0.5_f64]);
    let keep_count = s
        .compare(&threshold, CompareDir::Gt)
        .convert(DType::F64)
        .reduce_sum(&[0]);

    let u_truncated = u.dynamic_truncate(&keep_count, 1);
    let s_truncated = s.dynamic_truncate(&keep_count, 0);
    let vt_truncated = vt.dynamic_truncate(&keep_count, 0);

    let mut compiler = GraphCompiler::new();
    let reconstructed = tenferro_einsum::traced_tensor::einsum_with(
        &mut compiler,
        &[&u_truncated, &s_truncated, &vt_truncated],
        "ik,k,kj->ij",
        tenferro_einsum::EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )?;
    let program = compiler.compile_many(&[&reconstructed, &s_truncated])?;

    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_linalg::register_runtime)?;
    executor.register_extension(tenferro_einsum::register_runtime)?;

    let rank2 = diagonal_matrix(&[4.0, 3.0, 0.1, 0.01]);
    run_case(
        &mut executor,
        &program,
        &x,
        &rank2,
        2,
        &truncated_expected(&[4.0, 3.0, 0.1, 0.01], 0.5),
    )?;

    let rank3 = diagonal_matrix(&[4.0, 3.0, 2.0, 0.01]);
    run_case(
        &mut executor,
        &program,
        &x,
        &rank3,
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
