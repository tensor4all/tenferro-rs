# Traced Autodiff, JAX Style

Use `TracedTensor` when you want graph construction separated from execution.
This is the natural entry point for JAX-like `grad`, `vjp`, and `jvp`
workflows: build a graph, compile and run it, then derive graphs for gradients
or Jacobian-vector products.

The example below evaluates `sum(x * x)`, builds its gradient with respect to
`x`, and evaluates a directional derivative with `jvp`.

<!-- snippet-source: docs/tutorial-code/src/bin/traced_autodiff_jax_style.rs -->
```rust
use tenferro_ad::TracedTensorAdExt;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn run(tensor: &TracedTensor) -> Result<tenferro_runtime::Tensor, tenferro_runtime::Error> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(tensor)?;
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    let registration = tenferro_cpu::runtime_engine_registration(&backend).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tutorial_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    builder.register_engine(registration).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tutorial_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let runtime = builder.build().map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tutorial_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.len(), 1);
    Ok(outputs.remove(0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    let y = (&x * &x)?.reduce_sum(Some(&[0]))?;

    let y_value = run(&y)?;
    assert_eq!(y_value.shape(), &[] as &[usize]);
    assert_close(y_value.as_slice::<f64>().unwrap(), &[14.0]);

    let grad = y.grad(&x)?;
    let grad_value = run(&grad)?;
    assert_eq!(grad_value.shape(), &[3]);
    assert_close(grad_value.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let tangent = TracedTensor::from_vec_col_major(vec![3], vec![0.1_f64, 1.0, -2.0])?;
    let directional = y.jvp(&x, &tangent)?;
    let directional_value = run(&directional)?;
    assert_eq!(directional_value.shape(), &[] as &[usize]);
    assert_close(directional_value.as_slice::<f64>().unwrap(), &[-7.8]);

    Ok(())
}
```
<!-- end-snippet-source -->

For execution model details, see the [execution models guide](../guides/execution-models.md)
and the [autodiff guide](../guides/autodiff.md).
