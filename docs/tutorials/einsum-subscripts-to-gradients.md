# Einsum: Subscripts To Gradients

Use `tenferro-einsum` when a contraction is clearer as labeled axes than as a
chain of matrix multiplies. The extension crate owns both eager and traced
einsum APIs, and traced execution requires registering the einsum runtime on
the `GraphExecutor`.

The example below starts with eager `"ij,jk->ik"`, compares two contraction
planning choices for a three-operand contraction, then differentiates
`sum(einsum("ij,jk->ik"))` with respect to the left operand.

<!-- snippet-source: docs/tutorial-code/src/bin/einsum_subscripts_to_gradients.rs -->
```rust
use std::sync::Arc;

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_einsum::{EagerEinsumExt, EinsumOptimize, TraceContextEinsumExt};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TraceContext};

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

fn matrix_a() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
    )?)
}

fn matrix_b() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![3, 2],
        vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0],
    )?)
}

fn matrix_c() -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0_f64, 3.0, 2.0, 4.0],
    )?)
}

fn trace_and_run(
    inputs: &[Tensor],
    optimize: EinsumOptimize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut trace = TraceContext::new();
    let values = inputs
        .iter()
        .map(|tensor| {
            trace.input_with_default(
                ProgramInputSpec::new(tensor.dtype(), DimExpr::from_concrete(tensor.shape())),
                Arc::new(tensor.clone()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = trace.einsum_with(&values, "ij,jk,kl->il", optimize)?;
    let graph = trace.finish(&[output])?;
    let program = GraphCompiler::new().compile_traced_graph(&graph)?;
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_einsum::register_runtime)?;
    Ok(executor.run(&program)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a_tensor = matrix_a()?;
    let b_tensor = matrix_b()?;
    let c_tensor = matrix_c()?;
    let runtime = EagerRuntime::new();
    let a = runtime.variable_from(a_tensor.clone())?;
    let b = runtime.variable_from(b_tensor.clone())?;
    let product = [&a, &b].einsum("ij,jk->ik")?;

    assert_eq!(product.shape(), &[2, 2]);
    assert_close(
        product.materialized()?.as_slice::<f64>().unwrap(),
        &[58.0, 139.0, 64.0, 154.0],
    );

    let inputs = [a_tensor, b_tensor, c_tensor];
    let auto_value = trace_and_run(&inputs, EinsumOptimize::default())?;
    let left_to_right_value = trace_and_run(&inputs, EinsumOptimize::False)?;
    assert_close(
        auto_value.as_slice::<f64>().unwrap(),
        left_to_right_value.as_slice::<f64>().unwrap(),
    );
    assert_close(
        auto_value.as_slice::<f64>().unwrap(),
        &[250.0, 601.0, 372.0, 894.0],
    );

    let loss = product.reduce_sum(Some(&[0, 1]))?;
    let grad_a = runtime.grad(&loss, &a)?;
    let grad_value = grad_a.materialized()?;

    assert_eq!(grad_value.shape(), &[2, 3]);
    assert_close(
        grad_value.as_slice::<f64>().unwrap(),
        &[15.0, 15.0, 19.0, 19.0, 23.0, 23.0],
    );

    Ok(())
}
```
<!-- end-snippet-source -->

`EinsumOptimize::default()` chooses an automatic contraction order.
`EinsumOptimize::False` keeps the straightforward left-to-right order. Both
produce the same values; the choice changes cost and cache identity.

For more notation details, path controls, and cache behavior, see the
[einsum guide](../guides/einsum.md).
