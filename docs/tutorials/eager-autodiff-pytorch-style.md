# Eager Autodiff, PyTorch Style

Use `EagerTensor` when you want operations to execute immediately and want a
PyTorch-like `backward()` workflow on scalar losses. Tracked tensors created from an
`EagerRuntime` accumulate gradients until you clear them.

The example below computes `loss = sum(x * x)`, runs `backward()`, and checks
that the gradient is `2 * x`.

<!-- snippet-source: docs/tutorial-code/src/bin/eager_autodiff_pytorch_style.rs -->
```rust
use tenferro_ad::{EagerRuntime, Tensor};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = EagerRuntime::new();
    let x = runtime.variable_from(Tensor::from_vec_row_major(vec![3], vec![1.0_f64, 2.0, 3.0]));

    let prediction = &x * &x;
    let loss = prediction.reduce_sum(&[0])?;

    assert_eq!(loss.data().shape(), &[]);
    assert_close(loss.data().as_slice::<f64>().unwrap(), &[14.0]);

    loss.backward()?;

    let grad = x
        .grad()
        .expect("tracked variable should receive a gradient");
    assert_eq!(grad.shape(), &[3]);
    assert_close(grad.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    x.clear_grad();
    assert!(x.grad().is_none());

    Ok(())
}
```
<!-- end-snippet-source -->

For the broader eager API, see the [eager operations guide](../guides/eager-operations.md).
