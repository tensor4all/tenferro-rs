# KdV PINN Sample

The `kdv_pinn` workspace package is an end-to-end physics-informed neural
network sample for the Korteweg-de Vries equation. It trains a small MLP to
approximate the single-soliton solution on `x in [-5, 5]` and `t in [0, 1]`
while minimizing three terms:

- the interior PDE residual `u_t + 6 u u_x + u_xxx`,
- the initial condition at `t = 0`,
- exact Dirichlet boundary data at `x = -5` and `x = 5`.

Use this sample after the [Traced autodiff tutorial](traced-autodiff-jax-style.md)
when you want to see a larger traced-graph workflow with reusable compiled
programs, repeated `jvp` calls, scalar loss gradients, and a custom optimizer
loop.

## Run The Sample

From the repository root:

```bash
cargo run -p kdv_pinn --release
```

The run prints training loss every 50 epochs, then reports the final loss and
the relative L2 error at `t = 0.5`. The default configuration trains for 3000
epochs on CPU, so it is intentionally a longer-running sample than the small
`docs/tutorial-code` binaries.

Optional plot outputs are available:

```bash
cargo run -p kdv_pinn --release -- --gif kdv_pinn.gif --loss-png loss.png
```

`--loss-png` writes a log-scale training-loss plot. `--gif` writes an animated
comparison between the analytic solution and the trained model prediction.

For a fast correctness check without training the full model:

```bash
cargo test -p kdv_pinn
```

## How It Is Structured

The sample keeps each PINN concern in a small module:

| File | Role |
| --- | --- |
| `kdv_pinn/src/network.rs` | Builds the MLP from `TracedTensor` parameter placeholders. |
| `kdv_pinn/src/pde.rs` | Builds the KdV residual with chained `jvp` calls for `u_t`, `u_x`, `u_xx`, and `u_xxx`. |
| `kdv_pinn/src/loss.rs` | Combines PDE, initial-condition, and boundary losses with scalar weights. |
| `kdv_pinn/src/sampler.rs` | Samples collocation, initial-condition, and boundary points. |
| `kdv_pinn/src/optimizer.rs` | Applies Adam updates to the concrete parameter tensors. |
| `kdv_pinn/src/plot.rs` | Writes optional loss-curve PNG and solution-comparison GIF outputs. |

The graph-building phase creates placeholders for network parameters and
training batches, then compiles a scalar loss program plus one gradient program
per parameter. The training loop reuses those compiled programs with fresh
sampled tensors at every epoch.

## Traced Autodiff Pattern

The PDE residual is the key traced-AD pattern:

```rust
let ones_x = ones_like(x)?;
let ones_t = ones_like(t)?;
let u_t = u.jvp(t, &ones_t)?;
let u_x = u.jvp(x, &ones_x)?;
let u_xx = u_x.jvp(x, &ones_x)?;
let u_xxx = u_xx.jvp(x, &ones_x)?;
let nonlinear = u.mul(&u_x)?.scale_real(6.0);
let residual = u_t.add(&nonlinear)?.add(&u_xxx)?;
```

That residual is squared and averaged before the PDE loss weight is applied,
matching the composite objective:

```text
loss = lambda_pde * mean(residual^2)
     + lambda_ic  * mean((u_ic - u_ic_true)^2)
     + lambda_bc  * mean((u_bc - u_bc_true)^2)
```

This is different from scaling the residual before squaring it, which would
apply the PDE weight twice.

## When To Use This Pattern

Use this structure when the model parameters, PDE inputs, and training data
change independently across many executions. `TracedTensor` lets the sample
compile the computational structure once, then bind concrete tensors for each
training batch and parameter update.

For shorter introductions, start with [Traced autodiff, JAX style](traced-autodiff-jax-style.md)
and [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md).
