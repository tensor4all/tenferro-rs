# KdV PINN Sample

This tutorial is about a concrete question: can we train a neural network
\(u_\theta(x,t)\) to behave like the solution of a partial differential
equation, without giving the network a dense table of solution values?

The `kdv_pinn` workspace package answers that question for the
Korteweg-de Vries (KdV) equation. It builds a neural approximation to a known
single-soliton solution, constructs the PDE residual with tenferro traced
autodiff, differentiates one scalar objective with respect to the network
parameters, and updates those parameters with Adam.

Use this sample after the [Traced autodiff tutorial](traced-autodiff-jax-style.md)
when you want to see a larger traced-graph workflow with reusable compiled
programs, repeated `jvp` calls, scalar loss gradients, and a custom optimizer
loop.

## Problem Being Solved

The unknown is a scalar field \(u(x,t)\), where \(x\) is a spatial coordinate
and \(t\) is time. The Korteweg-de Vries equation used by the sample is

$$
u_t(x,t) + 6u(x,t)u_x(x,t) + u_{xxx}(x,t) = 0.
$$

Here \(u_t\) is the time derivative, \(u_x\) is the first spatial derivative,
and \(u_{xxx}\) is the third spatial derivative. The nonlinear term
\(6u u_x\) makes the wave steepen as it moves, while the dispersive term
\(u_{xxx}\) spreads different wavelengths at different speeds. Their balance
is what allows KdV soliton waves to keep their shape while traveling.

This sample solves the equation on the finite space-time domain

$$
(x,t) \in \Omega = [-5,5] \times [0,1].
$$

For training targets and evaluation, the sample uses the exact single-soliton
solution

$$
u_\star(x,t) = 2\,\operatorname{sech}^2(x - 4t),
\qquad
\operatorname{sech}(z)=\frac{1}{\cosh(z)}.
$$

The point is not to fit a dense labelled trajectory. The network sees the exact
solution only at the initial line and the two spatial boundaries. Inside the
domain, it is trained by asking the predicted field to satisfy the differential
equation.

## Model

The model is a fully connected neural network

$$
u_\theta(x,t) = f_\theta([x,t]) \in \mathbb{R}.
$$

The input is one coordinate pair \([x,t]\), and the output is one scalar value
that approximates \(u(x,t)\). In the code, the architecture is `[2, 64, 64, 1]`:
two inputs, two hidden layers with 64 units each, and one output. The hidden
layers use `tanh` activations.

The optimized variables are the network parameters

$$
\theta = \{W_\ell,b_\ell\}_{\ell=1}^{3},
$$

not the coordinates \(x\) and \(t\). In tenferro, each \(W_\ell\) and \(b_\ell\)
is represented by a `TracedTensor` placeholder while the graph is built. The
training loop binds those placeholders to concrete `Tensor` buffers and updates
those buffers after each gradient evaluation.

## Training Points

Each epoch samples three sets of points:

$$
\mathcal{C} = \{(x_i^c,t_i^c)\}_{i=1}^{N_\mathrm{col}},
\qquad N_\mathrm{col}=1024,
$$

for interior collocation points, sampled uniformly in \(\Omega\);

$$
\mathcal{I} = \{x_i^0\}_{i=1}^{N_\mathrm{ic}},
\qquad N_\mathrm{ic}=128,
$$

for initial-condition points at \(t=0\), with target
\(u_\star(x_i^0,0)=2\,\operatorname{sech}^2(x_i^0)\); and

$$
\mathcal{B} = \{(x_i^b,t_i^b)\}_{i=1}^{N_\mathrm{bc}},
\qquad N_\mathrm{bc}=128,
$$

for boundary points on \(x=-5\) and \(x=5\), with target
\(u_\star(x_i^b,t_i^b)\). Half of the boundary samples are placed at the left
boundary and half at the right boundary.

The collocation points are the PINN-specific part. They do not have labelled
\(u_\star\) values in the loss. Instead, they are where the network is penalized
for violating the KdV equation.

## Objective Function

The PDE residual of the neural network is

$$
r_\theta(x,t)
  = \partial_t u_\theta(x,t)
  + 6u_\theta(x,t)\partial_x u_\theta(x,t)
  + \partial_x^3 u_\theta(x,t).
$$

If \(u_\theta\) exactly solved the KdV equation, this residual would be zero at
every point in the domain. The sample minimizes the composite loss

$$
\mathcal{L}(\theta)
  = \lambda_\mathrm{pde}\mathcal{L}_\mathrm{pde}
  + \lambda_\mathrm{ic}\mathcal{L}_\mathrm{ic}
  + \lambda_\mathrm{bc}\mathcal{L}_\mathrm{bc},
$$

where the default weights are
\(\lambda_\mathrm{pde}=\lambda_\mathrm{ic}=\lambda_\mathrm{bc}=1\), and

$$
\mathcal{L}_\mathrm{pde}
  = \frac{1}{N_\mathrm{col}}\sum_{i=1}^{N_\mathrm{col}}
    r_\theta(x_i^c,t_i^c)^2,
$$

$$
\mathcal{L}_\mathrm{ic}
  = \frac{1}{N_\mathrm{ic}}\sum_{i=1}^{N_\mathrm{ic}}
    \left(u_\theta(x_i^0,0)-u_\star(x_i^0,0)\right)^2,
$$

and

$$
\mathcal{L}_\mathrm{bc}
  = \frac{1}{N_\mathrm{bc}}\sum_{i=1}^{N_\mathrm{bc}}
    \left(u_\theta(x_i^b,t_i^b)-u_\star(x_i^b,t_i^b)\right)^2.
$$

Training minimizes \(\mathcal{L}(\theta)\) with respect to the network
parameters \(\theta\). The \(x\) and \(t\) values are sampled inputs used to
evaluate the loss; they are not learned.

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
let nonlinear = u.mul(&u_x)?.scale_real(6.0)?;
let residual = u_t.add(&nonlinear)?.add(&u_xxx)?;
```

Each `jvp` differentiates with respect to a coordinate placeholder while using a
tensor of ones as the tangent direction. The first `jvp` with respect to `t`
gives \(u_t\). Repeated `jvp` calls with respect to `x` give \(u_x\), \(u_{xx}\),
and \(u_{xxx}\). This builds \(r_\theta(x,t)\) as another traced tensor in the
same graph.

That residual is squared and averaged before the PDE loss weight is applied,
matching the composite objective:

```text
loss = lambda_pde * mean(residual^2)
     + lambda_ic  * mean((u_ic - u_ic_true)^2)
     + lambda_bc  * mean((u_bc - u_bc_true)^2)
```

This is different from scaling the residual before squaring it, which would
apply the PDE weight twice.

After the scalar `total_loss` tensor is built, the sample requests one gradient
per parameter placeholder:

```rust
let param_grads: Vec<TracedTensor> = net
    .parameters()
    .iter()
    .map(|p| total_loss.grad(p))
    .collect::<tenferro_runtime::Result<Vec<_>>>()?;
```

Those gradients are \(\partial\mathcal{L}/\partial W_\ell\) and
\(\partial\mathcal{L}/\partial b_\ell\). The compiled gradient programs are
executed each epoch, and Adam applies the resulting concrete tensors to the
current parameter buffers.

## When To Use This Pattern

Use this structure when the model parameters, PDE inputs, and training data
change independently across many executions. `TracedTensor` lets the sample
compile the computational structure once, then bind concrete tensors for each
training batch and parameter update.

For shorter introductions, start with [Traced autodiff, JAX style](traced-autodiff-jax-style.md)
and [Eager autodiff, PyTorch style](eager-autodiff-pytorch-style.md).
