# KdV PINN on tenferro-rs (TracedTensor + AdContext)

**Date:** 2026-06-15
**Status:** Design approved (revised: switched to TracedTensor)
**Scope:** Independent Rust sample/application

---

## 1. Purpose

Implement a Physics-Informed Neural Network (PINN) for the Korteweg-de Vries (KdV) equation using `tenferro-rs` as the backend. The sample is an independent Rust binary, not a workspace crate, and uses the `TracedTensor` + `AdContext` API so that third-order spatial derivatives can be computed automatically.

### Target PDE

```
u_t + u * u_x + u_xxx = 0
```

where `x` is 1D space and `t` is time. The network learns a scalar field `u(x, t)`.

### Reference Solution

Use the single-soliton solution

```
u(x, t) = 2 * sech^2(x - 4t)
```

with the initial condition

```
u(x, 0) = 2 * sech^2(x)
```

on the domain `x ∈ [-5, 5]`, `t ∈ [0, 1]`.

---

## 2. Success Criteria

1. The total training loss decreases (or converges) over epochs.
2. After training, the predicted `u(x, t)` is close to the analytic solution (small L2 error).
3. The sample runs on CPU in a reasonable amount of time (minutes to tens of minutes).

---

## 3. Chosen Approach

**Approach A: `TracedTensor` + `AdContext` (JAX-style graph AD).**

`EagerTensor` does not expose a `create_graph` equivalent, so higher-order automatic differentiation through `EagerTensor::backward()` is not supported. To compute the KdV term `u_xxx` exactly, we build a `TracedTensor` graph and use `TracedTensorAdExt::grad` repeatedly. Network parameters and collocation points are `TracedTensor::input_concrete_shape` placeholders, so the same compiled `GraphProgram` can be re-evaluated each epoch with `GraphExecutor::run_with_inputs`.

### Rejected Alternatives

- **`EagerTensor` + `EagerRuntime`**: Familiar, but only first-order AD is available. Would require numerical differentiation for `u_xxx` and lose exact higher-order gradients.
- **Mixed Approach**: Adds complexity by switching between Eager and Traced tensors. Avoided for the first version.

---

## 4. Architecture

### Directory Layout

```
kdv_pinn/
├── Cargo.toml
└── src/
    ├── main.rs        # Entry point and training loop
    ├── network.rs     # MLP, Linear, activations (placeholder-based)
    ├── pde.rs         # KdV differential operators
    ├── loss.rs        # PDE, initial, and boundary losses
    ├── optimizer.rs   # SGD and optional Adam (in-place on `Tensor`)
    └── sampler.rs     # Collocation point generation
```

### Dependencies

- `tenferro-ad`
- `tenferro-cpu`
- `tenferro-runtime`
- `tenferro-tensor`
- `rand` (for sampling)
- optional `csv` / `serde` (for output)

---

## 5. Components

### 5.1 Network (`network.rs`)

- `Linear { weight: TracedTensor, bias: TracedTensor }`: both are placeholders created with `TracedTensor::input_concrete_shape`.
- `Mlp { layers: Vec<Linear> }`
- `forward(&self, input: &TracedTensor) -> TracedTensor`
- Activation: `tanh` (`tenferro_runtime::traced_tensor::tanh`).
- Initialization: Xavier/Glorot-like scaling on the underlying `Tensor` values that are bound at runtime.

### 5.2 Differentiation Helper (`pde.rs`)

- `grad(output: &TracedTensor, input: &TracedTensor) -> Result<TracedTensor>`
- `grad_n(output: &TracedTensor, input: &TracedTensor, order: usize) -> Result<TracedTensor>`

Uses `TracedTensorAdExt::grad`. Because `grad` returns a new `TracedTensor` whose graph retains the primal computation, it can be differentiated again. Thus:

```rust
let u_x = grad(&u, &x)?;
let u_xx = grad(&u_x, &x)?;
let u_xxx = grad(&u_xx, &x)?;
```

### 5.3 PDE Residual (`pde.rs`)

`kdv_residual(u, x, t)` computes:

```
r = u_t + u * u_x + u_xxx
```

where `u` is the network output, and `x`, `t` are input placeholders.

### 5.4 Loss (`loss.rs`)

- `pde_loss`: mean squared PDE residual over collocation points.
- `initial_loss`: mean squared error at `t = 0`.
- `boundary_loss`: mean squared error at `x = ±5`.
- `total_loss = pde_loss + λ_ic * initial_loss + λ_bc * boundary_loss`.

Mean is implemented as `reduce_sum(...) * (1.0 / n)`.

### 5.5 Optimizer (`optimizer.rs`)

- Start with `Sgd { lr: f64 }`.
- Optionally add `Adam { lr, beta1, beta2, eps, m, v }`.
- `step(&mut self, params: &mut [Tensor], grads: &[Tensor])` updates parameters in-place using `Tensor::as_slice_mut::<f64>()` and `Tensor::as_slice::<f64>()`.

### 5.6 Sampler (`sampler.rs`)

- `collocation(n)`: uniform or random `(x, t)` pairs in the domain.
- `initial(n)`: points `(x, 0)` with analytic target `u(x, 0)`.
- `boundary(n)`: points `(±5, t)` with analytic target `u(±5, t)`.

---

## 6. Training Loop

```rust
// Build placeholders once
let net = Mlp::new(&ctx, ...)?;
let (x_col, t_col) = make_placeholders(batch_size)?;
let (x_ic, t_ic, u_ic_true) = make_placeholders(batch_size)?;
let (x_bc, t_bc, u_bc_true) = make_placeholders(batch_size)?;

// Build loss graph
let total_loss = build_loss(&net, &x_col, &t_col, &x_ic, &t_ic, &u_ic_true,
                            &x_bc, &t_bc, &u_bc_true)?;

// Build gradient graphs for every parameter
let param_grads: Vec<TracedTensor> = net.parameters()
    .iter()
    .map(|p| total_loss.grad(p).unwrap())
    .collect();

// Compile programs once
let mut compiler = GraphCompiler::new();
let input_specs = net.input_specs().chain(sample_input_specs()).collect::<Vec<_>>();
let loss_program = compiler.compile_with_input_specs(&total_loss, &input_specs)?;
let grad_programs: Vec<GraphProgram> = param_grads
    .iter()
    .map(|g| compiler.compile_with_input_specs(g, &input_specs))
    .collect::<Result<Vec<_>>>()?;

let mut executor = GraphExecutor::new(CpuBackend::new());

for epoch in 0..epochs {
    // 1. Sample concrete values
    let bindings = sampler.bindings_for_epoch(..., &net, ...)?;

    // 2. Evaluate loss
    let loss_tensor = executor.run_with_inputs(&loss_program, &bindings)?;
    let loss_value = loss_tensor.as_slice::<f64>().unwrap()[0];

    // 3. Evaluate gradients
    let mut grads = Vec::with_capacity(grad_programs.len());
    for program in &grad_programs {
        grads.push(executor.run_with_inputs(program, &bindings)?);
    }

    // 4. Update parameters in-place on `Tensor`
    optimizer.step(net.parameters_mut(), &grads);

    // 5. Logging
    if epoch % log_every == 0 {
        println!("epoch {}: loss={:.6e}", epoch, loss_value);
    }
}
```

---

## 7. Differentiation Strategy

`TracedTensorAdExt::grad` computes the gradient of a scalar output with respect to a traced input. Because the returned tensor is itself traced, the operation can be repeated for higher-order derivatives:

```rust
let u_t = grad(&u, &t)?;    // first-order
let u_x = grad(&u, &x)?;    // first-order
let u_xx = grad(&u_x, &x)?; // second-order
let u_xxx = grad(&u_xx, &x)?; // third-order
```

This gives exact automatic derivatives of the network output with respect to the spatial and temporal inputs, which is exactly what the KdV residual requires.

---

## 8. Test and Validation Plan

### 8.1 Unit Tests

- `Linear::forward` shape and value correctness using placeholder parameters.
- `Mlp::forward` with `tanh` activation.
- Differentiation helper: `x^3` gives `6` for the third derivative; `sin(x)` gives `-cos(x)` for the second derivative.
- Loss functions return expected shapes and values.

### 8.2 Integration Tests

- Train a tiny network only on the initial condition; verify it reproduces `u(x, 0)`.
- Train the full KdV PINN and verify loss decreases.
- Compare predicted `u(x, t)` with `2 * sech^2(x - 4t)` at `t = 0.0, 0.5, 1.0`.

### 8.3 Milestones

| # | Milestone | Acceptance |
|---|-----------|------------|
| 1 | 3rd-order differentiation PoC | `u_xxx` computed correctly for simple functions via `TracedTensor::grad` |
| 2 | Placeholder network + compile/run | Network forward evaluates through `GraphExecutor::run_with_inputs` |
| 3 | Initial-condition-only PINN | Network learns `u(x, 0)` |
| 4 | Full KdV PINN | PDE residual loss drives training |
| 5 | Evaluation | CSV output and L2 error against analytic solution |

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `TracedTensor::grad` does not compose to 3rd order | High | Fall back to central finite differences for `u_xxx`; validate against analytic derivatives first |
| Compile time per graph is high | Medium | Compile once per program; use placeholder bindings at runtime |
| Training does not converge | Medium | Tune learning rate, loss weights, network width/depth; try Adam |
| Slow CPU training | Medium | Reduce batch/domain size; profile; consider GPU backend later |
| Boundary conditions are ill-posed | Low | Use Dirichlet with analytic values at `x = ±5` |

---

## 10. Non-Goals

- GPU support is out of scope for the first version; CPU only.
- Generalization to other PDEs is not required.
- Real-time visualization is not required; CSV output is sufficient.
- Mixed Eager/Traced execution is not required.
