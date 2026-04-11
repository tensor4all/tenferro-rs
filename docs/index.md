# tenferro

tenferro is a Rust tensor library for lazy numerical computation. It is aimed at users who already think in PyTorch or JAX terms but want a facade crate that fits naturally into Rust code. Use it when you want traced tensor operations, einsum-heavy workloads, and first-order autodiff without reaching into internal crates.

## PyTorch, JAX, and tenferro

| Topic | PyTorch | JAX | tenferro |
|---|---|---|---|
| Tensor creation | `torch.tensor(data)` | `jnp.array(data)` | `TracedTensor::new(shape, data)` |
| Execution model | Eager by default | Array ops are eager; `jit` stages work when you ask for it | Lazy until `.eval(&mut engine)` |
| Gradients | `loss.backward()` / `torch.autograd.grad(...)` | `jax.grad`, `jax.vjp`, `jax.jvp` | `loss.grad(&x)`, `y.vjp(&x, &cotangent)`, `y.jvp(&x, &tangent)` |
| Einsum | `torch.einsum(...)` | `jnp.einsum(...)` | `einsum(&mut engine, inputs, subscripts)` |
| Device model | Device on tensors and modules | Device on arrays, often with `jax.device_put(...)` | Backend owned by `Engine` |

## Navigate

- [Getting Started](getting-started/index.md)
- [Guides](guides/tensor-operations.md)
- [API](api/index.md)
- [Internals](internals/index.md)
