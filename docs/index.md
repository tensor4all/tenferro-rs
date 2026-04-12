# tenferro

tenferro-rs is a dense tensor computation workspace in Rust, inspired by
PyTorch and JAX. It provides:

- **Eager execution** — `Tensor` and `TypedTensor` with `CpuBackend` for
  direct computation without AD, like NumPy
- **Lazy traced execution** — `TracedTensor` with graph optimization and
  automatic differentiation (VJP/JVP/HVP)
- **Einsum** — with automatic contraction-tree planning
- **Linear algebra** — SVD, QR, Cholesky, eigh, solve

CPU execution is fully supported; GPU support is planned.

## PyTorch, JAX, and tenferro

| Topic | PyTorch | JAX | tenferro |
|---|---|---|---|
| Eager tensor | `torch.tensor(data)` | `jnp.array(data)` | `Tensor::new(shape, data)` |
| Traced tensor | — | staged via `jit` | `TracedTensor::new(shape, data)` |
| Execution | Eager by default | Eager; `jit` stages when asked | Eager (`Tensor`) or lazy (`TracedTensor` + `.eval()`) |
| Gradients | `loss.backward()` / `torch.autograd.grad` | `jax.grad`, `jax.vjp`, `jax.jvp` | `loss.grad(&x)`, `.vjp()`, `.jvp()` |
| Einsum | `torch.einsum(...)` | `jnp.einsum(...)` | `einsum(...)` / `eager_einsum(...)` |
| Device | Device on tensors | `jax.device_put(...)` | Backend owned by `Engine` or `CpuBackend` |

## Navigate

- [Getting Started](getting-started/index.md)
- [Guides](guides/tensor-operations.md)
- [API](api/index.md)
- [Internals](internals/index.md)
