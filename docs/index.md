# tenferro

tenferro-rs is a dense tensor computation workspace in Rust, inspired by
PyTorch and JAX. It provides:

- **Eager execution** — `Tensor`, `TypedTensor`, `EagerTensor`, and
  `EagerContext` with `CpuBackend` for immediate computation and
  scalar-loss reverse-mode via `backward()`
- **Lazy traced execution** — `TracedTensor` with graph optimization and
  transform-oriented automatic differentiation (`grad`, `vjp`, `jvp`, HVP)
- **Einsum** — with automatic contraction-tree planning
- **Linear algebra** — SVD, QR, Cholesky, eigh, LU, solve

CPU execution is fully supported; GPU support is planned.

## PyTorch, JAX, and tenferro

| Topic | PyTorch | JAX | tenferro |
|---|---|---|---|
| Eager tensor | `torch.tensor(data)` | `jnp.array(data)` | `Tensor::from_vec(shape, data)` |
| Traced tensor | — | staged via `jit` | `TracedTensor::from_vec(shape, data)` |
| Execution | Eager by default | Eager; `jit` stages when asked | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `.eval()`) |
| Eager gradients | `loss.backward()` | — | `EagerTensor::backward()` with accumulation |
| Transform AD | `torch.autograd.grad(...)` | `jax.grad`, `jax.vjp`, `jax.jvp`, `hvp` via composition | `loss.grad(&x)`, `.vjp()`, `.jvp()` |
| Einsum | `torch.einsum(...)` | `jnp.einsum(...)` | `einsum(...)` / `eager_einsum(...)` |
| Device | Device on tensors | `jax.device_put(...)` | Backend owned by `Engine` or `CpuBackend` |

## Navigate

- [Getting Started](getting-started/index.md)
- [Guides](guides/tensor-operations.md)
- [API](api/index.md)
- [Internals](internals/index.md)
