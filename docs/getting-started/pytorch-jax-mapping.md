# PyTorch and JAX Mapping

This page is for readers who already know either `torch` or `jax.numpy` and want to find the tenferro equivalent quickly.

## Concept mapping

| Concept | PyTorch | JAX | tenferro |
|---|---|---|---|
| Tensor handle | `torch.Tensor` | `jax.Array` / `jnp.ndarray` | `TracedTensor` |
| Concrete result | `torch.Tensor` | `jax.Array` | `Tensor` returned by `.eval(&mut engine)` |
| Execution | Eager by default | Eager arrays, often staged with `jit` | Lazy until `eval` |
| Gradients | `loss.backward()`, `torch.autograd.grad` | `jax.grad`, `jax.vjp`, `jax.jvp` | `.grad`, `.vjp`, `.jvp` |
| Device/runtime | Device is attached to tensors | Device is attached to arrays | Backend lives inside `Engine` |
| Matrix contraction | `torch.einsum` | `jnp.einsum` | `tenferro::einsum::einsum` |

## Function mapping

| Task | PyTorch | JAX | tenferro |
|---|---|---|---|
| Create tensor | `torch.tensor(data)` | `jnp.array(data)` | `TracedTensor::new(shape, data)` |
| Matrix multiply | `torch.matmul(a, b)` | `jnp.matmul(a, b)` | `tenferro::matmul(&a, &b)` |
| Reshape | `x.reshape(shape)` | `jnp.reshape(x, shape)` | `x.reshape(&shape)` |
| Transpose | `x.transpose(0, 1)` | `jnp.transpose(x, axes)` | `x.transpose(&perm)` |
| Broadcast | `x.expand(...)` / implicit broadcast | implicit broadcast in many ops | `x.broadcast(&shape, &dims)` |
| Reduce sum | `x.sum(dim=...)` | `jnp.sum(x, axis=...)` | `x.reduce_sum(&axes)` |
| Einsum | `torch.einsum(spec, ...)` | `jnp.einsum(spec, ...)` | `einsum(&mut engine, inputs, spec)` |
| SVD | `torch.linalg.svd(x)` | `jnp.linalg.svd(x)` | `tenferro::svd(&x)` |
| QR | `torch.linalg.qr(x)` | `jnp.linalg.qr(x)` | `tenferro::qr(&x)` |
| Cholesky | `torch.linalg.cholesky(x)` | `jnp.linalg.cholesky(x)` | `tenferro::cholesky(&x)` |
| Solve | `torch.linalg.solve(a, b)` | `jnp.linalg.solve(a, b)` | `tenferro::solve(&a, &b)` |
| Reverse-mode gradient | `torch.autograd.grad(loss, x)` | `jax.grad(f)(x)` | `loss.grad(&x)` |

## Key differences

### Column-major storage

tenferro stores dense tensors in column-major order. If you write:

```rust
let a = TracedTensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

then the columns are `[1, 2]`, `[3, 4]`, and `[5, 6]`.

### Lazy evaluation

PyTorch users usually expect every operation to execute immediately. JAX users often switch between eager execution and `jit`. tenferro stays lazy until you call `.eval(&mut engine)`.

### Engine ownership

In tenferro, the backend and reusable execution state live in `Engine`, not on each tensor. That means most user code follows this pattern:

1. Create `TracedTensor` values.
2. Build tensor expressions.
3. Reuse one `Engine` to evaluate them.
