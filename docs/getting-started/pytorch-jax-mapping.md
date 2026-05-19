# PyTorch and JAX Mapping

This page is for readers who already know either `torch` or `jax.numpy` and want to find the tenferro equivalent quickly.

## Concept mapping

| Concept | PyTorch | JAX | tenferro |
|---|---|---|---|
| Eager tensor | `numpy.ndarray` | — | `Tensor` + a backend |
| Tensor handle | `torch.Tensor` | `jax.Array` / `jnp.ndarray` | `TracedTensor` |
| Concrete result | `torch.Tensor` | `jax.Array` | `Tensor` returned by `.eval(&mut engine)` |
| Execution | Eager by default | Eager arrays, often staged with `jit` | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `.eval()`) |
| Eager gradients | `loss.backward()` | — | `EagerTensor::backward()` with accumulation |
| Transform AD | `torch.autograd.grad(...)` | `jax.grad`, `jax.vjp`, `jax.jvp`, `hvp` via composition | `loss.grad(&x)`, `.vjp()`, `.jvp()` |
| Device/runtime | Device is attached to tensors | Device is attached to arrays | Backend lives in direct tensor calls, `EagerContext`, or `Engine` |
| CUDA execution | `x.to("cuda")` | `jax.device_put(x)` | `tenferro::cuda::upload_tensor(...)` and `download_tensor(...)` |
| Matrix contraction | `torch.einsum` | `jnp.einsum` | `tenferro::traced_tensor::einsum` |

## Function mapping

| Task | PyTorch | JAX | tenferro (eager) | tenferro (lazy/AD) |
|---|---|---|---|---|
| Create tensor | `torch.tensor(data)` | `jnp.array(data)` | `Tensor::from_vec(shape, data)` | `TracedTensor::from_vec(shape, data)` |
| Matrix multiply | `torch.matmul(a, b)` | `jnp.matmul(a, b)` | `a.matmul(&b, &mut ctx)` | `tenferro::traced_tensor::matmul(&a, &b)` |
| Reshape | `x.reshape(shape)` | `jnp.reshape(x, shape)` | `x.reshape(&shape, &mut ctx)` | `x.reshape(&shape)` |
| Transpose | `x.transpose(0, 1)` | `jnp.transpose(x, axes)` | `x.transpose(&perm, &mut ctx)` | `x.transpose(&perm)` |
| Broadcast | `x.expand(...)` / implicit broadcast | implicit broadcast in many ops | backend-level op | `x.broadcast(&shape, &dims)` |
| Reduce sum | `x.sum(dim=...)` | `jnp.sum(x, axis=...)` | `x.reduce_sum(&axes, &mut ctx)` | `x.reduce_sum(&axes)` |
| Einsum | `torch.einsum(spec, ...)` | `jnp.einsum(spec, ...)` | `tenferro::tensor::einsum(&mut ctx, ...)` | `tenferro::traced_tensor::einsum(&mut engine, ...)` |
| SVD | `torch.linalg.svd(x)` | `jnp.linalg.svd(x)` | `x.svd(&mut ctx)` | `tenferro::traced_tensor::svd(&x)` |
| QR | `torch.linalg.qr(x)` | `jnp.linalg.qr(x)` | `x.qr(&mut ctx)` | `tenferro::traced_tensor::qr(&x)` |
| Cholesky | `torch.linalg.cholesky(x)` | `jnp.linalg.cholesky(x)` | `x.cholesky(&mut ctx)` | `tenferro::traced_tensor::cholesky(&x)` |
| Solve | `torch.linalg.solve(a, b)` | `jnp.linalg.solve(a, b)` | `a.solve(&b, &mut ctx)` | `tenferro::traced_tensor::solve(&a, &b)` |
| Scalar-loss backward | `loss.backward()` | — | `loss.backward()` on `EagerTensor` | — |
| Reverse-mode grad | `torch.autograd.grad(loss, x)` | `jax.grad(f)(x)` | — | `loss.grad(&x)` |
| VJP | `torch.autograd.grad(..., grad_outputs=...)` | `jax.vjp` | — | `y.vjp(&x, &cotangent)` |
| JVP | `torch.func.jvp` | `jax.jvp` | — | `y.jvp(&x, &tangent)` |

## Key differences

### Column-major storage

tenferro stores dense tensors in column-major order. If you write:

```rust
use tenferro::TracedTensor;
let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

then the columns are `[1, 2]`, `[3, 4]`, and `[5, 6]`.

Convert flat input data first when it is already in PyTorch, NumPy, or JAX
row-major order, then construct the tensor with `Tensor::from_vec`.

### Explicit CUDA transfer

tenferro follows the PyTorch convention that CPU and CUDA tensors do not move
between devices implicitly. Upload CPU tensors with
`tenferro::cuda::upload_tensor` before CUDA backend operations, and download
with `tenferro::cuda::download_tensor` before inspecting values on the host.

CUDA support targets NVIDIA CUDA through the CubeCL backend. See
[Devices and GPU](../guides/devices-and-gpu.md) for the current coverage and
setup commands.

### Lazy evaluation

PyTorch users usually expect every operation to execute immediately. JAX users
often switch between eager execution and `jit`. tenferro stays lazy until you
call `.eval(&mut engine)`.

### Autodiff split

Eager tenferro matches PyTorch's scalar-loss `loss.backward()` workflow with
accumulation semantics. Traced tenferro is the transform surface for
`torch.autograd.grad`, `jax.grad`, `jax.vjp`, `jax.jvp`, and higher-order
compositions such as HVPs.

### Engine ownership

In tenferro, the backend and reusable execution state live in `Engine`, not on each tensor. That means most user code follows this pattern:

1. Create `TracedTensor` values.
2. Build tensor expressions.
3. Reuse one `Engine` to evaluate them.
