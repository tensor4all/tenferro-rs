# PyTorch and JAX Mapping

This page is for readers who already know either `torch` or `jax.numpy` and want to find the tenferro equivalent quickly.

## Concept mapping

| Concept | PyTorch | JAX | tenferro |
|---|---|---|---|
| Typed concrete tensor | `torch.Tensor` with fixed dtype | `jax.Array` with fixed dtype | `TypedTensor<T>` |
| Dynamic concrete tensor | `torch.Tensor` | `jax.Array` / `jnp.ndarray` | `Tensor` + a backend |
| Graph-building tensor handle | `torch.Tensor` under compiled/tracing tools | traced `jax.Array` values | `TracedTensor` |
| Concrete result | `torch.Tensor` | `jax.Array` | `Tensor` returned by `GraphExecutor::run` |
| Execution | Eager by default | Eager arrays, often staged with `jit` | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `GraphCompiler` + `GraphExecutor`) |
| Eager forward and gradients | eager ops plus `loss.backward()` | — | `EagerTensor` forward ops, with `backward()` for tracked scalar losses |
| Transform AD | `torch.autograd.grad(...)` | `jax.grad`, `jax.vjp`, `jax.jvp`, `hvp` via composition | `loss.grad(&x)`, `.vjp()`, `.jvp()`; HVP via composition |
| Device/runtime | Device is attached to tensors | Device is attached to arrays | Backend lives in direct tensor calls, `EagerRuntime`, or `GraphExecutor` |
| CUDA execution | `x.to("cuda")` | `jax.device_put(x)` | `tenferro_gpu::cubecl::upload_tensor(...)` and `download_tensor(...)` |
| Matrix contraction | `torch.einsum` | `jnp.einsum` | `tenferro_einsum::traced_tensor::einsum` standard extension |

## Function mapping

| Task | PyTorch | JAX | tenferro (eager) | tenferro (lazy/AD) |
|---|---|---|---|---|
| Create typed tensor | `torch.tensor(data, dtype=...)` | `jnp.array(data, dtype=...)` | `TypedTensor::<T>::from_vec_col_major(shape, data)` | — |
| Create dynamic tensor | `torch.tensor(data)` | `jnp.array(data)` | `Tensor::from_vec_col_major(shape, data)` | `TracedTensor::from_vec_col_major(shape, data)` |
| Matrix multiply | `torch.matmul(a, b)` | `jnp.matmul(a, b)` | `tenferro_runtime::tensor::matmul(&a, &b, &mut ctx)` | `tenferro_runtime::traced_tensor::matmul(&a, &b)` |
| Reshape | `x.reshape(shape)` | `jnp.reshape(x, shape)` | `x.reshape(&shape, &mut ctx)` | `x.reshape(&shape)` |
| Transpose | `x.transpose(0, 1)` | `jnp.transpose(x, axes)` | `x.transpose(&perm, &mut ctx)` | `x.transpose(&perm)` |
| Broadcast | `x.expand(...)` / implicit broadcast | implicit broadcast in many ops | backend-level op | `x.broadcast(&shape, &dims)` |
| Reduce sum | `x.sum(dim=...)` | `jnp.sum(x, axis=...)` | `x.reduce_sum(&axes, &mut ctx)` | `x.reduce_sum(&axes)` |
| Einsum | `torch.einsum(spec, ...)` | `jnp.einsum(spec, ...)` | `tenferro_einsum::eager_tensor::einsum(...)` | `tenferro_einsum::traced_tensor::einsum(&mut compiler, ...)` plus `register_runtime` |
| SVD | `torch.linalg.svd(x)` | `jnp.linalg.svd(x)` | `x.svd(&mut ctx)` | `tenferro_linalg::traced_tensor::svd(&x)` |
| QR | `torch.linalg.qr(x)` | `jnp.linalg.qr(x)` | `x.qr(&mut ctx)` | `tenferro_linalg::traced_tensor::qr(&x)` |
| Cholesky | `torch.linalg.cholesky(x)` | `jnp.linalg.cholesky(x)` | `x.cholesky(&mut ctx)` | `tenferro_linalg::traced_tensor::cholesky(&x)` |
| Solve | `torch.linalg.solve(a, b)` | `jnp.linalg.solve(a, b)` | `a.solve(&b, &mut ctx)` | `tenferro_linalg::traced_tensor::solve(&a, &b)` |
| Scalar-loss backward | `loss.backward()` | — | `loss.backward()` on `EagerTensor` | — |
| Reverse-mode grad | `torch.autograd.grad(loss, x)` | `jax.grad(f)(x)` | — | `loss.grad(&x)` |
| VJP | `torch.autograd.grad(..., grad_outputs=...)` | `jax.vjp` | — | `y.vjp(&x, &cotangent)` |
| JVP | `torch.func.jvp` | `jax.jvp` | — | `y.jvp(&x, &tangent)` |

## Key differences

### Column-major storage

tenferro stores dense tensors in column-major order. If you write:

```rust
use tenferro_runtime::TracedTensor;
let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

then the columns are `[1, 2]`, `[3, 4]`, and `[5, 6]`.

Use `Tensor::from_vec_row_major` or `TracedTensor::from_vec_row_major` for flat
data copied from PyTorch, NumPy, or JAX row-major arrays. Use
`from_vec_col_major` only when the flat buffer is already in tenferro's
physical order.

### Explicit CUDA transfer

tenferro follows the PyTorch convention that CPU and CUDA tensors do not move
between devices implicitly. Upload CPU tensors with
`tenferro_gpu::cubecl::upload_tensor` before CUDA backend operations, and download
with `tenferro_gpu::cubecl::download_tensor` before inspecting values on the host.

For eager CUDA execution, operation calls submit work and return CUDA tensor
handles. The host synchronizes at download/read boundaries or at operations
that must inspect device-side status; there is no user-visible ready flag.

CUDA support targets NVIDIA CUDA. See
[Devices and GPU](../guides/devices-and-gpu.md) for the current coverage and
setup commands.

### Lazy traced execution

PyTorch users usually expect every operation to execute immediately. JAX users
often switch between eager execution and `jit`. tenferro's traced surface stays
lazy until you lower a `TracedTensor` graph with `GraphCompiler` and run the
resulting `GraphProgram` with `GraphExecutor`.

### Autodiff split

Eager tenferro matches PyTorch-style eager forward execution. When tensors are
tracked, it also matches the scalar-loss `loss.backward()` workflow with
accumulation semantics. Traced tenferro is the transform surface for
`torch.autograd.grad`, `jax.grad`, `jax.vjp`, `jax.jvp`, and higher-order
compositions such as HVPs.

### Compiler and executor ownership

In tenferro, graph lowering state and backend runtime state are separate. That
means most traced user code follows this pattern:

1. Create `TracedTensor` values.
2. Build tensor expressions.
3. Reuse one `GraphCompiler` for graph lowering and static planning caches.
4. Reuse one `GraphExecutor<B>` for backend execution and runtime caches.
