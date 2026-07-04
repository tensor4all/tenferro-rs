# PyTorch and JAX Mapping

This page is a translation guide for readers who already know `torch` or `jax.numpy`: find the tenferro equivalent quickly, then read [Core Concepts](./core-concepts.md) for tenferro's own mental model.

## Concept mapping

| Concept | PyTorch | JAX | tenferro |
|---|---|---|---|
| Typed concrete tensor | `torch.Tensor` with fixed dtype | `jax.Array` with fixed dtype | `TypedTensor<T>` |
| Dynamic concrete tensor | `torch.Tensor` | `jax.Array` / `jnp.ndarray` | `Tensor` + a backend |
| Graph-building tensor handle | `torch.Tensor` under compiled/tracing tools | traced `jax.Array` values | `TracedTensor` |
| Concrete result | `torch.Tensor` | `jax.Array` | `Tensor` returned by `GraphExecutor::run` |
| Execution | Eager by default | Eager arrays, often staged with `jit` | Eager (`Tensor` / `EagerTensor`) or lazy traced (`TracedTensor` + `GraphCompiler` + `GraphExecutor`) |
| Eager forward and gradients | eager ops plus `loss.backward()` | — | `EagerTensor` forward ops, with `backward()` for tracked scalar losses |
| Transform AD | `torch.autograd.grad(...)` | `jax.grad`, `jax.vjp`, `jax.jvp`, `hvp` via composition | `loss.grad(&x)`, `.vjp()?`, `.jvp()?`; HVP via composition |
| Device/runtime | Device is attached to tensors | Device is attached to arrays | Backend lives in direct tensor calls, `EagerRuntime`, or `GraphExecutor` |
| CUDA execution | `x.to("cuda")` | `jax.device_put(x)` | `tenferro_gpu::upload_tensor(...)` and `download_tensor(...)` |
| Matrix contraction | `torch.einsum` | `jnp.einsum` | `compiler.einsum(...)` via `GraphCompilerEinsumExt` |

## Function mapping

| Task | PyTorch | JAX | tenferro (eager) | tenferro (lazy/AD) |
|---|---|---|---|---|
| Create typed tensor | `torch.tensor(data, dtype=...)` | `jnp.array(data, dtype=...)` | `TypedTensor::<T>::from_vec_col_major(shape, data)` | — |
| Create dynamic tensor | `torch.tensor(data)` | `jnp.array(data)` | `Tensor::from_vec_col_major(shape, data)` | `TracedTensor::from_vec_col_major(shape, data)` |
| Matrix multiply | `torch.matmul(a, b)` | `jnp.matmul(a, b)` | `a.matmul(&b, &mut ctx)` via `TensorOpsExt` | `a.matmul(&b)` |
| Reshape | `x.reshape(shape)` | `jnp.reshape(x, shape)` | `x.reshape(&shape, &mut ctx)` via `TensorOpsExt` | `x.reshape(&shape)?` |
| Transpose | `x.transpose(0, 1)` | `jnp.transpose(x, axes)` | `x.transpose(&perm, &mut ctx)` via `TensorOpsExt` | `x.transpose(&perm)` |
| Broadcast | `x.expand(...)` / implicit broadcast | implicit broadcast in many ops | backend-level op | `x.broadcast_in_dim(&shape, &dims)` |
| Reduce sum | `x.sum(dim=...)` | `jnp.sum(x, axis=...)` | `x.reduce_sum(&axes, &mut ctx)` via `TensorOpsExt` | `x.reduce_sum(&axes)` |
| Einsum | `torch.einsum(spec, ...)` | `jnp.einsum(spec, ...)` | `[&a, &b].einsum(...)` via `EagerEinsumExt` | `compiler.einsum(...)` via `GraphCompilerEinsumExt` plus `register_runtime` |
| SVD | `torch.linalg.svd(x)` | `jnp.linalg.svd(x)` | `tenferro_linalg::LinalgBackend::svd(&mut ctx, &x)?` | `x.svd()?` via `TracedTensorLinalgExt` |
| QR | `torch.linalg.qr(x)` | `jnp.linalg.qr(x)` | `tenferro_linalg::LinalgBackend::qr(&mut ctx, &x)?` | `x.qr()?` via `TracedTensorLinalgExt` |
| Cholesky | `torch.linalg.cholesky(x)` | `jnp.linalg.cholesky(x)` | `tenferro_linalg::LinalgBackend::cholesky(&mut ctx, &x)?` | `x.cholesky()?` via `TracedTensorLinalgExt` |
| Solve | `torch.linalg.solve(a, b)` | `jnp.linalg.solve(a, b)` | `tenferro_linalg::LinalgBackend::solve(&mut ctx, &a, &b)?` | `a.solve(&b)?` via `TracedTensorLinalgExt` |
| Scalar-loss backward | `loss.backward()` | — | `loss.backward()` on `EagerTensor` | — |
| Reverse-mode grad | `torch.autograd.grad(loss, x)` | `jax.grad(f)(x)` | — | `loss.grad(&x)` |
| VJP | `torch.autograd.grad(..., grad_outputs=...)` | `jax.vjp` | — | `y.vjp(&x, &cotangent)?` |
| JVP | `torch.func.jvp` | `jax.jvp` | — | `y.jvp(&x, &tangent)?` |

## Key differences

### Column-major storage

tenferro stores dense tensors in column-major order. If you write:

```rust
use tenferro_runtime::TracedTensor;
let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

then the columns are `[1, 2]`, `[3, 4]`, and `[5, 6]`.

Use `from_vec_col_major` only with buffers already in tenferro's physical
column-major order. Flat buffers copied from PyTorch, NumPy, or JAX row-major
arrays must be reordered explicitly before tensor construction.

### Explicit CUDA transfer

tenferro follows the PyTorch convention that CPU and CUDA tensors do not move
between devices implicitly. Upload CPU tensors with
`tenferro_gpu::upload_tensor` before CUDA backend operations, and download
with `tenferro_gpu::download_tensor` before inspecting values on the host.

For eager CUDA execution, operation calls submit work and return CUDA tensor
handles. The host synchronizes at download/read boundaries or at operations
that must inspect device-side status; there is no user-visible ready flag.

CUDA support targets NVIDIA CUDA. See
[Devices and GPU](../guides/devices-and-gpu.md) for the current coverage and
setup commands.

### Dynamic shapes and static-shape specialization

JAX/XLA specializes compiled programs to known shapes. That is a strong fit for
large static-shape workloads where compiler optimization dominates. tenferro's
native traced runtime is built for the complementary case: programs whose
output sizes may depend on runtime values, such as thresholded or truncated
linear algebra.

When shapes are static, the proposed optional XLA backend
([#984](https://github.com/tensor4all/tenferro-rs/issues/984)) may eventually
route traced programs to XLA. When ranks or extents are value-dependent,
tenferro keeps dynamic shape metadata in the traced program and resolves the
concrete sizes during execution. See
[Dynamic and Symbolic Shape Metadata](../design/dynamic-symbolic-shapes.md).

### Lazy traced execution

PyTorch users usually expect every operation to execute immediately. JAX users
often switch between eager execution and `jit`. tenferro's traced API stays
lazy until you lower a `TracedTensor` graph with `GraphCompiler` and run the
resulting `GraphProgram` with `GraphExecutor`.

### Autodiff split

Eager tenferro matches PyTorch-style eager forward execution. When tensors are
tracked, it also matches the scalar loss `loss.backward()` workflow with
accumulation semantics. Traced tenferro is the API for `torch.autograd.grad`,
`jax.grad`, `jax.vjp`, `jax.jvp`, and higher-order compositions such as HVPs.

For complex reverse-mode AD, tenferro uses a Hermitian real-inner-product
cotangent representation. When comparing scalar `grad` values to JAX, the
JAX-like value is `conj(tenferro_grad)` for the same scalar seed-`1`
calculation. See [Complex Autodiff](../guides/complex-ad.md) for the full
convention and VJP seed comparison.

### Compiler and executor ownership

In tenferro, graph lowering state and backend runtime state are separate. That
means most traced user code follows this pattern:

1. Create `TracedTensor` values.
2. Build tensor expressions.
3. Reuse one `GraphCompiler` for graph lowering and static planning caches.
4. Reuse one `GraphExecutor<B>` for backend execution and runtime caches.
