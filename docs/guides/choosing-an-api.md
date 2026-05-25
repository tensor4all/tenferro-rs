# Choosing a Tensor Layer

This page is about choosing a tensor layer, not choosing one monolithic API.
tenferro has separate axes for data representation, execution timing, and
backend/device placement.

## Start Here

![Decision tree for choosing a tensor model](../assets/tensor-layer-decision.svg)

For most non-AD projects, `TypedTensor<T>` or `Tensor` should come first. Move
to `EagerTensor` when you want immediate execution under an `EagerRuntime`;
make tensors tracked only when the workflow needs scalar-loss `backward()`.
Move to `TracedTensor` when the workflow needs graph transforms.

Quick reference:

| If your project needs | Start with |
| --- | --- |
| No autodiff, scalar type known at compile time | `TypedTensor<T>` |
| No autodiff, dtype selected at runtime | `Tensor` |
| Immediate forward execution in one runtime, optionally scalar-loss `backward()` | `EagerTensor` + `EagerRuntime` |
| JAX-like `grad`, `vjp`, `jvp`, HVP via composition, graph reuse | `TracedTensor` + `GraphCompiler` + `GraphExecutor<B>` |

## Data Layer

`TypedTensor<T>` owns dense tensor data with a compile-time scalar type. It is
the simplest layer for fixed-dtype no-AD code and for applications that want
ordinary Rust type checking around scalar values.

`Tensor` owns the same kind of dense data, but wraps supported scalar types in
a runtime dtype enum. Use it when dtype must be selected dynamically, when you
want the broad concrete tensor operation surface, or when you need to pass CPU
or CUDA tensors through backend dispatch.

`EagerTensor` is concrete eager execution. It wraps `Tensor` values in an
`EagerRuntime`, so each operation computes a concrete result immediately.
Untracked eager tensors are forward-only. Tracked eager tensors additionally
record reverse-mode state for scalar-loss `backward()`.

`TracedTensor` is a graph-building handle. It is the transform and compilation
surface, not the default concrete tensor type.

## Execution Model

| Model | Similar to | What happens on each op |
| --- | --- | --- |
| Direct tensor execution | NumPy-style explicit backend calls | The backend runs the op immediately and returns a concrete `Tensor` |
| Eager execution | PyTorch eager/autograd | The op runs immediately; tracked values record enough state for `backward()` |
| Traced execution | JAX tracing/jit/grad | The op records graph structure; compute runs after compile/execute |

See [Execution Models](execution-models.md) for the time-axis diagram,
including the difference between Eager CPU, Eager GPU, and Traced mode.

## Device And Backend

CPU and CUDA are backend choices. They do not decide whether your program is
typed, eager, or traced.

CUDA support is provided by the feature-gated CUDA backend for concrete,
eager, and traced workflows. CPU/GPU transfer is explicit:

- upload CPU tensors before CUDA backend operations,
- keep intermediate tensors on CUDA while doing CUDA work,
- download only when the host must inspect values,
- do not expect an unsupported CUDA operation to silently fall back to CPU.

The current CUDA operation and dtype table is in
[Devices and GPU](devices-and-gpu.md).

## Operation Entry Points

Choose the tensor layer first, then choose the operation family. CUDA is not a
separate operation entry point; it is a backend/device choice for supported
operations.

| Need | No-AD concrete path | Eager path | Traced path |
| --- | --- | --- | --- |
| Everyday tensor ops | `tenferro::tensor` functions; selected `tenferro::typed_tensor` wrappers | `tenferro::eager_tensor` functions | `tenferro::traced_tensor` functions |
| Einsum | Internal to `tenferro-einsum` runtime execution | `tenferro_einsum::eager_tensor::einsum` | `tenferro_einsum::traced_tensor::einsum` plus `register_runtime` |
| Tensordot sugar | Use `matmul` or `dot_general` directly | `tenferro_einsum::eager_tensor::tensordot` | `tenferro_einsum::traced_tensor::tensordot` |
| Linear algebra | `Tensor` methods; selected `TypedTensor<T>` methods | `tenferro_linalg::eager_tensor` helpers | `tenferro_linalg::traced_tensor` helpers |
| Automatic differentiation | Not applicable | Scalar-loss `backward()` on tracked values | `grad`, `vjp`, `jvp`, HVP via composition |
| External operations | Extension-defined concrete hooks | Extension-defined eager hooks and optional AD rules | Extension-defined graph hooks and optional AD rules |

Use CPU or CUDA with these paths according to backend coverage. CUDA tensors
must be moved explicitly with upload/download helpers, and unsupported CUDA
operations do not silently fall back to CPU.

## Extension Story

Automatic differentiation is externally extensible. An extension crate can add
operations, eager/traced execution hooks, and AD rules without forcing the core
crate to grow application-specific APIs. [FFT (extension)](tenferro-fft.md) is
the example extension package: it adds Fourier transform operations and
registers AD rules for supported transforms.
