# Choosing a Tensor Layer

This page is about choosing a tensor layer, not choosing one monolithic API.
tenferro has separate axes for data representation, execution timing, and
backend/device placement.

## Start Here

| If your project needs | Start with | Why |
| --- | --- | --- |
| No autodiff, scalar type known at compile time | `TypedTensor<T>` | Static dtype, direct owned data, typed linalg/einsum |
| No autodiff, dtype selected at runtime | `Tensor` | Dynamic dtype enum and broad concrete op surface |
| PyTorch-like scalar-loss `backward()` | `EagerTensor` + `EagerRuntime` | Immediate execution with gradient accumulation |
| JAX-like `grad`, `vjp`, `jvp`, HVP, graph reuse | `TracedTensor` + `GraphCompiler` + `GraphExecutor<B>` | Build graph, transform/compile it, run it repeatedly |
| CPU or CUDA execution | A backend: `CpuBackend` or `tenferro::cuda::CudaBackend` | Device is orthogonal to tensor layer |
| Operation outside the built-in surface | An extension crate | External ops can also register AD rules |

For most non-AD projects, `TypedTensor<T>` or `Tensor` should come before any
AD surface. Only move upward when the workflow needs gradient state or graph
transforms.

## Data Layer

`TypedTensor<T>` owns dense tensor data with a compile-time scalar type. It is
the simplest layer for fixed-dtype no-AD code and for applications that want
ordinary Rust type checking around scalar values.

`Tensor` owns the same kind of dense data, but wraps supported scalar types in
a runtime dtype enum. Use it when dtype must be selected dynamically, when you
want the broad concrete tensor operation surface, or when you need to pass CPU
or CUDA tensors through backend dispatch.

`EagerTensor` is not a replacement for `Tensor`; it is `Tensor` plus eager AD
state. Use it when the computation should run immediately and a scalar loss
will call `backward()`.

`TracedTensor` is a graph-building handle. It is the transform and compilation
surface, not the default concrete tensor type.

## Execution Model

| Model | Similar to | What happens on each op |
| --- | --- | --- |
| Direct tensor execution | NumPy-style explicit backend calls | The backend runs the op immediately and returns a concrete `Tensor` |
| Eager AD | PyTorch eager/autograd | The op runs immediately and records enough state for `backward()` |
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

## Operation Availability

The operation guides describe each operation family across tensor layers:

| Operation family | Concrete no-AD | Eager AD | Traced graph | CUDA |
| --- | --- | --- | --- | --- |
| Elementwise and shape ops | `Tensor` | `EagerTensor` | `TracedTensor` | Supported subset |
| Einsum | `typed_tensor::einsum`, `tensor::einsum` | Through eager execution | `traced_tensor::einsum` | Supported through backend coverage |
| Linear algebra | `Tensor` and selected `TypedTensor<T>` methods | Through eager execution | `traced_tensor` helpers | Supported subset |
| Transform AD | Not applicable | Not the transform surface | `grad`, `vjp`, `jvp`, HVP | Backend-dependent execution |
| Extension ops | Extension-provided eager hooks | If the extension provides eager hooks and AD | If the extension provides graph hooks and AD | Extension-dependent |

## Extension Story

Automatic differentiation is externally extensible. An extension crate can add
operations, eager/traced execution hooks, and AD rules without forcing the core
crate to grow application-specific APIs. [`tenferro-fft`](tenferro-fft.md) is
the example extension package: it adds Fourier transform operations and
registers AD rules for supported transforms.
