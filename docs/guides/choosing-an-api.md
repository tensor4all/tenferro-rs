# Choosing an API

Use the simplest tensor layer that matches the workflow.

| Need | Use |
| --- | --- |
| Direct concrete computation on a selected backend | `Tensor` + a `TensorBackend` |
| Compile-time scalar type while still owning dense data | `TypedTensor<T>` |
| PyTorch-style scalar-loss `backward()` | `EagerTensor<B>` + `EagerContext<B>` |
| `grad`, `vjp`, `jvp`, HVP, graph optimization | `TracedTensor` + `Engine<B>` |

## Rule of Thumb

Start with `Tensor` for direct concrete work. Move to `EagerTensor` when you need
gradient accumulation, and move to `TracedTensor` when you need transform AD or
graph reuse.

`Tensor` and `TypedTensor<T>` are concrete data containers. They represent CPU
data by default, and under the `cuda` feature they can also represent
explicitly uploaded CUDA-resident data. `EagerTensor<B>` keeps PyTorch-style
gradient state for immediate workflows on a backend `B`. `TracedTensor` builds
a lazy expression graph that an `Engine<B>` can evaluate and reuse.

CUDA support is provided by the feature-gated CubeCL backend for concrete,
eager, and traced workflows. GPU tensors use explicit upload/download at
backend boundaries; tenferro does not silently fall back to CPU when a GPU
operation or dtype is unavailable.
