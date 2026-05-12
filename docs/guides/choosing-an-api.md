# Choosing an API

Use the simplest tensor layer that matches the workflow.

| Need | Use |
| --- | --- |
| Direct concrete computation | `Tensor` + `CpuBackend` |
| Compile-time scalar type while still owning dense data | `TypedTensor<T>` |
| PyTorch-style scalar-loss `backward()` | `EagerTensor` + `EagerContext` |
| `grad`, `vjp`, `jvp`, HVP, graph optimization | `TracedTensor` + `Engine` |

## Rule of Thumb

Start with `Tensor` for concrete CPU work. Move to `EagerTensor` when you need
gradient accumulation, and move to `TracedTensor` when you need transform AD or
graph reuse.

`Tensor` and `TypedTensor<T>` are concrete data containers. `EagerTensor` keeps
PyTorch-style gradient state for immediate workflows. `TracedTensor` builds a
lazy expression graph that an `Engine` can evaluate and reuse.
