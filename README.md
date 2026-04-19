# tenferro-rs

General-purpose dense tensor computation in Rust, inspired by PyTorch and JAX.

tenferro provides both eager tensor operations with scalar-loss reverse-mode
autodiff and lazy traced execution with einsum, linear algebra, and transform
AD (VJP/JVP/HVP) on CPU and CUDA GPU (via CubeCL + cuTENSOR + cuSOLVER).

## Workspace Crates

| Crate | Role |
| --- | --- |
| `tenferro` | User-facing facade: eager execution and scalar-loss reverse-mode AD via `EagerTensor` / `EagerContext`, plus traced execution and transform AD via `TracedTensor`, `Engine`, and public einsum/linalg APIs |
| `tenferro-tensor` | Dense runtime tensors, backend traits, CPU backend, CubeCL GPU backend |
| `tenferro-einsum` | Subscripts, contraction trees, and fragment-building utilities |
| `tenferro-ops` | Graph op vocabulary (`StdTensorOp`, `SemiringOp`) and AD rule implementations |
| `tenferro-device` | Shared device and error infrastructure |

## Documentation

**<https://tensor4all.org/tenferro-rs/>**

- [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/) — installation, hello-world examples
- [Guides](https://tensor4all.org/tenferro-rs/guides/tensor-operations.html) — tensor ops, einsum, linalg, autodiff, eager mode
- [API Reference](https://tensor4all.org/tenferro-rs/api/) — rustdoc links for every crate
- [Internals](https://tensor4all.org/tenferro-rs/internals/) — architecture, specification, contributor pointers

## Quick Start

```toml
[dependencies]
tenferro = { path = "../tenferro-rs/tenferro" }
```

See the [Getting Started guide](https://tensor4all.org/tenferro-rs/getting-started/) for code examples.
