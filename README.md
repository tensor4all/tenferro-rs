# tenferro-rs

General-purpose dense tensor computation in Rust, inspired by PyTorch and JAX.

tenferro provides lazy traced tensors with einsum, linear algebra, and
first-order autodiff on CPU. GPU support is planned.

## Workspace Crates

| Crate | Role |
| --- | --- |
| `tenferro` | Traced frontend: `Engine`, `TracedTensor`, public einsum and linalg APIs, VJP/JVP |
| `tenferro-tensor` | Dense runtime tensors, backend traits, CPU backend (GPU planned) |
| `tenferro-einsum` | Subscripts, contraction trees, and fragment-building utilities |
| `tenferro-ops` | Graph op vocabulary (`StdTensorOp`, `SemiringOp`) and AD rule implementations |
| `tenferro-algebra` | Semiring/algebra traits |
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
