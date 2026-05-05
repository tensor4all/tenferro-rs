# tenferro-rs

General-purpose dense tensor computation in Rust, inspired by PyTorch and JAX.

tenferro provides both eager tensor operations with scalar-loss reverse-mode
autodiff and lazy traced execution with einsum, linear algebra, and transform
AD (VJP/JVP/HVP) on CPU and CUDA GPU (via CubeCL + cuTENSOR + cuSOLVER).

AD shape and dtype metadata is owned by the live eager/traced tensor handles
that need it. The backing lookup table is process-global, but entries are
scope-owned and are removed when the final graph or tensor handle using them is
dropped, so retrace-heavy long-running processes do not need a manual metadata
reset step.

## Workspace Crates

| Crate | Role |
| --- | --- |
| `tenferro` | User-facing facade: eager execution and scalar-loss reverse-mode AD via `EagerTensor` / `EagerContext`, plus traced execution and transform AD via `TracedTensor`, `Engine`, and public einsum/linalg APIs |
| `tenferro-tensor` | Dense runtime tensors, backend traits, CPU backend, CubeCL GPU backend |
| `tenferro-einsum` | Subscripts, contraction trees, and fragment-building utilities |
| `tenferro-ops` | Graph op vocabulary (`StdTensorOp`), `ExtensionOp` trait and registry, and AD rule implementations |
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

The default CPU backend is `cpu-faer`. To use the LAPACK/BLAS CPU backend
instead, disable default features and enable `cpu-blas`:

```toml
[dependencies]
tenferro = { path = "../tenferro-rs/tenferro", default-features = false, features = ["cpu-blas"] }
```

Exactly one CPU backend must be enabled. Builds using `cpu-blas` must link a
BLAS/LAPACK provider, either from the system toolchain or with the
`src-openblas` feature.

See the [Getting Started guide](https://tensor4all.org/tenferro-rs/getting-started/) for code examples.
