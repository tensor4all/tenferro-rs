# tenferro-rs

General-purpose dense tensor computation in Rust, inspired by PyTorch and JAX.

tenferro's complete default path is CPU: eager tensor operations with optional
scalar-loss reverse-mode autodiff, lazy traced execution, einsum, linear
algebra, and transform AD (VJP/JVP, with HVP via composition). CUDA execution
is available through the feature-gated CUDA backend with explicit CPU/GPU
transfers.

AD shape and dtype metadata is owned by the live eager/traced tensor handles
that need it. The backing lookup table is process-global, but entries are
scope-owned and are removed when the final graph or tensor handle using them is
dropped, so retrace-heavy long-running processes do not need a manual metadata
reset step.

## Crate Layout

Use the `tenferro` crate for application code. It is the public facade for
eager execution, traced execution, einsum, linear algebra, autodiff, and backend
selection. Internal workspace crates are documented through the API reference
for contributors who need implementation details.

## Development Model

tenferro-rs uses agentic AI development: a small human-maintainer team develops
the project with AI agents for implementation, documentation, review, issue
triage, and verification. Human maintainers own design decisions, review
outcomes, and merge decisions.

Issues are easiest to handle when they describe the user problem clearly. Bug
reports should include a minimal reproducer, expected behavior, actual
behavior, and the backend/device involved. Feature requests can stay informal:
describe what you are trying to do, what is hard today, and any examples or
related APIs that help explain the desired workflow.

## Documentation

**<https://tensor4all.org/tenferro-rs/>**

- [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/) — install and run the first checked CPU example
- [Guides](https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html) — tensor layers, execution models, tensor ops, einsum, linalg, autodiff, memory order, and CUDA
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
