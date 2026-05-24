# tenferro-rs

General-purpose dense tensor computation in Rust, inspired by PyTorch and JAX.

tenferro's complete default path is CPU: eager tensor operations with optional
scalar-loss reverse-mode autodiff, lazy traced execution, extension-based
operation families, and transform AD (VJP/JVP, with HVP via composition).
CUDA execution is available through the feature-gated CUDA backend with
explicit CPU/GPU transfers.

AD shape and dtype metadata is owned by the live eager/traced tensor handles
that need it. The backing lookup table is process-global, but entries are
scope-owned and are removed when the final graph or tensor handle using them is
dropped, so retrace-heavy long-running processes do not need a manual metadata
reset step.

## Crate Layout

Use the `tenferro` crate for application code. It is the public facade for
eager execution, traced execution, autodiff, extension registration, and backend
selection.

Operation families such as einsum, linear algebra, and FFT are standard
extensions, not `tenferro` APIs. The `tenferro` crate must not depend on
`tenferro-einsum`, `tenferro-linalg`, or `tenferro-fft`, and it must not expose
paths such as `tenferro::einsum`, `tenferro::linalg`, or `tenferro::fft`.
Applications import and register the operation crates they use explicitly.

Core runtime infrastructure is split below the facade:

- `tenferro-tensor` owns dense tensors and backend kernels.
- `tenferro-ops` owns the graph operation vocabulary and core AD rules.
- `tenferro-runtime` owns extension runtime registration and extension cache
  storage.
- `tenferro` owns the public tensor facade, graph construction, eager AD, and
  reexports selected runtime types for application ergonomics.

Internal workspace crates are documented through the API reference for
contributors who need implementation details.

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

To use standard extensions, depend on the extension crate and register its
runtime on the executor:

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_einsum::einsum;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);

let mut compiler = GraphCompiler::new();
let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_einsum::register_runtime).unwrap();
let out = executor.run(&program).unwrap();
assert_eq!(out.shape(), &[2, 2]);
```

AD is enabled by default. For primal-only builds, disable default features and
enable a CPU backend explicitly:

```toml
[dependencies]
tenferro = { path = "../tenferro-rs/tenferro", default-features = false, features = ["cpu-faer"] }
tenferro-einsum = { path = "../tenferro-rs/tenferro-einsum", default-features = false, features = ["cpu-faer"] }
```
