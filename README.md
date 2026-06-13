# tenferro-rs

tenferro-rs is a modular Rust tensor stack for scientific computing and
general-purpose tensor workflows.

## Why tenferro-rs Exists

`tenferro` means tensor computation with an iron/Rust flavor: `tensor` +
`ferro`.

The project aims to provide a Rust-native tensor stack for typed tensor
computation, immediate execution with optional `backward()`, traced graph
execution, automatic differentiation, linear algebra, einsum, FFT, and explicit
CPU/CUDA backend control.

tenferro-rs is influenced by JAX and PyTorch in its split between immediate
execution, traced graphs, and AD. It is also influenced by the Julia numerical
computing ecosystem, where operation semantics and AD rules can live outside a
single all-in-one tensor type.

tenferro-rs is Rust-native, but it should fit established numerical computing
conventions rather than replacing them with project-specific rules. It uses
Rust's type system, ownership model, and crate boundaries where those choices
help.

## Design Principles

- Keep tensor types and operation crates modular, rather than building one
  all-in-one tensor type.
- Let users choose the lowest API that solves their problem. Autodiff, traced
  graphs, CUDA, linalg, einsum, and FFT are opt-in capabilities.
- Stay aligned with established numerical computing conventions where they
  fit: Rust's crate ecosystem, column-major conventions from Fortran, Julia,
  MATLAB, and LAPACK-oriented workflows, and Rust numerics crates such as
  `faer`, `num-traits`, and `num-complex`.
- Make dependencies, feature flags, backends, and devices explicit. tenferro
  does not silently move tensor data between CPU and GPU.
- Support both Rust-typed tensors and tensors whose dtype is selected at
  runtime.
- Treat extension operations and AD rules as part of the design, so external
  crates can add operations without growing the core tensor crates.
- Validate numerical behavior with tests, oracle data, invariants, residual
  checks, provenance checks, and reproducible benchmarks.

## When tenferro Is a Good Fit

tenferro-rs is designed for workloads where tensor shapes are not always fixed
before execution. If your computation involves runtime-dependent ranks,
threshold-based filtering, data-dependent iteration counts, or shape-parametric
models, tenferro's traced execution can reuse a compiled program while resolving
the concrete sizes at runtime.

| tenferro is a good fit when | XLA/JAX may be a better fit when |
| --- | --- |
| Shapes depend on runtime values | All shapes are static and known before compilation |
| Small-to-medium batched linear algebra with AD | Large static-shape matmul or contraction throughput dominates |
| You need `grad`, `vjp`, `jvp`, or HVP workflows in Rust | You can use Python as the host language |
| You are building a Rust-native tensor application | You need the mature compiler and ecosystem around JAX today |
| You need extension points for custom algebra or operation families | You only need standard tensor operations |

The planned optional XLA backend in
[#984](https://github.com/tensor4all/tenferro-rs/issues/984) is intended for
static-shape traced graphs. Dynamic-shape graphs remain the native runtime's
core use case.

![tenferro-rs architecture overview](docs/assets/tenferro-architecture.svg)

## How tenferro-rs Relates To Other Tools

tenferro-rs is **not** an attempt to replace JAX or PyTorch, and it does not
require Python. The goal is a pure-Rust software stack that offers comparable
capabilities — typed tensors, eager execution with `backward()`, traced graphs,
`grad`/`vjp`/`jvp` autodiff, linear algebra, einsum, and FFT — to applications
that want to stay in the Rust ecosystem. The comparison tables above describe
*workload fit*, not a rivalry: if Python is your host language, JAX and PyTorch
remain excellent choices.

- **Versus ML-focused Rust frameworks (`candle`, `burn`).** Those target deep
  learning. tenferro-rs targets scientific and numerical computing, where
  linear algebra, einsum, FFT, and autodiff *through* those operations are the
  primary workload. They are complementary, not competitors: tenferro-rs builds
  on [CubeCL](https://github.com/tracel-ai/cubecl) for GPU kernels — the same
  infrastructure `burn` uses — and aims to contribute upstream rather than
  reinvent that layer.
- **Versus binding-based stacks (`tch-rs`).** A key tenferro-rs advantage is
  that operations and their AD rules are extensible from *outside* the core
  tensor crates: an external crate can add a new operation family, register its
  `linearize`/`transpose_rule`, and have it participate in the same eager and
  traced AD workflows as built-in ops (see
  [`docs/guides/custom-operations.md`](docs/guides/custom-operations.md) and the
  worked `ext/tropical` semiring extension). Binding-based stacks like `tch-rs`
  cannot add a custom primitive with custom autodiff rules without dropping down
  to C++ (`TORCH_LIBRARY`) or Python (`PyO3`); the custom-autograd surface is not
  reachable from Rust alone.
- **Column-major is an interop feature, not a hazard.** Owned tensors are
  column-major to connect cleanly with Fortran, Julia, MATLAB, Eigen3, and
  LAPACK/BLAS conventions. The column-major↔row-major boundary is bridged by
  strided views and slices, so interop does not require eager copies (see
  [`docs/guides/memory-order.md`](docs/guides/memory-order.md)). Do not treat the
  ordering as a barrier to adoption.

## Which API Should I Use?

| If your workflow needs | Start with |
| --- | --- |
| Fixed scalar type, ordinary tensor computation, no autodiff | `TypedTensor<T, R>` |
| Runtime dtype selection or direct backend dispatch | `Tensor` with an explicit backend |
| Immediate execution in one runtime, optionally with `backward()` on scalar losses | `EagerTensor` and `EagerRuntime` |
| `grad`, `vjp`, and `jvp` on traced graphs, graph reuse, or repeated compile/run execution | `TracedTensor`, `GraphCompiler`, and `GraphExecutor<B>` |
| CUDA execution | The same tensor API plus explicit CUDA upload/download and supported CUDA backend features |

## Crates

tenferro-rs is a multi-crate workspace. There is intentionally no `tenferro` facade crate; users depend on the crates they need directly.

### Core User Crates

| Crate | Use when you need |
| --- | --- |
| `tenferro-tensor` | Tensor values, typed tensors, views, dtype/runtime tensor contracts, and backend traits |
| `tenferro-cpu` | CPU backend execution |
| `tenferro-gpu` | CUDA backend support, ROCm feature stub, and explicit device transfers |
| `tenferro-runtime` | Eager/traced execution, graph compilation, and extension runtime support |
| `tenferro-ad` | Automatic differentiation |

### Standard Operation Extensions

| Crate | Use when you need |
| --- | --- |
| `tenferro-linalg` | Linear algebra operations |
| `tenferro-einsum` | Einsum and contraction planning |
| `tenferro-fft` | FFT operations |

### Implementation Crates

| Crate | Use when you need |
| --- | --- |
| `tenferro-tensor-core` | Host tensor storage, dtype tags, and metadata-only layouts |
| `tenferro-core-ops` | Core primitive operation metadata shared by runtimes and backends |
| `tenferro-internal-ops` | Unpublished internal graph op vocabulary and AD rule implementations used by tenferro crates |
| `tenferro-internal-extension-macros` | Procedural macros for registering internal extension operation descriptors |

## Minimal CPU Example

<!-- snippet-source: crates/tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]);

    let c = tensor::matmul(&a, &b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

## Documentation

The full guides, tutorials, API reference, architecture notes, and
specifications live at <https://tensor4all.org/tenferro-rs/>. Start with
Getting Started if you are using tenferro-rs for the first time.

- [Design: dynamic and symbolic shapes](https://tensor4all.org/tenferro-rs/design/dynamic-symbolic-shapes.html)
  explains how tenferro represents runtime-dependent dimensions in traced
  programs.
- [Oracle-based AD validation](https://tensor4all.org/tenferro-rs/oracle/tensor-ad-oracles-support.html)
  documents how AD rules are validated against finite-difference and Torch
  reference oracles, and which operation/output gradients are covered. The
  oracle datasets live in
  [`tensor4all/tensor-ad-oracles`](https://github.com/tensor4all/tensor-ad-oracles).

## Benchmarks And Numerical Validation

Official benchmark suites and result tooling live in
[`tensor4all/tenferro-benchmark`](https://github.com/tensor4all/tenferro-benchmark).
Run them there for reproducible cross-backend performance numbers rather than
relying on ad hoc local timings.

Numerical correctness — especially automatic differentiation — is validated
against reference *oracles* (finite-difference checks and Torch reference data)
rather than line coverage alone. See the
[oracle support table](https://tensor4all.org/tenferro-rs/oracle/tensor-ad-oracles-support.html)
for the per-operation AD coverage status and the
[`tensor4all/tensor-ad-oracles`](https://github.com/tensor4all/tensor-ad-oracles)
dataset repository.

## Community

The broader tensor4all community uses the
[tensor4all mailing list](https://groups.google.com/g/tensor4all) for
announcements and [Matrix](https://tensor4all.org/matrix.html) for real-time
chat. The community entry point is <https://tensor4all.org/>.

## Project Status

tenferro-rs is pre-1.0. Public APIs, crate boundaries, backend contracts,
feature flags, and internal architecture are still evolving. The stack is
dogfooded in [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) and
related tensor4all projects.

If you build against `main`, pin commits and expect breaking changes. For
non-trivial upgrades, AI-assisted migration is recommended.

## Development And Trust Model

AI agents are accepted development and review tools in this repository. Their
output is not trusted by itself. Changes are trusted only after explicit
repository rules, CI, oracle-based validation, reproducible benchmarks,
provenance checks, and maintainer review.

## Contributing

Bug-fix pull requests, feature requests, design discussions, documentation
improvements, and benchmark reports are welcome. New feature ideas are welcome
too; implementation PRs for new features must start from accepted feature
request issues so API, backend, AD, dependency, and testing implications can be
reviewed first.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the external contribution policy and
the supported AI-assisted issue-intake and bug-fix PR workflows across Codex
CLI, Claude Code, and OpenCode.

See [GOVERNANCE.md](GOVERNANCE.md) for maintainer roles, merge authority, and
the project-direction decision model. Maintainers are listed in
[CONTRIBUTORS.md](CONTRIBUTORS.md).
