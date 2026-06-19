# tenferro-rs

tenferro-rs is a modular Rust tensor stack for scientific computing and
general-purpose tensor workflows.

![tenferro-rs architecture overview](docs/assets/tenferro-architecture.svg)

## Why tenferro-rs Exists

`tenferro` means tensor computation with an iron/Rust flavor: `tensor` +
`ferro`.

The project aims to provide a Rust-native tensor stack for typed tensor
computation, immediate execution with optional `backward()`, traced graph
execution, automatic differentiation, linear algebra, einsum, FFT, and explicit
CPU, CUDA, and experimental WebGPU backend control.

tenferro-rs is influenced by JAX and PyTorch in its split between immediate
execution, traced graphs, and AD. It is also influenced by the Julia numerical
computing ecosystem, where operation semantics and AD rules can live outside a
single all-in-one tensor type.

tenferro-rs is Rust-native, but it should fit established numerical computing
conventions rather than replacing them with project-specific rules. It uses
Rust's type system, ownership model, and crate boundaries where those choices
help.

We also see Rust's package system — fine-grained crates, feature flags, and
semver-aware dependency composition — as a strong place for **cross-ecosystem
collaboration**. tenferro-rs prefers to build on and contribute back to existing
crates (GPU kernels, linear algebra, autodiff, contraction ordering) rather than
reinvent them, and to bridge cleanly to the Fortran/LAPACK, Julia, and JAX/PyTorch
worlds. The modular crate layout is meant to make tenferro components reusable as
shared infrastructure, not just as an internal stack.

## Design Principles

- Keep tensor types and operation crates modular, rather than building one
  all-in-one tensor type.
- Let users choose the lowest API that solves their problem. Autodiff, traced
  graphs, CUDA, experimental WebGPU, linalg, einsum, and FFT are opt-in
  capabilities.
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
- Manage engineering quality through three pillars — reference **oracles**
  (finite-difference and Torch reference data), a comprehensive **unit-test**
  suite with enforced per-file coverage thresholds, and a reproducible
  **benchmark suite** — backed by invariants, residual checks, and provenance
  checks. This engineering discipline is what lets the project iterate quickly
  (see [Stability Policy](#stability-policy)) without sacrificing correctness.

## When tenferro Is a Good Fit

tenferro-rs is designed for workloads where tensor shapes are not always fixed
before execution. If your computation involves runtime-dependent ranks,
threshold-based filtering, data-dependent iteration counts, or shape-parametric
models, tenferro's traced execution can reuse a compiled program while resolving
the concrete sizes at runtime.

It also fits small-to-medium batched linear algebra with autodiff,
`grad`/`vjp`/`jvp`/HVP workflows in Rust, Rust-native tensor applications, and
extension points for custom algebra or operation families.

The experimental `tenferro-xla` crate lowers static-shaped traced graph
programs to StableHLO and can load PJRT plugins at runtime through environment
variables. Dynamic-shape graphs remain the native runtime's core use case. See
the [XLA and PJRT guide](https://tensor4all.org/tenferro-rs/guides/xla.html).

## Why Build On tenferro-rs

tenferro-rs aims to be a pure-Rust tensor stack for scientific and numerical
computing — one you can build on, extend, and ship as a single binary, rather
than a wrapper around a C++ or Python engine. What that gives you:

- **A complete stack, natively in Rust.** Typed tensors, eager execution with
  `backward()`, traced graphs with reuse, `grad`/`vjp`/`jvp`/HVP autodiff, linear
  algebra, einsum, and FFT — all carried by Rust's type system, ownership model,
  and tooling, with no Python runtime.
- **Extensible operations and AD rules.** Operations and their AD rules live
  *outside* the core tensor crates. An external crate can introduce a new
  operation family — even a different algebra — register its
  `linearize`/`transpose_rule`, and have it flow through the same eager and
  traced autodiff as the built-in ops. The `ext/tropical` semiring extension is a
  worked example; see
  [`docs/guides/custom-operations.md`](docs/guides/custom-operations.md).
- **Fits the numerical-computing world.** Column-major storage lines up with
  Fortran, Julia, MATLAB, Eigen3, and LAPACK/BLAS, and strided views bridge to
  row-major data without eager copies (see
  [`docs/guides/memory-order.md`](docs/guides/memory-order.md)). tenferro-rs
  builds on `faer` for dense linear algebra and on
  [CubeCL](https://github.com/tracel-ai/cubecl) for GPU kernels, contributing to
  the Rust ecosystem rather than reinventing it.
- **Built for shapes you only know at runtime.** A traced program is compiled
  once and reused while concrete sizes — ranks, truncation thresholds,
  data-dependent iteration counts — resolve at execution time.

If your host language is Python, JAX and PyTorch are the natural choice.
tenferro-rs is for the projects that want this kind of stack natively in Rust.

## Which API Should I Use?

| If your workflow needs | Start with |
| --- | --- |
| Fixed scalar type, ordinary tensor computation, no autodiff | `TypedTensor<T, R>` |
| Runtime dtype selection or direct backend dispatch | `Tensor` with an explicit backend |
| Immediate execution in one runtime, optionally with `backward()` on scalar losses | `EagerTensor` and `EagerRuntime` |
| `grad`, `vjp`, and `jvp` on traced graphs, graph reuse, or repeated compile/run execution | `TracedTensor`, `GraphCompiler`, and `GraphExecutor<B>` |
| CUDA or experimental WebGPU execution | The same tensor API plus explicit GPU upload/download and supported provider backend features |
| Static-shaped StableHLO and PJRT plugin experiments | `GraphCompiler` plus `tenferro-xla` |

## Crates

tenferro-rs is a multi-crate workspace. There is intentionally no `tenferro` facade crate; users depend on the crates they need directly.

### Core User Crates

| Crate | Use when you need |
| --- | --- |
| `tenferro-tensor` | Tensor values, typed tensors, views, dtype/runtime tensor contracts, and backend traits |
| `tenferro-cpu` | CPU backend execution |
| `tenferro-gpu` | CUDA backend support, experimental WebGPU support, future ROCm substrate, and explicit device transfers |
| `tenferro-runtime` | Eager/traced execution, graph compilation, and extension runtime support |
| `tenferro-ad` | Automatic differentiation |
| `tenferro-xla` | Experimental StableHLO lowering and runtime-loaded PJRT plugin support for static-shaped traced graphs |

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

Create a binary crate that depends on `tenferro-runtime` and `tenferro-cpu`.
The same runnable example lives at
`crates/tenferro-runtime/examples/cpu_quickstart.rs`.

<!-- snippet-source: crates/tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])?;

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
- [XLA and PJRT](https://tensor4all.org/tenferro-rs/guides/xla.html)
  documents the experimental StableHLO lowering path and the CUDA, cuTENSOR,
  and PJRT environment variables used for local verification.
- [XLA backend: einsum to StableHLO](https://tensor4all.org/tenferro-rs/tutorials/xla-einsum-backend.html)
  shows a runnable fixed-shape N-ary einsum lowering path through
  `tenferro-xla`.
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

## Get In Touch

Questions, design discussions, and contributor coordination for tenferro happen
in the tenferro Matrix room:

- [#tenferro-tensor4all:matrix.org](https://matrix.to/#/#tenferro-tensor4all:matrix.org)

Matrix is an open, federated chat protocol. You can join from a Matrix client
such as Element, or through the browser flow opened by the link above.

Use GitHub issues for bug reports, feature requests, and decisions that need
tracking; use Matrix for lightweight discussion before filing or implementing
changes. The broader tensor4all community also uses the
[tensor4all mailing list](https://groups.google.com/g/tensor4all) for
announcements, and the community entry point is <https://tensor4all.org/>.

## Acknowledgments

tenferro-rs stands on excellent work across several ecosystems, and aims to
integrate with and contribute back to them rather than wall itself off — even as
it iterates quickly.

- **GPU.** The GPU backend is built on
  [CubeCL](https://github.com/tracel-ai/cubecl) by the
  [tracel-ai](https://github.com/tracel-ai) team (also the foundation of the
  [Burn](https://github.com/tracel-ai/burn) deep learning framework); we use it
  for portable GPU kernels and aim to contribute improvements upstream rather
  than fork the ecosystem.
- **CPU / numerics.** Dense CPU linear algebra builds on
  [`faer`](https://github.com/sarah-quinones/faer-rs), with numeric foundations
  from [`num-traits`](https://github.com/rust-num/num-traits) and
  [`num-complex`](https://github.com/rust-num/num-complex).
- **Contraction ordering.** Einsum contraction-order optimization uses
  [`omeco`](https://crates.io/crates/omeco), carrying over ideas from the Julia
  tensor-network ecosystem (OMEinsum/contraction-order tooling).
- **Design heritage.** The modular structure — operation semantics and AD rules
  living *outside* an all-in-one tensor type — is influenced by the **Julia
  numerical-computing community** (e.g. the ChainRules and OMEinsum/tensor-network
  ecosystems), and by JAX/PyTorch for the eager/traced/AD split.
- **Generic autodiff.** Autodiff is built on `tidu`, a tensor4all crate providing
  `Primitive`-generic graph transforms (`linearize` / `linear_transpose`) that are
  **not tied to tensors** and can drive autodiff for other domains.

Thanks to these projects, communities, and their maintainers.

## Stability Policy

tenferro-rs is a **pre-1.0 experimental research platform**. Before v1.0, public
APIs may change substantially, including across 0.1.x releases: at this stage we
prioritize rapid iteration, backend exploration, and AI-assisted development over
API stability.

This does **not** mean correctness or engineering discipline is relaxed. We keep
tests, runnable examples, documentation, and migration notes for major changes up
to date, and changes are accepted only after the checks described in
[Development And Trust Model](#development-and-trust-model) — repository rules, CI,
oracle-based validation, reproducible benchmarks, provenance checks, and maintainer
review. The detailed, current API documentation is the primary reference for
following the library as it evolves; we write migration notes for major breaking
changes rather than maintaining exhaustive compatibility tables.

For downstream projects, **pin exact versions or commits**, and when upgrading use
AI-assisted refactoring against the current docs to follow breaking changes. The
stack is dogfooded in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) and related tensor4all
projects.

## AI-Assisted Development

tenferro-rs assumes AI-assisted and agentic coding for development, migration,
and review. AI output is not accepted as authority by itself: changes are
validated against repository rules, source-of-truth code and doc comments, CI,
oracle-based numerical checks, reproducible benchmarks when relevant,
provenance checks, and maintainer review.

Correctness and consistency are also kept by a retained knowledge base that the
project builds on rather than rediscovers each time: durable **repository and
engineering rules** ([`REPOSITORY_RULES.md`](REPOSITORY_RULES.md) and the shared
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)),
**design documents** ([`docs/design/`](docs/design/), [`docs/spec/`](docs/spec/)),
and the **history of past development decisions** (work logs in
[`docs/worklogs/`](docs/worklogs/) and review-decision records). This institutional
memory is what keeps a fast, AI-assisted workflow coherent: rules and recorded
decisions constrain new changes — including AI-generated ones — toward the
established design.

Documentation–implementation consistency is itself audited regularly by agents,
on top of the automated doc-snippet and docs-site checks in CI (every doc example
must compile and run). Because the current API documentation is the primary
reference for following the library as it evolves, keeping the docs aligned with
the code is treated as a correctness concern, not an afterthought.

## Contributing

Bug reports, minimal reproducers, proposed regression tests, feature requests,
design discussions, documentation improvements, benchmark reports, and
prototype branches are welcome in issues. Pull request creation is currently
restricted to collaborators.

Collaborator implementation PRs for new features must start from accepted
feature request issues so API, backend, AD, dependency, and testing
implications can be reviewed first.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the external contribution policy and
the supported AI-assisted issue-intake and collaborator bug-fix PR workflows
across Codex CLI, Claude Code, OpenCode, and Kimi CLI.

See [GOVERNANCE.md](GOVERNANCE.md) for maintainer roles, merge authority, and
the project-direction decision model. Maintainers are listed in
[CONTRIBUTORS.md](CONTRIBUTORS.md).
