# tenferro-rs

**A Rust-native tensor & autodiff stack for scientific computing.**

[![CI](https://github.com/tensor4all/tenferro-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tensor4all/tenferro-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/tenferro-runtime.svg)](https://crates.io/crates/tenferro-runtime)
[![docs](https://img.shields.io/badge/docs-tensor4all.org-blue)](https://tensor4all.org/tenferro-rs/)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](LICENSE-MIT)

tenferro-rs provides dense tensors, linear algebra, einsum, FFT, and
extensible automatic differentiation — eager like PyTorch, traced like JAX —
natively in Rust, with explicit CPU, CUDA, and experimental WebGPU backend
control. It is built for scientific workloads rather than deep learning:
storage is column-major, aligned with LAPACK, Fortran, and Julia conventions;
traced programs compile once and are reused while concrete sizes resolve at
runtime; and operation families and AD rules can be extended from external
crates.

![tenferro-rs architecture overview](docs/assets/tenferro-architecture.svg)

## Quick Example

Add the runtime and CPU backend crates:

```toml
[dependencies]
tenferro-runtime = "0.2"
tenferro-cpu = "0.2"
```

<!-- snippet-source: crates/tenferro-runtime/examples/cpu_quickstart.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0])?;

    let c = a.matmul(&b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

The same stack scales up to PyTorch-style eager autodiff with `backward()`
([tutorial](https://tensor4all.org/tenferro-rs/tutorials/eager-autodiff-pytorch-style.html))
and JAX-style traced `grad`/`vjp`/`jvp`
([tutorial](https://tensor4all.org/tenferro-rs/tutorials/traced-autodiff-jax-style.html)).
Setup notes, including local-checkout builds and BLAS provider selection, are
in [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/index.html).

## Is tenferro for You?

tenferro-rs is a good fit when:

- **Shapes are only known at runtime.** A traced program is compiled once and
  reused while ranks, truncation thresholds, and data-dependent iteration
  counts resolve at execution time — the daily reality of tensor networks and
  much of adaptive scientific computing (see
  [dynamic and symbolic shapes](https://tensor4all.org/tenferro-rs/design/dynamic-symbolic-shapes.html)).
- **You want autodiff in Rust without a Python runtime.** `backward()` on
  eager tensors; `grad`, `vjp`, `jvp`, and HVP on traced graphs; shipped as a
  single binary.
- **Your data lives in the column-major world.** Storage matches Fortran,
  Julia, MATLAB, and LAPACK/BLAS conventions, and strided views bridge
  row-major data without eager copies (see
  [memory order](https://tensor4all.org/tenferro-rs/guides/memory-order.html)).
- **You need operations the core does not ship.** Operations and their AD
  rules live outside the core tensor type, so an external crate can add an
  operation family — even a different algebra, such as the tropical-semiring
  example — and it flows through the same eager and traced autodiff (see
  [custom operations](https://tensor4all.org/tenferro-rs/guides/custom-operations.html)).

tenferro-rs deliberately builds on the Rust numerics ecosystem instead of
replacing it: [`faer`](https://github.com/sarah-quinones/faer-rs) for dense
linear algebra, [CubeCL](https://github.com/tracel-ai/cubecl) for GPU
kernels, [`omeco`](https://crates.io/crates/omeco) for contraction ordering,
and [`num-traits`](https://github.com/rust-num/num-traits) /
[`num-complex`](https://github.com/rust-num/num-complex) for generic
numerics.

Where else to look: [ndarray](https://github.com/rust-ndarray/ndarray) for
general N-dimensional arrays without autodiff or traced graphs;
[faer](https://github.com/sarah-quinones/faer-rs) directly for pure dense
linear algebra; [Burn](https://github.com/tracel-ai/burn) and
[candle](https://github.com/huggingface/candle) for deep-learning workloads.
If your host language is Python, JAX and PyTorch are the natural choice —
tenferro-rs is for projects that want this kind of stack natively in Rust.

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

tenferro-rs is a multi-crate workspace. There is intentionally no `tenferro`
facade crate; depend directly on the crates you need, starting with the
smallest API that solves your problem.

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

The implementation crates `tenferro-tensor-core`, `tenferro-core-ops`,
`tenferro-internal-ops`, and `tenferro-internal-extension-macros` are
published building blocks for the crates above; most users never depend on
them directly.

## Documentation

The full guides, tutorials, API reference, architecture notes, and
specifications live at <https://tensor4all.org/tenferro-rs/>. Start with
[Getting Started](https://tensor4all.org/tenferro-rs/getting-started/index.html);
PyTorch/JAX users can also jump in through the
[PyTorch and JAX mapping](https://tensor4all.org/tenferro-rs/getting-started/pytorch-jax-mapping.html).

Selected deep dives:

- [Devices and GPU](https://tensor4all.org/tenferro-rs/guides/devices-and-gpu.html)
  — explicit CPU, CUDA, and experimental WebGPU control; tensors never move
  between devices silently.
- [Dynamic and symbolic shapes](https://tensor4all.org/tenferro-rs/design/dynamic-symbolic-shapes.html)
  — how runtime-dependent dimensions work in traced programs.
- [XLA and PJRT](https://tensor4all.org/tenferro-rs/guides/xla.html)
  — experimental StableHLO lowering and PJRT plugin loading for static-shaped
  graphs via `tenferro-xla`.
- [Custom operations](https://tensor4all.org/tenferro-rs/guides/custom-operations.html)
  — extension operations and AD rules from external crates.

## Project

**Why tenferro-rs exists.** Moving tensor-network engines from Julia to Rust
exposed a missing layer: a scientific-computing tensor stack between ndarray,
faer, and the deep-learning frameworks — column-major, dynamic-shape,
autodiff-capable, and extensible. tenferro-rs fills that layer and bridges to
the Fortran/LAPACK, Julia, and JAX/PyTorch worlds. The background story is in
the introduction post:
[From Julia to Rust: a differentiable tensor stack for scientific computing in the agentic AI era](https://tensor4all.org/blog/introducing-tenferro-rs/).

**Stability.** tenferro-rs is a pre-1.0 experimental research platform;
public APIs may change substantially, including across 0.x releases. Pin
exact versions or commits, and follow the current documentation as the
primary reference — migration notes accompany major breaking changes. The
stack is dogfooded in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs) and related
tensor4all projects.

**Engineering discipline.** Numerical correctness — especially automatic
differentiation — is validated against reference oracles (finite-difference
and Torch reference data in
[tensor-ad-oracles](https://github.com/tensor4all/tensor-ad-oracles); see the
[oracle support table](https://tensor4all.org/tenferro-rs/oracle/tensor-ad-oracles-support.html)),
enforced per-file coverage thresholds, and the reproducible
[tenferro-benchmark](https://github.com/tensor4all/tenferro-benchmark) suite.
Every documentation example compiles and runs in CI.

**AI-assisted development.** tenferro-rs assumes AI-assisted and agentic
coding for development, migration, and review. AI output is never accepted as
authority by itself: changes are validated against repository rules
([REPOSITORY_RULES.md](REPOSITORY_RULES.md)), oracle checks, CI, reproducible
benchmarks, and maintainer review, with design records under
[docs/design/](docs/design/) and [docs/worklogs/](docs/worklogs/) keeping the
process coherent. Why we work this way is part of the
[introduction post](https://tensor4all.org/blog/introducing-tenferro-rs/).

## Community

Questions, design discussions, and contributor coordination happen in the
tenferro Matrix room:

- [#tenferro-tensor4all:matrix.org](https://matrix.to/#/#tenferro-tensor4all:matrix.org)

Use GitHub issues for bug reports, feature requests, and decisions that need
tracking; use Matrix for lightweight discussion before filing or implementing
changes. The broader tensor4all community uses the
[tensor4all mailing list](https://groups.google.com/g/tensor4all) for
announcements, and the community entry point is <https://tensor4all.org/>.

## Acknowledgments

tenferro-rs stands on excellent work across several ecosystems, and aims to
integrate with and contribute back to them.

- **GPU.** The GPU backend builds on
  [CubeCL](https://github.com/tracel-ai/cubecl) by the
  [tracel-ai](https://github.com/tracel-ai) team (also the foundation of
  [Burn](https://github.com/tracel-ai/burn)). The temporary `t4a-*` crates
  stage tensor4all patches until they land upstream.
- **CPU / numerics.** Dense CPU linear algebra builds on
  [`faer`](https://github.com/sarah-quinones/faer-rs), with numeric
  foundations from [`num-traits`](https://github.com/rust-num/num-traits) and
  [`num-complex`](https://github.com/rust-num/num-complex).
- **Contraction ordering.** Einsum contraction-order optimization uses
  [`omeco`](https://crates.io/crates/omeco), carrying over ideas from the
  Julia tensor-network ecosystem.
- **Design heritage.** Operation semantics and AD rules living outside an
  all-in-one tensor type follow the Julia numerical-computing community
  (ChainRules, OMEinsum); the eager/traced/AD split follows JAX and PyTorch.
  Autodiff builds on [`tidu`](https://github.com/tensor4all/tidu-rs), a
  tensor4all crate for `Primitive`-generic graph transforms that are not tied
  to tensors.

Thanks to these projects, communities, and their maintainers.

## Contributing

Bug reports, minimal reproducers, proposed regression tests, feature
requests, design discussions, documentation improvements, benchmark reports,
and prototype branches are welcome in issues. Pull request creation is
currently restricted to collaborators, and collaborator feature PRs must
start from accepted feature-request issues.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution policy and the
supported AI-assisted workflows, and [GOVERNANCE.md](GOVERNANCE.md) for
maintainer roles, merge authority, and the project-direction decision model.
Maintainers are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).
