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
| `tenferro-gpu` | CUDA/ROCm backend support and explicit device transfers |
| `tenferro-runtime` | Eager/traced execution, graph compilation, and extension runtime support |
| `tenferro-ad` | Automatic differentiation |

### Standard Operation Extensions

| Crate | Use when you need |
| --- | --- |
| `tenferro-linalg` | Linear algebra operations |
| `tenferro-einsum` | Einsum and contraction planning |
| `tenferro-fft` | FFT operations |

### Published Implementation Crates

These crates are published so the user-facing crates can depend on them through
crates.io. They are not recommended starting points for application code.

| Crate | Role |
| --- | --- |
| `tenferro-tensor-core` | Low-level host tensor model and scalar/layout primitives |
| `tenferro-core-ops` | Core primitive operation catalog |
| `tenferro-internal-ops` | Internal graph op vocabulary and AD rule plumbing |
| `tenferro-internal-extension-macros` | Extension registration procedural macros |

## Minimal CPU Example

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

## Documentation

The full guides, tutorials, API reference, architecture notes, and
specifications live at <https://tensor4all.org/tenferro-rs/>. Start with
Getting Started if you are using tenferro-rs for the first time.

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
