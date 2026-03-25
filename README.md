# tenferro-rs

**A Rust-native tensor computation engine bridging scientific computing and machine learning.**

## Vision

Most tensor libraries are designed for one world: either ML frameworks (PyTorch, JAX) tuned
for neural networks, or scientific computing tools (NumPy, Julia) tuned for physics and
mathematics. tenferro-rs is designed for the intersection.

```
C / C++ / Fortran / Python / Julia
            │
            │  C FFI  (AD-compatible)
            ▼
       tenferro-rs
  ┌─────────────────────────────┐
  │  einsum over any algebra *  │
  │  full AD  (VJP / JVP / HVP) │
  │  extended precision (xprec) │
  └─────────────────────────────┘
            │
     integration with burn
            │
            ▼
  hybrid applications
  (neural networks + tensor networks, etc.)
```

**\* einsum over any algebra**: Standard einsum multiplies and sums numbers. tenferro-rs lets
you swap out those operations — for example, replacing addition with `max` gives the Tropical
algebra used in combinatorial optimization; replacing `f64` with a double-double type gives
extended precision. The same einsum engine and AD machinery work across all of them.

### Key strengths

- **Callable from C, C++, Fortran, Python, and Julia** via a stable C FFI — drop it into existing HPC codebases without rewriting anything.
- **AD-compatible across language boundaries** — reverse-mode (VJP), forward-mode (JVP), and Hessian-vector products (HVP) are exposed through the C API so Python and Julia AD systems can interoperate.
- **Algebra-parameterized** — einsum, linalg, and AD rules are generic over the algebra, not hardwired to floating-point arithmetic.
- **Extended precision** — planned support for double-double and higher-precision types for numerically demanding simulations.
- **ML bridge** — designed to interoperate with [burn](https://github.com/tracel-ai/burn) for hybrid neural network + tensor network models.

A general-purpose tensor computation library in Rust with CPU support today and planned GPU support later.

> [!WARNING]
> GPU support is still partial and experimental. CUDA-only allocation,
> CPU<->GPU transfer, and limited cuTENSOR-backed primitive execution exist,
> but broad GPU coverage is incomplete and HIP remains stubbed.

## Overview

`tenferro-rs` is a Rust workspace providing:

- Dense tensor types with CPU support today and planned GPU support later
- Family-based primitive execution protocol
  (`TensorSemiringCore`, `TensorSemiringFastPath`, `TensorScalarPrims`,
  `TensorAnalyticPrims`)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP)
- C FFI for Julia/Python integration

Built on top of [strided-rs](https://github.com/tensor4all/strided-rs) for cache-optimized strided array operations.

## GPU Status

CPU functionality is actively implemented and tested. The CUDA path now has
basic allocation, CPU<->GPU transfer, and a small set of runtime-loaded
primitive execution paths, but broader GPU coverage is still incomplete and
HIP remains a stub. Outside explicit GPU implementation tasks, do not assume a
GPU code path works just because the type, trait, or FFI entrypoint exists.

## Quickstart

For a local checkout, a minimal CPU-only downstream crate needs these
workspace members:

```toml
[dependencies]
tenferro-algebra = { path = "../tenferro-rs/tenferro-algebra" }
tenferro-device = { path = "../tenferro-rs/tenferro-device" }
tenferro-tensor = { path = "../tenferro-rs/tenferro-tensor" }
tenferro-prims = { path = "../tenferro-rs/tenferro-prims" }
tenferro-einsum = { path = "../tenferro-rs/tenferro-einsum" }
```

```rust
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::einsum;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

fn main() {
    let col = MemoryOrder::ColumnMajor;
    let mem = LogicalMemorySpace::MainMemory;
    let mut ctx = CpuContext::new(4);

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
    let c = einsum::<Standard<f64>, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None)
        .unwrap();

    assert_eq!(c.dims(), &[2, 2]);
    assert_eq!(c.logical_memory_space(), mem);
}
```

### Autodiff quickstart

For automatic differentiation, use the `tenferro` umbrella crate which
wraps the lower-level crates with a dynamic, AD-aware `Tensor` type
(similar to PyTorch's autograd):

```toml
[dependencies]
tenferro       = { path = "../tenferro-rs/tenferro" }
tenferro-prims = { path = "../tenferro-rs/tenferro-prims" }
```

```rust
use tenferro::{backward, grad, set_default_runtime, BackwardOptions, GradOptions,
               RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

fn main() {
    // 1. Configure the default runtime (required before any operation).
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    // 2. Create tensors and enable gradient tracking.
    let mut x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
    x.set_requires_grad(true).unwrap();

    // 3. Forward pass: loss = sum(exp(x))
    let loss = x.exp().unwrap().sum().unwrap();

    // 4. Compute gradients (functional style, like torch.autograd.grad).
    let grads = grad(&[&loss], &[&x], None, GradOptions::default()).unwrap();
    // grads[0] ≈ exp(x) = [e^1, e^2, e^3]
    assert!(grads[0].is_some());
}
```

For more examples, see the crate docs for `tenferro-einsum` and `tenferro-tensor`.

### Influences

The API and internal architecture are strongly influenced by
[PyTorch / libtorch](https://github.com/pytorch/pytorch):

- **Tensor type** — `Tensor<T>` with reference-counted storage and zero-copy
  view operations (permute, broadcast, diagonal, narrow, select) mirrors
  `at::Tensor` / `c10::Storage`.
- **Plan-based execution** — The primitive family traits keep a
  describe-plan-execute contract that follows the cuTENSOR / BLAS pattern used
  by PyTorch's GPU backend.
- **Automatic differentiation** — Tape-based reverse mode (VJP) and
  dual-number forward mode (JVP) follow PyTorch's autograd and `torch.func`
  design, factored into standalone `chainrules-core` / `chainrules` crates
  (inspired by Julia's
  [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)).
- **Einsum** — Ported from Julia's
  [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl); string notation
  (`"ij,jk->ik"`) is compatible with `torch.einsum`, with N-ary contraction
  tree optimization.
- **Linear algebra** — `tenferro-linalg` mirrors `torch.linalg` (SVD, QR, LU,
  eigen, Cholesky, solve) with differentiable decompositions.

Key differences from PyTorch: column-major default layout with `(m, n, *)`
batch convention, compile-time generics (`Tensor<T>`) instead of runtime dtype
dispatch, and algebra-parameterized primitive families enabling custom
semirings (e.g., tropical).

### Memory Layout Policy

tenferro uses **column-major** layout as its internal canonical order.

This choice is deliberate:

- it aligns naturally with **Julia** and **Fortran**
- it matches the default storage convention of **Eigen3**
- it fits the surrounding **BLAS/LAPACK** ecosystem and the linalg backends
  tenferro targets

Rust integrations such as `ndarray`, Burn, or downstream row-major
applications are expected to normalize at the boundary:

- import row-major data explicitly
- convert into tenferro's column-major canonical tensors for computation
- materialize row-major buffers again when exporting back out

That keeps the internal semantics simple and avoids ambiguous reshape behavior
for unit-dimension layouts where row-major and column-major strides can look
identical.

For a detailed feature-by-feature mapping, see
[`docs/design/reference/libtorch.md`](docs/design/reference/libtorch.md).

## `tenferro` AD Status

| Area | Status | Notes |
| --- | --- | --- |
| Multi-input backward | Strong | `einsum`, `solve`, `solve_triangular`, `lstsq` |
| Forward mode | Strong | Best supported when only a few inputs carry tangents |
| Multi-input HVP | Partial | Explicitly exposed for `einsum` |
| Higher-order derivatives (non-HVP) | Partial | Low-level `tidu::Tape<Tensor<T>>` flows are available, but the `tenferro` frontend validation depth is still limited |
| Linalg AD surface | Available | Broad op coverage, but validation depth is uneven across ops |
| Complex/real matrices | Strong | Complex `einsum`, complex `solve_triangular`, and real-to-complex `eig` are covered |

## Design

See [`docs/design/`](docs/design/) for architecture and design documents, including:

- [Architecture](docs/design/architecture.md) — workspace layers, crate dependency graph, device layer
- [Design Documents](docs/design/README.md) — per-crate API designs (tensor, prims, einsum, algebra, autodiff, etc.)

## AI Workflows

This repository vendors shared agent rules from `template-rs` under `ai/vendor/template-rs/`.
Project-local PR and agent-asset workflows live in:

- `.claude/commands/createpr.md`
- `.claude/commands/check-agent-assets.md`
- `.claude/commands/sync-agent-assets.md`
- `scripts/create-pr.sh`
- `scripts/check-agent-assets.sh`
- `scripts/sync-agent-assets.sh`
- `.github/workflows/docs.yml`
- `scripts/build_docs_site.sh`
- `scripts/check-docs-site.py`

## Documentation

Generate a unified local docs site (design docs + Rust API docs):

```bash
bash scripts/build_docs_site.sh
python3 scripts/check-docs-site.py
```

Extra local tools:

- `quarto` is required for `target/docs-site/design/`
- Graphviz `dot` is required for the dependency graph assets
- without those tools, the script still generates `target/docs-site/api/` and
  `target/docs-site/index.html`

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/design/` (formal design docs)
- `target/docs-site/api/` (`cargo doc --workspace` output)

The shared docs deploy workflow publishes the same `target/docs-site` tree to GitHub Pages on pushes to `main`.

## Oracle Replay Coverage

`tenferro-linalg` continuously replays the vendored
`third_party/tensor-ad-oracles` database during workspace tests. Supported
families are validated against the published first-order references and, where
available, scalarized HVP payloads. Published families that tenferro does not
yet replay are tracked explicitly in:

- [`docs/generated/tensor-ad-oracles-support.md`](docs/generated/tensor-ad-oracles-support.md)

## Coverage

Per-file line coverage is checked against thresholds in `coverage-thresholds.json`.
Files listed in `exclude` are skipped from threshold checking.

```bash
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

## License

Licensed under either of:

- Apache License, Version 2.0 (`LICENSE-APACHE`)
- MIT license (`LICENSE-MIT`)

at your option.
