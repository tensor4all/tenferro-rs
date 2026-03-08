# tenferro-rs

A general-purpose tensor computation library in Rust with CPU support today and planned GPU support later.

> [!WARNING]
> GPU support is currently stubbed and not implemented. The repository contains
> CUDA/HIP-facing types and API surfaces, but production GPU allocation,
> transfer, and execution are still future work.

## Overview

`tenferro-rs` is a Rust workspace providing:

- Dense tensor types with CPU support today and planned GPU support later
- cuTENSOR/hipTensor-compatible operation protocol (`TensorPrims<A>` trait)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP)
- C FFI for Julia/Python integration

Built on top of [strided-rs](https://github.com/tensor4all/strided-rs) for cache-optimized strided array operations.

## GPU Status

CPU functionality is actively implemented and tested. GPU-facing modules exist
to stabilize the future API shape, but the current CUDA/HIP path is still a
stub. Outside explicit bug exploration or implementation work, do not assume a
GPU code path works just because the type, trait, or FFI entrypoint exists.

### Influences

The API and internal architecture are strongly influenced by
[PyTorch / libtorch](https://github.com/pytorch/pytorch):

- **Tensor type** — `Tensor<T>` with reference-counted storage and zero-copy
  view operations (permute, broadcast, diagonal, narrow, select) mirrors
  `at::Tensor` / `c10::Storage`.
- **Plan-based execution** — The `TensorPrims<A>` describe-plan-execute
  protocol follows the cuTENSOR / BLAS pattern used by PyTorch's GPU backend.
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
dispatch, and algebra parameterization (`TensorPrims<A>`) enabling custom
semirings (e.g., tropical).

For a detailed feature-by-feature mapping, see
[`docs/design/reference/libtorch.md`](docs/design/reference/libtorch.md).

## `tenferro-dyadtensor` AD Status

| Area | Status | Notes |
| --- | --- | --- |
| Multi-input backward | Strong | `einsum`, `solve`, `solve_triangular`, `lstsq` |
| Forward mode | Strong | Best supported when only a few inputs carry tangents |
| Multi-input HVP | Partial | Explicitly exposed for `einsum` |
| Higher-order derivatives (non-HVP) | Partial | Available through chainrules-backed `Variable<Tensor<T>>` flows, but the dyadtensor convenience surface and validation depth are still limited |
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

Output:

- `target/docs-site/index.html` (top page)
- `target/docs-site/design/` (formal design docs)
- `target/docs-site/api/` (`cargo doc --workspace` output)

The shared docs deploy workflow publishes the same `target/docs-site` tree to GitHub Pages on pushes to `main`.

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
