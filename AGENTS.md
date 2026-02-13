# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Implementation Status

**API skeleton only (POC phase)**: All public functions and trait methods are defined with full signatures and docstrings, but function bodies use `todo!()`. The purpose of this phase is to validate the API design before writing implementations. Do not add implementations — only define types, traits, and function signatures.

### Documentation Requirements

Every public type, trait, and function **must** include minimal but sufficient usage examples in its doc comments (`/// # Examples`). The examples should help a human quickly understand how to use the API. Use `ignore` attribute on examples that cannot run (due to `todo!()` bodies). Crate-level docs (`//!`) should include typical end-to-end usage examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU support
- cuTENSOR/hipTensor-compatible operation protocol (`TensorPrims<A>` trait, parameterized by algebra)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP) [future]
- C FFI for Julia/Python integration [future]

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [tensor4all/tensor4all-meta PR #1](https://github.com/tensor4all/tensor4all-meta/pull/1) for the latest architecture and design documents.

## Code Style

- `cargo fmt --all` for formatting (always run before committing)
- Avoid `unwrap()`/`expect()` in library code
- Use `thiserror` for public API error types

### Dependencies

Use **workspace dependencies** for libraries shared across multiple crates. Define the dependency once in the workspace `Cargo.toml` under `[workspace.dependencies]`, then reference it with `dep.workspace = true` in each crate's `Cargo.toml`.

## Pre-Push / PR Checklist

Before pushing or creating a pull request, **all** of the following must pass:

```bash
cargo fmt --all --check   # formatting
cargo test --workspace    # all tests
```

If `cargo fmt --all --check` fails, run `cargo fmt --all` to fix formatting automatically.

### PR Creation Rules

- PRs to `main` must be created using `gh pr create`
- AI-generated PRs must include `Generated with [Claude Code](https://claude.com/claude-code)` in the body
- Do not include AI-generated analysis reports as standalone files in PRs
- Enable auto-merge after creating a PR: `gh pr merge --auto --squash --delete-branch`

## Build Commands

```bash
# Build entire workspace
cargo build

# Build a specific crate
cargo build -p tenferro-prims

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p tenferro-einsum

# Run a single test
cargo test test_name

# Check formatting
cargo fmt --check

# Run benchmarks
cargo bench

# Run a specific benchmark
cargo bench -p tenferro-prims -- contraction

# Run benchmarks with native CPU features
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Workspace Architecture

### Layered Design

```
Layer 4: tenferro-einsum       — High-level einsum on Tensor<T>, N-ary tree, algebra dispatch
Layer 3: tenferro-tensor       — Tensor<T> = DataBuffer + shape + strides, zero-copy view ops
Layer 2: tenferro-prims        — "Tensor BLAS": TensorPrims<A> trait (algebra-parameterized), plan-based execution
Shared:  tenferro-algebra      — HasAlgebra trait, Semiring trait, Standard type
         tenferro-device       — Device enum, Error/Result types
Layer 1: CPU backends          — strided-kernel + GEMM (faer/cblas) [future]
         GPU backends          — cuTENSOR / hipTensor via tenferro-device vtable [future]

Foundation: strided-rs    — Independent workspace (strided-traits → strided-view → strided-kernel)
```

### Dependency Graph (POC)

```
tenferro-device (← strided-view for StridedError, ← thiserror)
    │
    ↓
tenferro-algebra (← strided-traits)
    │  HasAlgebra trait, Semiring trait, Standard type
    │
    ├────────────────────┐
    ↓                    ↓
tenferro-prims   tenferro-tensor
    │  (← strided-view,     │  (← strided-view,
    │   ← strided-traits)   │   ← strided-traits,
    │                        │   ← num-traits)
    │                        │
    └──────────┬─────────────┘
               ↓
          tenferro-einsum
              (← strided-traits)
```

