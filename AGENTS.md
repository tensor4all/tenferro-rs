# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU support
- cuTENSOR/hipTensor-compatible operation protocol (`TensorOps` trait)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP)
- C FFI for Julia/Python integration

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [tensor4all/tensor4all-meta](https://github.com/tensor4all/tensor4all-meta) for architecture and design documents.

## Pre-Push / PR Checklist

Before pushing or creating a pull request, **all** of the following must pass:

```bash
cargo fmt --check   # formatting
cargo test          # all tests
```

If `cargo fmt --check` fails, run `cargo fmt` to fix formatting automatically.

### PR Creation Rules

- PRs to `main` must be created using `gh pr create`
- AI-generated PRs must include `Generated with [Claude Code](https://claude.com/claude-code)` in the body
- Do not include AI-generated analysis reports as standalone files in PRs

## Build Commands

```bash
# Build entire workspace
cargo build

# Build a specific crate
cargo build -p tenferro-tensorops

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
cargo bench -p tenferro-tensorops -- contraction

# Run benchmarks with native CPU features
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Workspace Architecture

### Layered Design

```
Layer 4: tenferro-einsum      — High-level einsum on Tensor<T>, N-ary tree, algebra dispatch, backward
Layer 3: tenferro-tensor       — Tensor<T> = DataBuffer + shape + strides, zero-copy view ops
Layer 2: tenferro-tensorops    — "Tensor BLAS": cuTENSOR-compatible TensorOps trait, plan-based execution
Shared:  tenferro-device       — Device enum, BackendRegistry, GPU vtable (dlopen)
Layer 1: CPU backends          — strided-kernel + GEMM (faer/cblas)
         GPU backends          — cuTENSOR / hipTensor via tenferro-device vtable

Foundation: strided-rs    — Independent workspace (strided-traits → strided-view → strided-kernel)
```

