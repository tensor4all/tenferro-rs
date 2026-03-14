# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before acting, read the vendored shared rules from `template-rs`:

- `ai/vendor/template-rs/common-agent-rules.md`
- `ai/vendor/template-rs/numerical-rust-rules.md`
- `ai/vendor/template-rs/pr-workflow-rules.md`

The sections below are tenferro-specific additions and overrides.

## Current Implementation Status

The workspace contains active implementations alongside evolving APIs. Implementation work is allowed unless a task explicitly says otherwise.

### GPU Status

GPU support is still partial and experimental. CUDA-only allocation,
CPU<->GPU transfer, and limited cuTENSOR-backed primitive execution now exist,
but broader GPU coverage is incomplete and HIP remains stubbed. Outside
explicit GPU implementation tasks, do not assume a GPU path works just because
the symbol is present.

### Documentation Requirements

Every public type, trait, and function **must** include minimal but sufficient usage examples in its doc comments (`/// # Examples`). The examples should help a human quickly understand how to use the API. Use `ignore` attribute on examples that cannot run in docs. Crate-level docs (`//!`) should include typical end-to-end usage examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU support
- Family-based primitive execution protocol (`TensorSemiringCore/FastPath`, `TensorScalarPrims`, `TensorAnalyticPrims`)
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP) [future]
- C FFI for Julia/Python integration (`tenferro-capi`)

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [`docs/design/`](docs/design/) for architecture and design documents.

**Note**: Files under `docs/plans/` are historical records of past design discussions and decisions. They may contradict the current API or design — do not update them to match the current state.

## Performance-Critical Conventions

### Column-Major Dimension Ordering

tenferro uses **column-major** (Fortran order) storage: the leftmost dimension has the smallest stride and varies fastest in memory. When designing internal layouts for multi-dimensional operations (einsum GEMM, linalg, etc.), dimension ordering must respect this:

- **Batch dimensions go on the RIGHT (trailing)**: In col-major, rightmost dims have the largest stride. Placing batch dims on the right means each batch slice occupies a contiguous block of memory, giving good cache locality for the GEMM kernel operating within each slice.
- **Contraction/compute dimensions go on the LEFT (leading)**: lo (M), sum (K), ro (N) dims should be leftmost so the GEMM kernel accesses contiguous memory.

**Wrong** (batch on left in col-major): `A[batch..., m, k]` — batch has smallest stride, so elements within each `(m, k)` slice are scattered across memory.

**Correct** (batch on right in col-major): `A[lo..., sum..., batch...]` — each batch slice is contiguous, matching strided-rs's convention and standard GEMM cache behavior.

This applies to `target_a`, `target_b`, `c_gemm_shape` in einsum's `GemmPlan`, and to any future batched operation layout.

## Code Style

- `cargo fmt --all` for formatting (always run before committing)
- Avoid `unwrap()`/`expect()` in library code
- Use `thiserror` for public API error types

### File Organization

Keep source files **small and focused** — one logical concern per file. Avoid monolithic files that grow beyond ~500 lines. Benefits:

- **Abstraction review**: module boundaries make the public/private API surface explicit and easier to audit
- **Parallel editing**: multiple agents (or humans) can work on separate files without merge conflicts
- **Navigation**: smaller files are faster to read and search

When a file grows large, split it by functionality (e.g., parsing, plan computation, execution, public API, AD rules) rather than by arbitrary line count.

### Unit Test Organization

For Rust modules, keep production source files focused on production code.
Do not keep inline `#[cfg(test)]` blocks in normal modules unless the file is a
genuinely tiny leaf module and the test is trivially small. Prefer
module-local test directories such as `src/<module>/tests/*.rs` and leave only
`#[cfg(test)] mod tests;` in the source file. Reserve crate-root `tests/` for
integration tests. Do not use `include!` to inject test files into modules.

When splitting tests, optimize for keeping AI and human reading context clean:
a developer reading `src/**` should not need to scroll through large unit-test
blocks to understand the implementation. Prefer splitting larger extracted test
suites by concern rather than keeping one monolithic test module.

### ASCII Diagrams

When writing ASCII flow diagrams or box diagrams in documentation or design docs:
- Use **uniform inner width** for all boxes in the same diagram to prevent misaligned borders
- **Avoid nested boxes** inside other boxes — they are fragile and prone to alignment errors
- Verify character counts between `│` delimiters match the dash count in `┌───┐` / `└───┘` borders

### Dependencies

Use **workspace dependencies** for libraries shared across multiple crates. Define the dependency once in the workspace `Cargo.toml` under `[workspace.dependencies]`, then reference it with `dep.workspace = true` in each crate's `Cargo.toml`.

## Git Worktree Rules

When using git worktrees for feature development, **always branch from the latest `main`** before starting implementation. Run `git fetch origin && git checkout -b <branch-name> origin/main` to ensure the branch is up-to-date. Never branch from a stale local state or from another feature branch unless explicitly intended.

## Pre-Push / PR Checklist

Before pushing or creating a pull request, **all** of the following must pass:

```bash
cargo fmt --all --check   # formatting
cargo test --workspace --release   # all tests
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If `cargo fmt --all --check` fails, run `cargo fmt --all` to fix formatting automatically.

### PR Creation Rules

- PRs to `main` must be created using `gh pr create`
- AI-generated PRs must include `Generated with [Claude Code](https://claude.com/claude-code)` in the body
- Do not include AI-generated analysis reports as standalone files in PRs
- Enable auto-merge after creating a PR: `gh pr merge --auto --squash --delete-branch`
- `createpr` must confirm auto-merge remains enabled and the required branch protection checks are still configured

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

# Coverage check (per-file thresholds)
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json

# Run benchmarks
cargo bench

# Run a specific benchmark
cargo bench -p tenferro-prims -- contraction

# Run benchmarks with native CPU features
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Common Performance Anti-Patterns

When writing performance-sensitive code (GEMM, tensor operations, inner loops), avoid these mistakes:

### 1. Duplicated f64/f32 functions instead of generic code

**Bad:** Copy-pasting the same function body for `f64` and `f32` (e.g., `run_f64` / `run_f32`).

**Good:** Use a trait (e.g., `FaerGemm`) or macro to share the logic. TypeId dispatch only at the outer boundary.

### 2. Allocating dense buffers when strided access is available

**Bad:** `vec![0.0; m*k]` + copy from strided source + GEMM + copy back to strided destination.

**Good:** Use `faer::MatRef::from_raw_parts(ptr, m, k, row_stride, col_stride)` to access strided data directly — zero allocation, zero copy.

### 3. Zero-initializing buffers that will be immediately overwritten

**Bad:** `vec![0.0; n]` followed by a loop that overwrites every element.

**Good:** `Vec::with_capacity(n)` + `unsafe { set_len(n) }` if you will write all elements, or avoid allocation entirely (see #2).

### 4. Per-element index multiplication in inner loops

**Bad:**
```rust
for j in 0..n {
    for i in 0..m {
        let off = i as isize * row_stride + j as isize * col_stride;
        *ptr.offset(off) *= beta;
    }
}
```

**Good:** Use incremental pointer offsets:
```rust
let mut col_off = 0isize;
for _ in 0..n {
    let mut off = col_off;
    for _ in 0..m {
        *ptr.offset(off) *= beta;
        off += row_stride;
    }
    col_off += col_stride;
}
```

### 5. Allocating Vec inside hot loops

**Bad:**
```rust
for_each_index(&dims, |idx| {
    for i in 0..n {
        let buf = vec![0usize; rank];  // ALLOCATION PER ITERATION
        // ...
    }
});
```

**Good:** Pre-allocate outside and reuse with `.fill(0)`:
```rust
let mut buf = vec![0usize; rank];
for_each_index(&dims, |idx| {
    for i in 0..n {
        buf.fill(0);
        // ...
    }
});
```

### 6. Calling `Backend::plan()` inside hot loops

**Bad:** Computing plans per-step inside the execution loop.

**Good:** Pre-compute all plans before the loop and pass them in.

## Workspace Architecture

### Layered Design

```
Layer 5: tenferro-capi         — C-API (FFI) for Julia/Python: exposes einsum + SVD with AD rules (f64, stateless rrule/frule)
Layer 4: tenferro-einsum       — High-level einsum on Tensor<T>, N-ary tree, algebra dispatch, einsum AD rules
         tenferro-linalg       — Public/composite tensor linalg, result shaping, linalg AD rules
Layer 3: tenferro-prims        — Semiring/scalar/analytic execution families
                                 (TensorSemiringCore/FastPath, TensorScalarPrims, TensorAnalyticPrims)
         tenferro-linalg-prims — Backend-facing factorization/solve/eigensolver kernel contracts
Layer 2: tenferro-tensor       — Tensor<T> = DataBuffer + shape + strides, zero-copy view ops,
                                 impl Differentiable for Tensor<T>
Shared:  chainrules-core     — Core AD traits: Differentiable, ReverseRule<V>, ForwardRule<V> (no tensor deps)
         chainrules           — AD engine: Tape<V>, TrackedValue<V>, DualValue<V> (← chainrules-core)
         tenferro-algebra      — HasAlgebra trait (UX sugar for algebra inference), Semiring trait, Standard<T> typed algebra
         tenferro-device       — Device enum, Error/Result types
Layer 1: CPU backends          — strided-kernel + GEMM (faer/cblas) [future]
         GPU backends          — cuTENSOR / hipTensor via tenferro-device vtable [future]

Foundation: strided-rs    — Independent workspace (strided-traits → strided-view → strided-kernel)
```

`chainrules-core` defines core AD traits (like Julia's ChainRulesCore.jl), independent
of any tensor type. `chainrules` provides the AD engine (Tape, TrackedValue, DualValue).
`Tensor<T>` implements `Differentiable` in `tenferro-tensor`.
Operation-specific AD rules live with their operations: `tenferro-einsum` owns einsum
AD functions (`tracked_einsum`, `dual_einsum`, `einsum_rrule`, `einsum_frule`);
`tenferro-linalg` owns linalg AD functions (`svd_rrule`, `svd_frule`, etc.),
while `tenferro-linalg-prims` owns only backend-facing execution contracts.

### Dependency Graph (POC)

```
chainrules-core (← thiserror only, no tensor deps)
    │  Differentiable trait, ReverseRule<V>, ForwardRule<V>
    │
    ↓
chainrules (← chainrules-core)
    │  Tape<V>, TrackedValue<V>, DualValue<V>
    │
tenferro-device (← strided-view for StridedError, ← thiserror)
    │
    ↓
tenferro-algebra (← strided-traits)
    │  HasAlgebra trait (UX sugar), Semiring trait, Standard<T> typed algebra
    │
    ├────────────────────┐
    ↓                    ↓
tenferro-device  tenferro-tensor
    │              (← strided-view,
    │               ← strided-traits,
    │               ← num-traits,
    │               ← chainrules-core)
    │               impl Differentiable for Tensor<T>
    │                    │
    └────────┬───────────┘
             ├───────────────┐
             ↓               ↓
        tenferro-prims   tenferro-linalg-prims
          (← strided-view,
           ← strided-traits,
           ← tenferro-tensor)
             │               │
             ▼               ▼
        tenferro-einsum
          (← strided-traits, ← chainrules)
        tenferro-linalg
          (← strided-traits, ← chainrules-core, ← tenferro-linalg-prims)
               ↓
          tenferro-capi
              (← tenferro-tensor, ← tenferro-einsum, ← tenferro-linalg, ← tenferro-device)
```
