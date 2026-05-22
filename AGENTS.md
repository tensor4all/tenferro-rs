# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before acting, read the latest shared Tensor4all agent rules from the
`tensor4all-agent-rules` repository. If internet access is unavailable or the
remote cannot be resolved, use the sibling checkout:

- `../tensor4all-agent-rules/rules/index.md`

Load only the common, Rust, performance, numerical, docs, or benchmark rule
files relevant to the task.

Then read the vendored `template-rs` baseline rules and the tenferro-specific
rules:

- `ai/vendor/template-rs/common-agent-rules.md`
- `ai/vendor/template-rs/numerical-rust-rules.md`
- `ai/vendor/template-rs/pr-workflow-rules.md`
- `REPOSITORY_RULES.md`

The sections below are tenferro-specific additions and overrides.

Before implementation work, review `REPOSITORY_RULES.md`.
Before creating a PR, review `REPOSITORY_RULES.md` again.
Before touching AD rules, oracle replay, or linearized boundary code, review `REPOSITORY_RULES.md` first.

## Current Implementation Status

The workspace contains active implementations alongside evolving APIs. Implementation work is allowed unless a task explicitly says otherwise.

## Repository Rule Source

Keep cross-repository implementation rules in `tensor4all-agent-rules`.
Keep tenferro-specific durable rules in `REPOSITORY_RULES.md`. This
`AGENTS.md` file is the entry point and tenferro-specific orientation; avoid
duplicating the detailed performance, layout, CPU kernel, slicing, cache,
threading, and GPU backend rules here.

### GPU Status

CUDA GPU support is implemented through the feature-gated CubeCL backend across
the concrete tensor, eager, and traced execution surfaces. Performance
optimization is still active work. The remaining CUDA limitations are specific:
`eig`, `full_piv_lu`, `full_piv_lu_solve`, `dynamic_update_slice`, `I64`
numeric/linalg gaps, and selected complex analytic or ordering operations.
HIP/ROCm remains stubbed. Outside explicit GPU implementation tasks, check
`docs/guides/devices-and-gpu.md` and the current CUDA/CubeCL backend tests
before assuming a specific op/dtype/backend combination is supported.

### Documentation Requirements

Every public type, trait, and function **must** include minimal but sufficient usage examples in its doc comments (`/// # Examples`). The examples should help a human quickly understand how to use the API. Doc examples must compile and run as doctests; do not use `ignore` or `no_run`. Crate-level docs (`//!`) should include typical end-to-end usage examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU placement metadata
- Graph-based traced execution via `TracedTensor` + `Engine`
- High-level einsum with N-ary contraction tree optimization
- Automatic differentiation (VJP/JVP/HVP) for the standard dense numeric path
- Single execution IR (`ExecOp`) plus a pass pipeline for backend dispatch

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [`docs/design/`](docs/design/) for architecture and design documents.

**Note**: Files under `docs/plans/` are historical records of past design discussions and decisions. They may contradict the current API or design — do not update them to match the current state.

## Performance, Layout, And Backend Rules

See `REPOSITORY_RULES.md` for the authoritative performance and layout
contracts, including column-major ordering, hidden materialization, range
checks and slicing, CPU kernel ownership, anti-patterns, cache ownership, CPU
threading, and GPU backend conventions.

Do not assume nearby existing implementation patterns are performance-correct.
This repository still contains active optimization work and some legacy
tradeoffs. Before copying patterns in tensor kernels, graph/compiler planning,
caches, GPU kernels, benchmarks, or user-facing examples, check for hidden
materialization, repeated per-element index work, avoidable large-state clones,
repeated linear scans, accidental single-thread GPU execution, and weak
shape-only validation. Prefer improving the pattern or documenting the residual
risk rather than propagating it.

## Code Style

- `cargo fmt --all` for formatting (always run before committing)
- Avoid `unwrap()`/`expect()` in library code
- Use `thiserror` for public API error types

### File Organization

Keep source files **small and focused** — one logical concern per file. Use **~1000 lines** as the soft upper bound; files in the 500–1000 range are fine when they cover a single coherent concern. Actively split files that exceed 1000 lines. Benefits:

- **Abstraction review**: module boundaries make the public/private API surface explicit and easier to audit
- **Parallel editing**: multiple agents (or humans) can work on separate files without merge conflicts
- **Navigation**: smaller files are faster to read and search

When a file grows large, split it by functionality (e.g., parsing, plan computation, execution, public API, AD rules) rather than by arbitrary line count.

### Test Coverage Target

Every source file should have **90%+ line coverage**. When adding new code,
add tests that cover the new paths. When modifying existing code, check
coverage for the modified file and add tests if below 90%.

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

Tests follow implementation ownership.

- Public facade crates should prefer integration tests for user-visible
  behavior.
- Private implementation details must be tested in the crate that owns the
  implementation, typically an internal crate, not through a public facade
  crate.
- If a crate sets `[lib] test = false`, do not add `src/**/tests`, inline
  `#[cfg(test)] mod tests`, or other crate-local unit-test entrypoints to that
  crate.
- If a private helper in a facade crate needs direct unit testing, move that
  helper into the owning internal crate instead of re-enabling facade-crate lib
  tests.
- This rule is enforced by repository contract tests and must stay green in CI.

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
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

If `cargo fmt --all --check` fails, run `cargo fmt --all` to fix formatting automatically.

Additionally, verify the following before pushing:

- **Side review**: Re-read `REPOSITORY_RULES.md` and review the local diff against repository rules before creating a PR. Fix any findings, or explicitly document residual risks.
- **Sample code verification**: All code examples in `README.md` and `docs/getting-started/` must compile and run correctly. Extract and test any changed examples.
- **Design document updates**: When code changes affect architecture or specifications, update the corresponding documents in `docs/architecture/`, `docs/spec/`, or `docs/design/`. Stale documentation is worse than no documentation.

### PR Creation Rules

- PRs to `main` must be created using `gh pr create`
- Do not include AI-generated analysis reports as standalone files in PRs
- Enable auto-merge after creating a PR: `gh pr merge --auto --squash --delete-branch`
- `createpr` must confirm auto-merge remains enabled and the required branch protection checks are still configured

## Build Commands

```bash
# Build entire workspace
cargo build

# Build a specific crate
cargo build -p tenferro

# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p tenferro-einsum

# Run a single test
cargo test test_name

# Check formatting
cargo fmt --check

# Coverage check (per-file thresholds)
# Target: 90%+ line coverage per file. Files below 90% should have tests added.
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json

# Build rustdoc and docs site inputs
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py

# GPU (CubeCL) tests — requires NVIDIA GPU + CUDA 12
# Set CUBECL_DEBUG_LOG=0 to suppress verbose JIT compilation logs.
# GPU tests are marked #[ignore] so they don't fail on non-GPU machines.
# Use --ignored to actually run them.
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-tensor --features cubecl -- --ignored
```

### CubeCL Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUBECL_DEBUG_LOG` | `0` | Suppress JIT compilation log output (default is verbose) |
| `CUDA_PATH` | `/usr/local/cuda-12.0` | CUDA toolkit root for NVRTC header resolution |
| `LD_LIBRARY_PATH` | Include CUDA + cuTENSOR lib dirs | Runtime library loading |

Set these in CI and local dev shells. Without `CUBECL_DEBUG_LOG=0`, cubecl
emits generated CUDA source for every JIT-compiled kernel, producing
millions of log lines during test runs.

### FFI Library Path Configuration

The CubeCL backend loads cuTENSOR, cuSOLVER, and cuBLAS at runtime via
`dlopen`. Default search paths try v12 first, then v11, then bare soname.
Override with environment variables:

| Variable | Library | Example |
|----------|---------|---------|
| `TENFERRO_CUTENSOR_PATH` | cuTENSOR | `/opt/cuda-12.4/lib64/libcutensor.so.2` |
| `TENFERRO_CUSOLVER_PATH` | cuSOLVER | `/opt/cuda-12.4/lib64/libcusolver.so.12` |
| `TENFERRO_CUBLAS_PATH` | cuBLAS | `/opt/cuda-12.4/lib64/libcublas.so.12` |

Colon-separated paths are supported (like `LD_LIBRARY_PATH`).

### Device Transfer Policy

tenferro follows the **PyTorch convention**: no implicit CPU↔GPU transfer.
Tensors must be on the correct device before passing to backend ops.

```rust
// Upload to GPU
let gpu_tensor = cubecl::upload_tensor(backend.runtime(), &cpu_tensor)?;
// Compute on GPU
let result = backend.add(&gpu_a, &gpu_b)?;
// Download to CPU
let cpu_result = cubecl::download_tensor(backend.runtime(), &result)?;
```

**Error behavior:**
- GPU op receives CPU tensor → `Error::BackendFailure` with message
  "expected GPU tensor ... use upload_tensor()"
- CPU op receives GPU tensor → panic (programming error, not recoverable)
- `TypedTensor::host_data()` on GPU buffer → panic with diagnostic message

The execution pipeline (`eval_exec_ir`, segmented dispatch) handles device
placement internally — `Constant` ops auto-upload via `upload_host_tensor()`,
and host-dependent ops (`ShapeOf`, `DynamicTruncate`) read only metadata or
download single scalars.

**cuSOLVER feature coverage:**

| Feature | cuSOLVER support | GPU status |
|---------|-----------------|------------|
| SVD, QR, Cholesky, LU, Eigh | All versions (11.4+) | GPU |
| Triangular solve | Via cuBLAS (all versions) | GPU |
| General eigendecomposition (eig) | **Not in cuSOLVER** | Returns `BackendFailure` — user must download to CPU explicitly |

`eig` (non-symmetric eigenvalue decomposition, LAPACK `dgeev`) is not
provided by any version of cuSOLVER. `CubeclBackend::eig` returns
`BackendFailure`. Users must explicitly download the tensor to CPU and
compute via `CpuBackend::eig`. This is a permanent cuSOLVER limitation.

## Workspace Architecture

### Layered Design

```
Layer 4: tenferro             — Public traced frontend: Engine, TracedTensor, lowering, execution,
                                einsum/linalg convenience APIs, VJP/JVP
Layer 3: tenferro-einsum      — High-level einsum syntax, contraction planning, fragment builder
         tenferro-ops         — Graph op vocabulary (`StdTensorOp`) and AD rules
Layer 2: tenferro-tensor      — Dense `Tensor` / `TypedTensor`, backend traits, CPU backend,
                                CUDA/ROCm backend stubs, execution kernels
Shared:  chainrules-core     — Core AD traits: Differentiable, ReverseRule<V>, ForwardRule<V> (no tensor deps)
         chainrules          — Engine-independent scalar AD rules and helpers (← chainrules-core)
         tidu                — AD engine: Tape<V>, TrackedValue<V>, DualValue<V> (← chainrules-core)
         tenferro-device       — Device enum, Error/Result types

Foundation: strided-rs    — Independent workspace (strided-traits → strided-view → strided-kernel)
```

`chainrules-core` defines core AD traits (like Julia's ChainRulesCore.jl), independent
of any tensor type. `chainrules` provides engine-independent scalar AD rules,
and `tidu` provides the AD engine (Tape, TrackedValue, DualValue).
`tenferro-tensor` owns the concrete dense runtime value types and backend
execution surface. `tenferro-ops/src/ad/` is the semantic source of truth for
AD rules. `tenferro` owns traced graph construction, lowering, and
public evaluation APIs.

## AI Workflow Scripts

Repository-local headless launchers live under `ai/`:

- `ai/run-codex-solve-bug.sh`
- `ai/run-claude-solve-bug.sh`

These scripts resolve their prompt path relative to `ai/`, but they always run
the agent from the repository top-level directory. Their default prompt is
`ai/solve_bug_issue.md`, and JSON output is the default mode unless `--text` is
passed.

### Dependency Graph (POC)

```
chainrules-core (← thiserror only, no tensor deps)
    │  Differentiable trait, ReverseRule<V>, ForwardRule<V>
    │
    ↓
chainrules (← chainrules-core)
    │  Engine-independent scalar AD rules
    │
tidu (← chainrules-core)
    │  Tape<V>, TrackedValue<V>, DualValue<V>
    │
tenferro-device (← strided-view for StridedError, ← thiserror)
    │
    ↓
tenferro-tensor
    (← strided-kernel,
     ← strided-traits,
     ← num-traits)
         │
         ▼
tenferro-ops
    (← computegraph,
     ← tidu,
     ← tenferro-tensor)
         │
         ▼
tenferro-einsum
    (← omeco, ← tenferro-ops)
         │
         ▼
tenferro
    (← computegraph, ← tidu, ← tenferro-einsum, ← tenferro-ops, ← tenferro-tensor)
```
