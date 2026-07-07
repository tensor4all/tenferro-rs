# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Before acting, read the latest shared tensor4all agent rules from the
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules)
repository online. Start from:

- `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`

If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout:

- `../tensor4all-agent-rules/rules/index.md`

Load only the common, Rust, performance, numerical, docs, or benchmark rule
files relevant to the task. If neither online access nor the sibling checkout
is available, continue from this repository's local rules and state the shared
rules were unavailable when creating a PR.

Then read the tenferro-specific rules:

- `REPOSITORY_RULES.md`

The sections below are tenferro-specific additions and overrides.

Before implementation work, review `REPOSITORY_RULES.md`.
Before creating a PR, review `REPOSITORY_RULES.md` again.
Before touching AD rules, oracle replay, or linearized boundary code, review `REPOSITORY_RULES.md` first.

## Current Implementation Status

The workspace contains active implementations alongside evolving APIs. Implementation work is allowed unless a task explicitly says otherwise.

## Repository Rule Source

Keep cross-repository implementation rules in `tensor4all-agent-rules`; do not
vendor a copy into this repository. Keep tenferro-specific durable rules in
`REPOSITORY_RULES.md`. This `AGENTS.md` file is the entry point and
tenferro-specific orientation; avoid duplicating the detailed performance,
layout, CPU kernel, slicing, cache, threading, and GPU backend rules here.

## Contribution Workflow Assets

Use repository-local contribution workflows when preparing external-facing
issues or bug-fix PRs:

- `ai/contribution-workflows/issue-intake.md` for bug reports, feature
  requests, design discussions, and documentation or article topic issues.
- `ai/contribution-workflows/bugfix-pr.md` for pull requests that fix existing
  intended behavior.
- `ai/contribution-workflows/repository-remediation.md` for batched,
  agent-assisted remediation of repository-rule violations across multiple
  related issues or findings.
- `ai/contribution-workflows/release-publish.md` for maintainer releases:
  workspace version bumps, tagging, dependency-order crates.io publication,
  and post-publish provenance verification.

Do not open a new-feature implementation PR before maintainers accept the
corresponding issue. If a proposed bug-fix PR needs a new public API, operation
family, backend, dependency, feature flag, architectural layer, or AD semantics
change, stop the PR path and use issue intake.

For batched repository-rule remediation work, follow
`ai/contribution-workflows/repository-remediation.md`. That workflow is a
deliberate exception to the normal one-bug-fix-PR path: collect all local fixes
and verification before opening a PR, keep coherent commits in a single PR, and
do not use squash merge.

Thin tool adapters live in `.agents/skills/`, `.claude/skills/`,
`.opencode/commands/`, and `.kimi/skills/`. Keep policy in `CONTRIBUTING.md` and
`REPOSITORY_RULES.md`; keep reusable workflow steps in
`ai/contribution-workflows/`.
See `ai/README.md` for the repository-local AI workflow layout.

### GPU Status

CUDA GPU support is implemented through the feature-gated CubeCL backend across
the concrete tensor, eager, and traced execution surfaces. Performance
optimization is still active work. The remaining CUDA limitations are specific:
`eig`, `full_piv_lu`, `full_piv_lu_solve`, `dynamic_update_slice`, integer
numeric/linalg gaps beyond add/mul/compare/select and sum/prod reductions, and
selected complex analytic or ordering operations.
HIP/ROCm remains stubbed. Outside explicit GPU implementation tasks, check
`docs/guides/devices-and-gpu.md` and the current CUDA/CubeCL backend tests
before assuming a specific op/dtype/backend combination is supported.

### Documentation Requirements

Every public type, trait, and function **must** include minimal but sufficient usage examples in its doc comments (`/// # Examples`). The examples should help a human quickly understand how to use the API. Doc examples must compile and run as doctests; do not use `ignore` or `no_run`. Crate-level docs (`//!`) should include typical end-to-end usage examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU placement metadata
- Graph-based traced execution via `TracedTensor` + `GraphExecutor`
- Standard extension crates for operation families such as einsum, linalg, and FFT
- Automatic differentiation (VJP/JVP/HVP) for the standard dense numeric path
- Single execution IR (`ExecOp`) plus a pass pipeline for backend dispatch

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [`docs/design/`](docs/design/) for architecture and design documents.

### Work Logs And Review Intent

For nontrivial refactors, cleanup streams, AI-assisted implementation, or
changes that make explicit design tradeoffs, check [`docs/worklogs/`](docs/worklogs/)
before reviewing the code. Work logs record the session summary, context read,
reference code, chosen design, rejected alternatives, and residual risks. A
review that challenges scope, abstraction, or design intent should engage with
the linked work log instead of treating the diff as context-free.

When a PR establishes or changes durable design intent, update the appropriate
document under [`docs/design/`](docs/design/) in the same PR. Work logs explain
why a session made a decision; design docs record decisions future work should
continue to follow.

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

See `REPOSITORY_RULES.md` for the authoritative tenferro file-organization
rules. Treat **~1000 lines** as a soft review trigger, not a mechanical
split requirement; split only along clear behavior, abstraction, feature,
ownership, or public/private API boundaries.

### Test Coverage Target

Every source file should have **90%+ line coverage**. When adding new code,
add tests that cover the new paths. When modifying existing code, check
coverage for the modified file and add tests if below 90%.

### Unit Test Organization

See `REPOSITORY_RULES.md` for the authoritative tenferro unit-test
organization rules, including inline `#[cfg(test)]` block restrictions and
module-local test file placement.

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
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/repository-rules-review.json
```

If `cargo fmt --all --check` fails, run `cargo fmt --all` to fix formatting automatically.
Run the local LLM review on the committed PR head; `--worktree` is acceptable
only as an earlier preview, and must be rerun without `--worktree` before PR
creation.

Additionally, verify the following before pushing:

- **Clippy parity**: Run clippy locally with the same command and options as the repository CI `clippy` job; do not use a relaxed local variant.
- **Side review**: Re-read `REPOSITORY_RULES.md` and review the local diff against repository rules before creating a PR. Fix any findings, or explicitly document residual risks.
- **Sample code verification**: All code examples in `README.md` and `docs/getting-started/` must compile and run correctly. Extract and test any changed examples.
- **Design document updates**: When code changes affect architecture or specifications, update the corresponding documents in `docs/architecture/`, `docs/spec/`, or `docs/design/`, and update any affected diagrams under `docs/assets/` or embedded in Markdown. Stale documentation is worse than no documentation.
- **Work log updates**: For nontrivial refactors, cleanup streams, AI-assisted implementation, or explicit tradeoff decisions, add or update a work log under `docs/worklogs/` and link it from the PR body.

### PR Creation Rules

- PRs to `main` must be created using `gh pr create`
- Do not include AI-generated analysis reports as standalone files in PRs
- Batched repository-rule remediation PRs must follow
  `ai/contribution-workflows/repository-remediation.md`: open one PR only after
  all local fixes and verification are complete, keep coherent commits, and
  merge with a non-squash method.
- For ordinary non-remediation PRs, enable auto-merge after creating a PR:
  `gh pr merge --auto --squash --delete-branch`
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

# GPU (CUDA/CubeCL) tests — requires NVIDIA GPU + CUDA 12.8+
# Set CUBECL_DEBUG_LOG=0 to suppress verbose JIT compilation logs.
# GPU tests are marked #[ignore] so they don't fail on non-GPU machines.
# Use --ignored to actually run them.
# If `/usr/local/cuda-12.8` is absent, find an installed CUDA root with `ls -d /usr/local/cuda*`.
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored
```

### CubeCL Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUBECL_DEBUG_LOG` | `0` | Suppress JIT compilation log output (default is verbose) |
| `CUDA_PATH` | `/usr/local/cuda-12.8` | CUDA toolkit root for NVRTC header resolution |
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

### Device Transfer And CUDA Limits

See `REPOSITORY_RULES.md` for the authoritative device-transfer, backend
buffer error, and CUDA library limitation contracts. See
`docs/guides/devices-and-gpu.md` for user-facing examples.

## Workspace Architecture

### Layered Design

```
Layer 4: tenferro-ad       - Eager runtime, eager tensors, traced AD extension traits
Layer 3: tenferro-runtime  - Concrete tensor helpers, traced tensors, graph compile/exec,
                             extension runtime registration, extension cache storage
         tenferro-einsum   - Subscripts, contraction planning, traced/eager einsum APIs,
                             extension runtime, AD rule
         tenferro-linalg   - Linear algebra traced APIs, eager helpers, extension runtime,
                             optional linalg AD rules
         tenferro-fft      - FFT extension runtime and public FFT APIs
Layer 2: tenferro-tensor   - Dense runtime tensors, backend traits,
                             backend-independent contracts
         tenferro-cpu      - CPU backend, execution sessions, kernels,
                             buffer pools
         tenferro-gpu      - CubeCL/CUDA backend and GPU transfer helpers
Layer 1: tenferro-tensor-core - Host-only tensor data model, dtype tags,
                                scalar trait, metadata-only views
Internal: tenferro-core-ops  - Internal core primitive operation catalog
          tenferro-internal-ops - Graph op vocabulary and AD rule implementations
          tenferro-internal-extension-macros - Extension-op registration macros
```

`tidu-rs` defines the generic `Primitive` AD contract, independent of any
tensor type, and owns graph transforms such as `linearize` and
`linear_transpose`.
`tenferro-tensor-core` owns the lightweight host tensor data model.
`tenferro-tensor` owns concrete dense runtime value types and
backend-independent tensor contracts. `tenferro-cpu` owns CPU backend
execution. `tenferro-gpu` owns CubeCL/CUDA backend code and transfer helpers.
`tenferro-internal-ops/src/ad/` is the semantic source of truth for core
primitive AD rules. `tenferro-runtime` owns traced graph construction,
lowering, execution, extension registration, and cache ownership. `tenferro-ad`
owns eager AD surfaces and traced AD helper APIs.

The workspace intentionally has no root `tenferro` facade crate. Standard
operation families remain separately imported crates such as
`tenferro_einsum`, `tenferro_linalg`, and `tenferro_fft`.

See `docs/architecture/tenferro-crates.md` for the full crate role table and
dependency-boundary rules.

## AI Workflow Scripts

Repository-local headless launchers live under `ai/`:

- `ai/run-codex-solve-bug.sh`
- `ai/run-claude-solve-bug.sh`

These scripts resolve their prompt path relative to `ai/`, but they always run
the agent from the repository top-level directory. Their default prompt is
`ai/solve_bug_issue.md`, and JSON output is the default mode unless `--text` is
passed.

### Dependency Graph

```
tenferro-tensor-core
    |
    v
tenferro-tensor <---------------- tenferro-gpu
    |                                  ^
    |                                  |
    v                                  |
tenferro-internal-ops <-------- tenferro-einsum
    |                         \-- tenferro-linalg
    |                         \-- tenferro-fft
    v
tenferro-runtime
    |
    v
tenferro-ad

tenferro-core-ops is internal metadata used by tensor, runtime, GPU, and ops.
tenferro-internal-extension-macros is used by operation-family crates.
```
