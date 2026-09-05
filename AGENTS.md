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

Read the applicable sections of `REPOSITORY_RULES.md` when starting work and
revisit them when scope changes, particularly for AD, unsafe or FFI boundaries.
Do not reload unchanged rules before every edit or PR step.

### Proportionate workflow

Work directly by default. Subagents and independent AI/cross-model reviews are
optional and require an explicit user request. Review the coherent final diff
once; corrections within its agreed design need affected checks, not a fresh
design approval or full review cycle. Required CI and human approvals are unchanged.

## Current Implementation Status

The workspace contains active implementations alongside evolving APIs. Implementation work is allowed unless a task explicitly says otherwise.

## Proportionate Safety And Validation

Tenferro is scientific-computing software, not a security product or a trust
boundary. Preserve Rust memory safety, aliasing and lifecycle soundness,
numerical correctness, reproducibility, and explicit device behavior, but do
not add security machinery for a malicious maintainer, repository checkout,
local build tool, or CI runner unless a concrete task explicitly introduces
that untrusted boundary.

Prefer the simplest design that covers reachable failures. Every additional
validation step should name the correctness failure or operational mistake it
prevents. Avoid cryptographic anti-tamper protocols, nonce/challenge
handshakes, redundant identity checks, and repeated validation whose only
purpose is defending against trusted local tooling forging its own output.

For a tracked artifact, an exact Git commit plus repository-relative path is
normally sufficient to identify its contents when execution uses a tracked
worktree that is clean against that commit. Add a content checksum only at a
concrete untracked or cross-system artifact boundary where Git identity is
unavailable or insufficient, and document that boundary. These proportionality
rules do not relax validation required before unsafe memory access, mutable
aliasing, asynchronous device retirement, or numerical execution.

## Repository Rule Source

Keep cross-repository implementation rules in `tensor4all-agent-rules`; do not
vendor a copy into this repository. Keep tenferro-specific durable rules in
`REPOSITORY_RULES.md`. This `AGENTS.md` file is the entry point and
tenferro-specific orientation; avoid duplicating the detailed performance,
layout, CPU kernel, slicing, cache, threading, and GPU backend rules here.
`REPOSITORY_RULES.md` is also routing input for
`scripts/repository-rules-review.py`; when adding or renaming a `##` section,
update that script's routed or human-only section lists in the same change.

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
- `ai/agent-workflow-lessons.md` for model-independent process self-audits
  during long-running, multi-phase implementation work.

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
numeric/linalg gaps outside the currently supported add/sub/mul/div/rem,
neg/abs/sign/pow, comparison/selection/minimum/maximum, and
sum/product/minimum/maximum reductions, `Bool` arithmetic/reduction/linalg and
additive-scatter gaps, and selected complex analytic or ordering operations.
HIP/ROCm remains stubbed. Outside explicit GPU implementation tasks, check
`docs/guides/devices-and-gpu.md` and the current CUDA/CubeCL backend tests
before assuming a specific op/dtype/backend combination is supported.

### Documentation Requirements

Every public type, trait, and function **must** include minimal but sufficient usage examples in its doc comments (`/// # Examples`). The examples should help a human quickly understand how to use the API. Doc examples must compile and run as doctests; do not use `ignore` or `no_run`. Crate-level docs (`//!`) should include typical end-to-end usage examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates). It provides:
- Dense tensor types with CPU/GPU placement metadata
- Graph-based traced execution via `TracedTensor` + `GraphCompiler` + `Runtime::run_compiled`
- Standard extension crates for operation families such as einsum, linalg, and FFT
- Automatic differentiation (VJP/JVP/HVP) for the standard dense numeric path
- Runtime-owned execution preparation and backend/provider dispatch

**strided-rs** (separate workspace) is an external foundation dependency providing:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: Dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: Cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it. strided-rs has no BLAS dependency and can be used standalone.

### Design Documents

See [`docs/design/`](docs/design/) for architecture and design documents.

### Work Logs And Review Intent

Use [`docs/worklogs/`](docs/worklogs/) for multi-phase work or non-obvious design
tradeoffs. Small fixes can keep their rationale and checks in the PR body; AI
assistance alone does not require a separate log. Consult a linked decision
record when reviewing the choices it explains, rather than duplicating it.

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

- `python3 scripts/ci/run_profile.py fmt` for repository formatting checks
  across the root workspace and standalone extension manifests
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

### Crates.io Publication

- Never publish a package that does not already exist on crates.io without the
  user's explicit approval for that specific package. A general request to
  finish a PR or release does not authorize publishing a new package, and
  agents must never publish new packages automatically.
- Before every `cargo publish`, inspect the packaged files and validate the
  package metadata for the intended audience. This includes at least the name,
  version, description, license, repository, homepage, documentation, README,
  `rust-version`, keywords, categories, and `include`/`exclude` rules. Run the
  appropriate `cargo package` checks before requesting or acting on publication
  approval.

## Git Worktree Rules

When using git worktrees for feature development, **always branch from the latest `main`** before starting implementation. Run `git fetch origin && git checkout -b <branch-name> origin/main` to ensure the branch is up-to-date. Never branch from a stale local state or from another feature branch unless explicitly intended.

## Pre-Push / PR Checklist

Before pushing or creating a pull request with code changes, run focused
non-release verification through the local gate. The code-change path also runs
CI-parity clippy for the root workspace and the standalone tropical and sparse
extension manifests:

```bash
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'cargo test -p tenferro-tensor checked_convert_follows_dtype_promotion_lattice'

python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/repository-rules-review.json
```

For documentation-only changes, run `bash scripts/check-pr-fast.sh` without a
Rust test or coverage acknowledgement. CI-only changes require a focused CI
helper command through `--test`. The default dev/test profiles use
`opt-level=0`, `debug=0`, and `incremental=true`, while preserving debug
assertions and overflow checks.

Hosted CI owns complete workspace tests, coverage enforcement, backend matrix,
docs-site builds, GPU validation, and clean builds through workspace
`[profile.ci]` (`opt-level=0`, `debug=0`, `incremental=false`,
`strip="symbols"`). Do not require those comprehensive hosted-CI commands
locally before every PR.

Run the relevant release test or benchmark locally when a change is
performance-sensitive, reproduces a release-only bug, touches unsafe or
optimization-sensitive behavior, or a maintainer explicitly requests it.

If the formatting step fails, run `cargo fmt --all`, then
`cargo fmt --manifest-path ext/tropical/Cargo.toml --all` and
`cargo fmt --manifest-path ext/sparse/Cargo.toml --all` to fix formatting
automatically. Run the local repository-rules review on the committed PR
head (the external LLM review is permanently disabled, so run it with
`--dry-run --llm-skipped-reason "local deterministic review"`);
`--worktree` is acceptable only as an earlier preview, and must be rerun
without `--worktree` before PR creation.

Additionally, verify the following before pushing:

- **Self-review**: Apply relevant repository rules to the coherent final diff once. Fix findings or document residual risks; do not repeat unchanged reviews.
- **Sample code verification**: All code examples in `README.md` and `docs/getting-started/` must compile and run correctly. Extract and test any changed examples.
- **Design document updates**: When code changes affect architecture or specifications, update the corresponding documents in `docs/architecture/`, `docs/spec/`, or `docs/design/`, and update any affected diagrams under `docs/assets/` or embedded in Markdown. Stale documentation is worse than no documentation.
- **Agent skill freshness**: When a PR changes public API surface, feature flags, crate boundaries, or documented idioms, review `.agents/skills/tenferro-compute/` and the other shipped skill mirrors, and update them in the same PR when they no longer match.
- **Work log updates**: Link a concise record for multi-phase work or non-obvious tradeoffs. For small fixes, rationale and checks in the PR body are sufficient.

### Local Rust Build Acceleration

Ordinary focused local development, including AI-assisted edit-test loops,
should use Cargo incremental compilation through the default dev/test profiles.
Do not recommend or enable `sccache` solely for these loops. Debugger symbols
may be enabled for one command with `CARGO_PROFILE_DEV_DEBUG=1` or
`CARGO_PROFILE_TEST_DEBUG=1`.

- Do not install sccache or edit global Cargo configuration without explicit
  user approval.
- Do not disable incremental compilation in the default dev/test profiles or
  set `RUSTC_WRAPPER=sccache` globally as a general local-development default.
- Use developer-local cache only. Do not introduce or recommend a shared remote
  sccache.
- Correctness checks and PR gates must work on a cache miss.
- Disable sccache for clean-build measurements and report whether each timing
  used a cold or warm cache.

### PR Creation Rules

- Minimize PR count for one agreed deliverable. Keep implementation steps and
  focused tests small, but collect related changes and validate them before
  opening a PR. Do not create a PR per task or correction; update the existing PR.
- Split only for real repository/dependency, reviewability, or independent
  release/rollback boundaries, or an explicit user request. Do not bundle
  unrelated changes merely to reduce PR count.
- Batch ready corrections before pushing to avoid redundant CI runs; preserve
  required CI and approvals for the final submitted state.

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

Repository scripts require Python 3.11 or newer. Set `$PYTHON` to one executable or path token to override the resolver when needed.

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

# Run a focused incremental local PR check
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'cargo test -p tenferro-tensor checked_convert_follows_dtype_promotion_lattice'

# Check formatting
python3 scripts/ci/run_profile.py fmt

# Coverage check (per-file thresholds)
# Target: 90%+ line coverage per file. Files below 90% should have tests added.
cargo llvm-cov --workspace --profile ci --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json

# Build rustdoc and docs site inputs
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py

# GPU (CUDA/CubeCL) tests — requires NVIDIA GPU + CUDA 12.4+
# Set CUBECL_DEBUG_LOG=0 to suppress verbose JIT compilation logs.
# GPU tests are marked #[ignore] so they don't fail on non-GPU machines.
# Use --ignored to actually run them.
# CUDA 12.8+ enables the full CubeCL feature set; CUDA 12.4 is the baseline.
# Find the installed CUDA root with `ls -d /usr/local/cuda*`.
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.4 \
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored
```

### CubeCL Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CUBECL_DEBUG_LOG` | `0` | Suppress JIT compilation log output (default is verbose) |
| `CUDA_PATH` | `/usr/local/cuda-12.4` or newer | CUDA toolkit root for NVRTC header resolution |
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

`tenferro-internal-ops/src/ad/` defines the core `StdTensorOp` primitive AD
rules. `tenferro-ad` owns eager/traced AD surfaces, semantic extension AD
rules, and context-owned AD transform caching.
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

tenferro-gpu              -> tenferro-cpu
tenferro-fft              -> tenferro-cpu
```

See [the complete crate map](docs/architecture/tenferro-crates.md) for the
complete direct-edge inventory.
