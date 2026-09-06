# AGENTS.md

Guidance for Claude Code (claude.ai/code) and other agents working in this repository.

Before acting, read the latest shared tensor4all agent rules online, starting
from `https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If internet access is unavailable or the remote cannot be resolved, use the
sibling checkout `../tensor4all-agent-rules/rules/index.md`. Load only the common, Rust,
performance, numerical, docs, or benchmark rule files relevant to the task. If
neither source is available, continue from this repository's rules and state
that the shared rules were unavailable when creating a PR.

Then read `REPOSITORY_RULES.md`, the tenferro-specific rules. Review it again
before implementation work, before creating a PR, and before touching AD rules,
oracle replay, or linearized boundary code. Read `PERFORMANCE_TIPS.md` in full
before implementing or reviewing performance-sensitive code (tensor kernels,
graph/compiler planning, caches, GPU kernels, benchmarks, examples) and before
creating a PR that touches it. The sections below are tenferro-specific
additions and overrides.

## Current Implementation Status

Active implementations coexist with evolving APIs. Implementation work is
allowed unless a task says otherwise.

## Proportionate Safety And Validation

Tenferro is scientific-computing software, not a security product or trust
boundary. Preserve Rust memory safety, aliasing and lifecycle soundness,
numerical correctness, reproducibility, and explicit device behavior. Do not add
security machinery against a malicious maintainer, checkout, local build tool,
or CI runner unless a concrete task introduces that untrusted boundary.

Prefer the simplest design that covers reachable failures. Every additional
validation step should name the correctness failure or operational mistake it
prevents. Avoid cryptographic anti-tamper protocols,
nonce/challenge handshakes, redundant identity checks, and repeated validation
that only guards against trusted local tooling forging its own output.

An exact Git commit plus repository-relative path identifies a tracked artifact
when execution uses a tracked worktree clean against that commit. Add a content
checksum only at a concrete untracked or cross-system boundary where Git identity
is unavailable or insufficient, and document that boundary. These rules do not
relax validation required before unsafe memory access, mutable aliasing,
asynchronous device retirement, or numerical execution.

## Repository Rule Source

Cross-repository rules live in `tensor4all-agent-rules`; do not vendor a copy.
Tenferro-specific durable rules live in `REPOSITORY_RULES.md`; performance,
layout, cache, threading, and Faer contracts live in `PERFORMANCE_TIPS.md`.
This file is the entry point and orientation; do not duplicate the detailed
performance, layout, CPU kernel, slicing, cache, threading, or GPU backend
rules here. Both rule files are routing input for
`scripts/repository-rules-review.py`: when adding or renaming a `##` section
in either, update that script's routed or human-only section lists in the same
change.

## Contribution Workflow Assets

Use the repository-local workflows for external-facing issues and PRs:

- `ai/contribution-workflows/issue-intake.md`: bug reports, feature requests,
  design discussions, documentation or article topic issues.
- `ai/contribution-workflows/bugfix-pr.md`: PRs that fix existing intended behavior.
- `ai/contribution-workflows/repository-remediation.md`: batched, agent-assisted
  remediation of repository-rule violations across related issues or findings.
- `ai/contribution-workflows/release-publish.md`: maintainer releases (version
  bumps, tagging, dependency-order crates.io publication, post-publish
  provenance verification).
- `ai/agent-workflow-lessons.md`: model-independent process self-audits during
  long multi-phase implementation work.

Do not open a new-feature implementation PR before maintainers accept the
issue. If a bug-fix PR needs a new public API, operation family, backend,
dependency, feature flag, architectural layer, or AD semantics change, stop and
use issue intake.

Batched remediation follows `repository-remediation.md`, a deliberate exception
to the one-bug-fix-PR path: finish all local fixes and verification before
opening a PR, keep coherent commits in a single PR, and do not squash merge.

Thin tool adapters live in `.agents/skills/`, `.claude/skills/`,
`.opencode/commands/`, and `.kimi/skills/`. Policy lives in `CONTRIBUTING.md`
and `REPOSITORY_RULES.md`; reusable workflow steps live in
`ai/contribution-workflows/`. See `ai/README.md` for the layout.

### GPU Status

CUDA support is implemented through the feature-gated CubeCL backend across the
concrete tensor, eager, and traced surfaces; performance work is ongoing.
Remaining CUDA limitations: `eig`, `full_piv_lu`, `full_piv_lu_solve`,
`dynamic_update_slice`, integer numeric/linalg gaps outside the supported
add/sub/mul/div/rem, neg/abs/sign/pow, comparison/selection/minimum/maximum, and
sum/product/minimum/maximum reductions, `Bool` arithmetic/reduction/linalg and
additive-scatter gaps, and selected complex analytic or ordering operations.
HIP/ROCm is stubbed. Outside explicit GPU tasks, check
`docs/guides/devices-and-gpu.md` and the CUDA/CubeCL backend tests before
assuming an op/dtype/backend combination is supported.

### Documentation Requirements

Every public type, trait, and function **must** carry minimal but sufficient
usage examples (`/// # Examples`) that compile and run as doctests; no `ignore`
or `no_run`. Crate-level docs (`//!`) include typical end-to-end examples.

## Project Overview

**tenferro-rs** is a general-purpose tensor computation library in Rust (`tenferro-*` crates):
- Dense tensor types with CPU/GPU placement metadata
- Graph-based traced execution via `TracedTensor` + `GraphCompiler` + `Runtime::run_compiled`
- Extension crates for operation families such as einsum, linalg, and FFT
- Automatic differentiation (VJP/JVP/HVP) for the standard dense numeric path
- Runtime-owned execution preparation and backend/provider dispatch

**strided-rs** (separate workspace) is an external foundation dependency:
- `strided-traits`: `ScalarBase`, `ElementOp` traits
- `strided-view`: dynamic-rank strided views (`StridedView`/`StridedViewMut`)
- `strided-kernel`: cache-optimized map/reduce/broadcast kernels

tenferro-rs depends on strided-rs but does not absorb it; strided-rs has no BLAS
dependency and works standalone.

### Design Documents

See [`docs/design/`](docs/design/).

### Work Logs And Review Intent

For nontrivial refactors, cleanup streams, AI-assisted implementation, or
explicit design tradeoffs, read [`docs/worklogs/`](docs/worklogs/) before
reviewing code. Work logs record session summary, context read, reference code,
chosen design, rejected alternatives, and residual risks. A review challenging
scope, abstraction, or design intent should engage with the linked work log.

When a PR establishes or changes durable design intent, update the relevant
document under [`docs/design/`](docs/design/) in the same PR. Work logs explain
why a session decided; design docs record decisions future work should continue to follow.

**Note**: `docs/plans/` holds historical records that may contradict the current
API or design. Do not update them to match the current state.

## Performance, Layout, And Backend Rules

`PERFORMANCE_TIPS.md` is authoritative for column-major ordering, hidden
materialization, range checks and slicing, anti-patterns, Faer integration,
cache ownership, CPU threading, and the performance-gated experiment protocol;
`REPOSITORY_RULES.md` keeps CPU kernel ownership, device transfer, and GPU
backend conventions. Read `PERFORMANCE_TIPS.md` in full before
performance-sensitive implementation or review. The `audit-performance` skill
runs its static audit procedure over a path or the whole repository.

Do not assume nearby code is performance-correct; active optimization work and
legacy tradeoffs remain. Before copying patterns in tensor kernels,
graph/compiler planning, caches, GPU kernels, benchmarks, or user-facing
examples, check for hidden materialization, repeated per-element index work,
avoidable large-state clones, repeated linear scans, accidental single-thread GPU
execution, and weak shape-only validation. Improve the pattern or document the
residual risk instead of propagating it.

## Code Style

- `python3 scripts/ci/run_profile.py fmt` checks formatting across the root
  workspace and standalone extension manifests
- No `unwrap()`/`expect()` in library code
- `thiserror` for public API error types

### File Organization

`REPOSITORY_RULES.md` is authoritative. **~1000 lines** is a soft review trigger,
not a mechanical split requirement; split only along clear behavior,
abstraction, feature, ownership, or public/private API boundaries.

### Test Coverage Target

**90%+ line coverage** per source file. Cover new paths; when modifying a file
below 90%, add tests.

### Unit Test Organization

`REPOSITORY_RULES.md` is authoritative, including inline `#[cfg(test)]`
restrictions and module-local test file placement.

### ASCII Diagrams

- Use **uniform inner width** for all boxes in one diagram
- **No nested boxes**; they misalign easily
- Verify character counts between `│` delimiters match the dash count in `┌───┐` / `└───┘` borders

### Dependencies

Use **workspace dependencies** for libraries shared across crates: define once
under `[workspace.dependencies]`, reference with `dep.workspace = true`.

### Crates.io Publication

- Never publish a package that does not already exist on crates.io without the
  user's explicit approval for that specific package. A general request to
  finish a PR or release does not authorize it; never publish new packages
  automatically.
- Before every `cargo publish`, inspect the packaged files and validate the
  metadata: at least name, version, description, license, repository, homepage,
  documentation, README, `rust-version`, keywords, categories, and
  `include`/`exclude`. Run `cargo package` checks before requesting or acting on
  publication approval.

## Git Worktree Rules

**Always branch from the latest `main`**:
`git fetch origin && git checkout -b <branch-name> origin/main`. Never branch
from stale local state or another feature branch unless explicitly intended.

## Pre-Push / PR Checklist

Before pushing or creating a PR with code changes, run focused non-release
verification through the local gate. The code-change path also runs CI-parity
clippy for the root workspace and the tropical and sparse extension manifests:

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
Rust test or coverage acknowledgement. CI-only changes need a focused CI helper
command through `--test`. The default dev/test profiles use `opt-level=0`,
`debug=0`, and `incremental=true`, keeping debug assertions and overflow checks.

Hosted CI owns complete workspace tests, coverage enforcement, backend matrix,
docs-site builds, GPU validation, and clean builds through workspace
`[profile.ci]` (`opt-level=0`, `debug=0`, `incremental=false`,
`strip="symbols"`). Do not require those commands locally before every PR.

Run the relevant release test or benchmark locally when a change is
performance-sensitive, reproduces a release-only bug, touches unsafe or
optimization-sensitive behavior, or a maintainer requests it.

If formatting fails, run `cargo fmt --all`,
`cargo fmt --manifest-path ext/tropical/Cargo.toml --all`, and
`cargo fmt --manifest-path ext/sparse/Cargo.toml --all`. Run the repository-rules
review on the committed PR head; the external LLM review is permanently
disabled, so pass `--dry-run --llm-skipped-reason "local deterministic review"`.
`--worktree` is acceptable only as an earlier preview and must be rerun without
it before PR creation.

Also verify before pushing:

- **Side review**: re-read `REPOSITORY_RULES.md`, plus `PERFORMANCE_TIPS.md` when the diff touches performance-sensitive code, and review the diff against them. Fix findings or document residual risks.
- **Sample code verification**: all examples in `README.md` and `docs/getting-started/` compile and run. Extract and test changed examples.
- **Design document updates**: when code changes affect architecture or specifications, update `docs/architecture/`, `docs/spec/`, or `docs/design/`, plus affected diagrams under `docs/assets/` or embedded in Markdown. Stale documentation is worse than none.
- **Agent skill freshness**: when a PR changes public API surface, feature flags, crate boundaries, or documented idioms, review `.agents/skills/tenferro-compute/` and the other shipped skill mirrors, and update them in the same PR when they no longer match.
- **Work log updates**: for nontrivial refactors, cleanup streams, AI-assisted implementation, or explicit tradeoffs, add or update a work log under `docs/worklogs/` and link it from the PR body.

### Local Rust Build Acceleration

Ordinary focused local development, including AI-assisted edit-test loops,
should use Cargo incremental compilation through the default dev/test profiles.
Do not recommend or enable `sccache` solely for these loops. Enable debugger symbols for one command
with `CARGO_PROFILE_DEV_DEBUG=1` or `CARGO_PROFILE_TEST_DEBUG=1`.

- Do not install sccache or edit global Cargo configuration without explicit
  user approval.
- Do not disable incremental compilation in the default dev/test profiles or set
  `RUSTC_WRAPPER=sccache` globally.
- Developer-local cache only; no shared remote sccache.
- Correctness checks and PR gates must work on a cache miss.
- Disable sccache for clean-build measurements and report whether each timing
  used a cold or warm cache.

### PR Creation Rules

- Create PRs to `main` with `gh pr create`
- No AI-generated analysis reports as standalone files in PRs
- Batched remediation PRs follow
  `ai/contribution-workflows/repository-remediation.md`: one PR after all local
  fixes and verification, coherent commits, non-squash merge.
- For ordinary PRs, enable auto-merge after creation:
  `gh pr merge --auto --squash --delete-branch`
- `createpr` must confirm auto-merge remains enabled and the required branch protection checks are still configured

## Build Commands

Repository scripts require Python 3.11 or newer. Set `$PYTHON` to one executable or path token to override the resolver.

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

# Coverage check (per-file thresholds, target 90%+ line coverage per file)
cargo llvm-cov --workspace --profile ci --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json

# Build rustdoc and docs site inputs
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py

# GPU (CUDA/CubeCL) tests: requires NVIDIA GPU + CUDA 12.4+ (12.8+ enables the full CubeCL feature set).
# GPU tests are #[ignore]; --ignored runs them. CUBECL_DEBUG_LOG=0 suppresses JIT logs.
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

Set these in CI and local dev shells. Without `CUBECL_DEBUG_LOG=0`, cubecl emits
generated CUDA source for every JIT-compiled kernel, millions of log lines per
test run.

### FFI Library Path Configuration

The CubeCL backend loads cuTENSOR, cuSOLVER, and cuBLAS at runtime via `dlopen`,
trying v12, then v11, then the bare soname. Override with:

| Variable | Library | Example |
|----------|---------|---------|
| `TENFERRO_CUTENSOR_PATH` | cuTENSOR | `/opt/cuda-12.4/lib64/libcutensor.so.2` |
| `TENFERRO_CUSOLVER_PATH` | cuSOLVER | `/opt/cuda-12.4/lib64/libcusolver.so.12` |
| `TENFERRO_CUBLAS_PATH` | cuBLAS | `/opt/cuda-12.4/lib64/libcublas.so.12` |

Colon-separated paths are supported (like `LD_LIBRARY_PATH`).

### Device Transfer And CUDA Limits

`REPOSITORY_RULES.md` is authoritative for device transfer, backend buffer
errors, and CUDA library limitations; `docs/guides/devices-and-gpu.md` has
user-facing examples.

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

Ownership: `tenferro-tensor-core` owns the lightweight host tensor data model.
`tenferro-tensor` owns concrete dense runtime value types and
backend-independent tensor contracts. `tenferro-cpu` owns CPU backend
execution. `tenferro-gpu` owns CubeCL/CUDA backend code and transfer helpers.
`tenferro-internal-ops/src/ad/` is the semantic source of truth for the core
`StdTensorOp` primitive AD rules. `tenferro-runtime` owns traced graph
construction, lowering, execution, extension registration, and cache ownership.
`tenferro-ad` owns eager/traced AD surfaces, traced AD helper APIs, semantic
extension AD rules, and context-owned AD transform caching.

There is intentionally no root `tenferro` facade crate; operation families stay
separately imported (`tenferro_einsum`, `tenferro_linalg`, `tenferro_fft`).

See `docs/architecture/tenferro-crates.md` for the full crate role table and
dependency-boundary rules.

## AI Workflow Scripts

Headless launchers under `ai/`:

- `ai/run-codex-solve-bug.sh`
- `ai/run-claude-solve-bug.sh`

They resolve the prompt path relative to `ai/` but run the agent from the
repository top level. Default prompt: `ai/solve_bug_issue.md`; JSON output is
the default unless `--text` is passed.

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
