# Issue #441 Completion Replan Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish `#441` on top of the latest `origin/main` by completing the remaining scalar/analytic surface work, splitting the remaining linalg AD monoliths, and keeping all high-level surfaces CPU/GPU-generic without adding custom CUDA kernels.

**Architecture:** Treat the rest of `#441` as a foundation-cleanup PR, not a CUDA parity PR. `extension/tenferro-dyadtensor` remains a wiring layer that delegates primal execution to `tenferro-prims`, `tenferro-einsum`, and `tenferro-linalg`; no new high-level API should mention `CpuContext`, `CpuBackend`, or backend names directly. CPU implementations should use `strided-rs` kernels where available, shared macros/helpers should be preferred over repetitive per-op code, and unsupported GPU paths should fail through truthful capability queries rather than CPU-only shortcuts.

**Tech Stack:** Rust workspace crates, `tenferro-prims`, `tenferro-linalg`, `extension/tenferro-dyadtensor`, `extern/chainrules-scalarops`, `strided-rs`, workspace docs and verification scripts.

---

### Task 0: Sync the branch onto the latest `origin/main`

**Files:**
- Modify: `tenferro-linalg/src/tests/organization.rs`

**Step 1: Checkpoint the currently dirty split-test change**

The branch is currently dirty because `tenferro-linalg/src/tests/organization.rs`
already has the beginnings of the `rrules`/`frules` split checks.

Run:

```bash
git status --short
git add tenferro-linalg/src/tests/organization.rs
git commit -m "test: prepare linalg rule module split checks"
```

Expected:

- the worktree becomes clean
- the new commit contains only the organization-test checkpoint

**Step 2: Fast-forward local `main` and rebase `implement-441`**

Run:

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tenferro-rs checkout main
git -C /sharehome/shinaoka/projects/tensor4all/tenferro-rs fetch origin
git -C /sharehome/shinaoka/projects/tensor4all/tenferro-rs merge --ff-only origin/main
git rebase origin/main
```

Expected:

- root `main` matches `origin/main`
- `implement-441` is rebased onto the latest merged docs/rule sync changes

**Step 3: Resolve any rebase conflicts immediately**

If conflicts appear, resolve them before continuing. Prioritize:

- keeping the latest docs/rule-vendor sync from `origin/main`
- preserving the `#441` runtime-generic and file-split work from this branch

**Step 4: Re-run a minimal checkpoint test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg result_types_is_split_into_focused_modules -- --nocapture
```

Expected: PASS

**Step 5: Commit any conflict resolution**

If conflict resolution changed files beyond the rebase replay, commit them:

```bash
git add <resolved-files>
git commit -m "chore: resolve origin main rebase for issue 441"
```

### Task 1: Split `tenferro-linalg` reverse rules into focused modules

**Files:**
- Delete: `tenferro-linalg/src/rrules.rs`
- Create: `tenferro-linalg/src/rrules/mod.rs`
- Create: `tenferro-linalg/src/rrules/svd_qr.rs`
- Create: `tenferro-linalg/src/rrules/lu_eigen.rs`
- Create: `tenferro-linalg/src/rrules/least_squares.rs`
- Create: `tenferro-linalg/src/rrules/linear_systems.rs`
- Create: `tenferro-linalg/src/rrules/spectral.rs`
- Create: `tenferro-linalg/src/rrules/matrix_functions.rs`
- Create: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/src/tests/organization.rs`

**Step 1: Keep the failing organization tests first**

Make sure the organization tests assert the new directory layout and file-size
guideline before moving any code.

Representative test names:

```rust
#[test]
fn rrules_is_split_into_focused_modules() {}

#[test]
fn split_rrule_modules_stay_under_size_guideline() {}
```

**Step 2: Move the rule functions without changing math**

Split by family:

- `svd_qr.rs`: `svd_rrule`, `qr_rrule`
- `lu_eigen.rs`: `lu_rrule`, `eigen_rrule`
- `least_squares.rs`: `lstsq_rrule`, `cholesky_rrule`
- `linear_systems.rs`: `solve_rrule`, `solve_triangular_rrule`, `inv_rrule`, `det_rrule`, `slogdet_rrule`
- `spectral.rs`: `eig_rrule`, `pinv_rrule`
- `matrix_functions.rs`: `matrix_exp_rrule`
- `norms.rs`: `norm_rrule`

`mod.rs` should re-export the public functions and keep the module tree explicit.

**Step 3: Keep tests out of production files**

Leave only `#[cfg(test)] mod tests;` where needed. Do not add inline helper
tests to the new production modules.

**Step 4: Run focused split checks**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg rrules_is_split_into_focused_modules -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg split_rrule_modules_stay_under_size_guideline -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg norm_rrule_matches_reference -- --nocapture
```

Expected: PASS

**Step 5: Commit**

```bash
git add tenferro-linalg/src/rrules tenferro-linalg/src/lib.rs tenferro-linalg/src/tests/organization.rs
git commit -m "refactor: split linalg reverse rule modules"
```

### Task 2: Split `tenferro-linalg` forward rules into focused modules

**Files:**
- Delete: `tenferro-linalg/src/frules.rs`
- Create: `tenferro-linalg/src/frules/mod.rs`
- Create: `tenferro-linalg/src/frules/svd_qr.rs`
- Create: `tenferro-linalg/src/frules/lu_eigen.rs`
- Create: `tenferro-linalg/src/frules/least_squares.rs`
- Create: `tenferro-linalg/src/frules/linear_systems.rs`
- Create: `tenferro-linalg/src/frules/spectral.rs`
- Create: `tenferro-linalg/src/frules/matrix_functions.rs`
- Create: `tenferro-linalg/src/frules/norms.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/src/tests/organization.rs`

**Step 1: Add or update the matching organization tests**

Representative test names:

```rust
#[test]
fn frules_is_split_into_focused_modules() {}

#[test]
fn split_frule_modules_stay_under_size_guideline() {}
```

**Step 2: Move the forward rules with the same family boundaries**

Split by family:

- `svd_qr.rs`: `svd_frule`, `qr_frule`
- `lu_eigen.rs`: `lu_frule`, `eigen_frule`
- `least_squares.rs`: `lstsq_frule`, `cholesky_frule`
- `linear_systems.rs`: `solve_frule`, `solve_triangular_frule`, `inv_frule`, `det_frule`, `slogdet_frule`
- `spectral.rs`: `eig_frule`, `pinv_frule`
- `matrix_functions.rs`: `matrix_exp_frule`
- `norms.rs`: `norm_frule`

Keep rule math unchanged. This task is organization-only.

**Step 3: Re-run focused checks**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg frules_is_split_into_focused_modules -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg split_frule_modules_stay_under_size_guideline -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg norm_frule_matches_reference -- --nocapture
```

Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-linalg/src/frules tenferro-linalg/src/lib.rs tenferro-linalg/src/tests/organization.rs
git commit -m "refactor: split linalg forward rule modules"
```

### Task 3: Finish the scalar family substrate with a ternary `Where` path

**Files:**
- Modify: `tenferro-prims/src/scalar_prims.rs`
- Modify: `tenferro-prims/src/scalar_cpu.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`
- Create: `tenferro-prims/src/tests/scalar_phase2.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/common.rs`
- Create: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/ternary.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs`

**Step 1: Add failing `Where` tests first**

Add focused tests for:

```rust
#[test]
fn cpu_scalar_where_executes_through_scalar_family() {}

#[test]
fn where_ad_routes_gradients_to_selected_branch_only() {}

#[test]
fn cuda_and_rocm_do_not_advertise_where_without_kernels() {}
```

**Step 2: Extend the scalar-family descriptor**

Introduce a ternary scalar family vocabulary for `Where` without weakening the
existing API boundaries. The public contract should stay family-based and
CPU/GPU-generic.

Prefer a small new enum and descriptor variant over open-coded special cases.

**Step 3: Implement CPU execution with `strided-rs`**

Use `strided-kernel` elementwise machinery if possible. Do not add a bespoke
nested indexing loop if the existing strided map/broadcast substrate can do the
same work.

**Step 4: Add dyadtensor primal and AD wiring**

Expose a macro-backed `where_ad` builder through `scalar_ad_builders`.
The predicate is non-differentiable; VJP/JVP should propagate only through the
selected data branch.

**Step 5: Re-run focused tests**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-prims scalar_phase2 -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
```

Expected: PASS

**Step 6: Commit**

```bash
git add tenferro-prims/src/scalar_prims.rs tenferro-prims/src/scalar_cpu.rs tenferro-prims/src/tests/mod.rs tenferro-prims/src/tests/scalar_phase2.rs extension/tenferro-dyadtensor/src/api/scalar_runtime.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/common.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/ternary.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs
git commit -m "feat: add scalar where family and ad wiring"
```

### Task 4: Expose the remaining scalar pointwise surface without repetitive boilerplate

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/common.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/binary.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Add failing surface tests**

Cover the scalar ops that already have CPU substrate support but are not yet
available as clean dyadtensor-facing builders:

- `maximum`
- `minimum`
- `clamp_min`
- `clamp_max`

**Step 2: Generalize the binary builder macros**

The implementation should reduce future boilerplate, not add one more custom
builder per op. Prefer declarative macro invocations that map:

- public constructor name
- docs name
- primal descriptor
- tangent/pullback formula

onto a shared helper.

**Step 3: Add the surface and AD rules**

Use mathematically honest subgradient choices and document any tie behavior in
tests or rustdoc.

**Step 4: Re-run focused tests**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
```

Expected: PASS

**Step 5: Update the parity audit**

Record that the scalar binary surface now exposes the non-analytic ordered-real
ops already supported by the CPU substrate.

**Step 6: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/scalar_ad_builders/common.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/binary.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs extension/tenferro-dyadtensor/src/api/scalar_runtime.rs extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "feat: expose ordered scalar binary ad surface"
```

### Task 5: Expose the remaining analytic unary and binary families through shared macros

**Files:**
- Modify: `extern/chainrules-scalarops/src/lib.rs`
- Modify: `extern/chainrules-scalarops/src/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/unary.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/binary.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs`
- Modify: `docs/AD/scalar_ops.md`

**Step 1: Add failing tests for the remaining public surface**

Cover:

- unary: `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- binary: `hypot`, `xlogy`

If `pow` still lacks a dyadtensor-facing builder on this branch, include it in
the same task rather than creating a one-off follow-up.

**Step 2: Extend backend-independent scalar rules**

Add the scalar rule helpers in `extern/chainrules-scalarops` first, keeping the
crate backend-agnostic.

**Step 3: Reuse the existing macro style**

Prefer more invocations of the shared unary/binary builder macros over new
custom builder bodies. If a new helper is needed, add it once in
`common.rs` and use it from multiple ops.

**Step 4: Re-run targeted tests**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p chainrules-scalarops --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
```

Expected: PASS

**Step 5: Commit**

```bash
git add extern/chainrules-scalarops/src/lib.rs extern/chainrules-scalarops/src/tests/mod.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/unary.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/binary.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders/mod.rs extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs docs/AD/scalar_ops.md
git commit -m "feat: expose remaining analytic scalar ad families"
```

### Task 6: Refresh support docs so deployed docs show crate-by-crate coverage

**Files:**
- Modify: `docs/AD/index.md`
- Modify: `docs/AD/scalar_ops.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/design/index.md`

**Step 1: Update the AD rule inventory**

Document the scalar AD rules that are actually implemented after Tasks 3-5.
Separate:

- scalar pointwise
- scalar reductions
- analytic pointwise
- linalg AD families

**Step 2: Update the deploy-facing support matrix**

`docs/design/supported-ops.md` should let a user quickly see what each crate
supports:

- `tenferro-prims`
- `tenferro-einsum`
- `tenferro-linalg`
- `tenferro-dyadtensor`
- `chainrules-scalarops`

For each crate, distinguish primal support from AD support where relevant.

**Step 3: Update the parity audit**

Mark the newly covered scalar/analytic families and call out what still remains
outside this PR:

- custom CUDA pointwise/reduction kernels
- any remaining CPU-only linalg kernels whose generic contract is still honest
  only through capability failure

**Step 4: Re-run docs checks**

Run:

```bash
cargo fmt --all
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/AD/index.md docs/AD/scalar_ops.md docs/design/supported-ops.md docs/design/reference/pytorch-dense-cpu-parity.md docs/design/index.md
git commit -m "docs: refresh issue 441 support matrices and ad inventory"
```

### Task 7: Run final focused and workspace verification, then open the PR

**Files:**
- Modify as needed from earlier tasks

**Step 1: Run focused crate tests first**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-prims --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p chainrules-scalarops --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor --lib -- --nocapture
```

Expected: PASS

**Step 2: Warn about NFS before heavy workspace verification**

Because the checkout lives under `/sharehome`, warn the user that heavy Rust
verification on NFS is discouraged, recommend a local-disk worktree, and only
continue if they explicitly want to proceed. If proceeding, keep
`CARGO_TARGET_DIR` on local disk.

**Step 3: Run the full PR gate**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS

**Step 4: Create the PR**

Run:

```bash
git push -u origin implement-441
gh pr create --base main --head implement-441 --title "Finish issue #441 scalar and analytic generic cleanup" --body-file /tmp/issue-441-pr.md
gh pr merge --auto --squash --delete-branch
bash scripts/monitor-pr-checks.sh <pr-number-or-url> --interval 30
```

The PR body must include:

- a concise summary of the runtime-generic cleanup
- the new scalar/analytic family coverage
- docs/support-matrix updates
- `Generated with [Claude Code](https://claude.com/claude-code)`

**Step 5: Record anything intentionally deferred**

If any item is consciously left out, document it in the PR description and final
handoff, especially:

- custom CUDA pointwise/reduction kernels
- GPU capability additions beyond truthful `false`
- any remaining follow-up for dense PyTorch parity not needed to close `#441`
