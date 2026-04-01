# Tenferro Linearize Hard-Cut Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `tenferro-rs` on top of the new `tidu-rs` linearize-first API so that `Tape`, `TrackedValue`, `expert`, `AdTensor<T>`, and `DynAdTensor` disappear from the final internal and public implementation, while `jvp`/`vjp` remain supported and HVP stays out of scope.

**Architecture:** Treat `tidu-rs` `linearize-first` as the fixed upstream contract. Do not preserve compatibility layers. Instead, hard-cut `tenferro` from the top of the dependency stack downward: first replace the AD core carrier, then migrate ops/linalg onto `LinearizableOp`/`LinearizedOp`, then reconnect the public `Tensor` surface, and only then re-enable downstream `tensor4all-rs` verification. During the migration, downstream crates are not “disabled” in source; they are simply not part of the verification target until their stage is reached.

**Tech Stack:** Rust 2021, `tidu-rs` linearize-first (`Value`, `LinearizableOp`, `LinearizedOp`, checkpoint policy scope), `tenferro` internal crates, `cargo fmt`, `cargo clippy`, `cargo nextest`, grep-based organization guards

---

## Non-Negotiable Constraints

- Do **not** keep `Tape`, `TrackedValue`, `expert`, `AdTensor<T>`, or `DynAdTensor` as compatibility shims.
- Do **not** add a second migration layer to emulate the old `tidu` API.
- Do **not** open a PR before **Stage E** is green.
- Do commit after each task or subtask batch that restores a local green state.
- Prefer crate-local verification over workspace-wide verification until the relevant stage is reached.
- HVP is explicitly out of scope. Only `primal`, `linearize`, `jvp`, `vjp`, and checkpoint policy should survive.

## Architecture Gate: Single Source Of Truth For Reverse Metadata

This migration must lock the public/internal carrier boundary **before** the core hard cut:

```rust
struct Tensor {
    inner: tidu::Value<DynTensor>,
}
```

If a small amount of frontend-only state is still needed, the only allowed extension is:

```rust
struct Tensor {
    inner: tidu::Value<DynTensor>,
    frontend_flags: FrontendFlags,
}
```

The following designs are forbidden:

- `Tensor` owns independent reverse metadata
- `Tensor` and `Value<DynTensor>` both store graph identity / grad state
- `AdTensor<T>` or `DynAdTensor` survive as compatibility carriers

Consequences of this gate:

- `Value<DynTensor>` is the only source of truth for reverse graph connectivity, grad state, and checkpoint replay ownership
- `Tensor` is only the public façade and error-translation layer
- all Stage B/C/D work must preserve this invariant

## Fixed Upstream Contract

Before touching `tenferro-rs`, fix the exact `tidu-rs` contract that downstream will target:

- Upstream worktree: `/home/shinaoka/tensor4all/tidu-rs/.worktrees/linearize-first`
- Upstream commit: `9bce912`
- Public `tidu` surface:
  - `Value`
  - `LinearizableOp`
  - `LinearizedOp`
  - `Schema`
  - `SlotSchema`
  - `CheckpointMode`
  - `AdExecutionPolicy`
  - `CheckpointHint`
  - `with_ad_policy(...)`

`tenferro-rs` should pin to that upstream revision during the migration.

## Stage Overview

- **Stage A:** Freeze dependency and add migration guards
- **Stage B:** Hard-cut `tenferro-internal-ad-core`
- **Stage C:** Migrate `tenferro-internal-ad-ops` and `tenferro-internal-ad-linalg`
- **Stage D:** Rebuild `tenferro-internal-ad-surface` and public `tenferro`
- **Stage E:** Reconnect `tensor4all-rs` downstream and run full verification

Each stage below is locally green before moving on.

### Task 1: Freeze the new `tidu-rs` dependency and lock migration guards

**Files:**
- Modify: `Cargo.toml`
- Modify: `tenferro/Cargo.toml`
- Modify: `internal/tenferro-internal-ad-core/Cargo.toml`
- Modify: `internal/tenferro-internal-ad-ops/Cargo.toml`
- Modify: `internal/tenferro-internal-ad-linalg/Cargo.toml`
- Modify: `internal/tenferro-internal-ad-surface/Cargo.toml`
- Modify: `tenferro/tests/integration.rs`
- Create: `tenferro/tests/integration/migration_organization.rs`

**Step 1: Point `tidu` to the fixed linearize-first revision**

Use the `linearize-first` revision from `tidu-rs` instead of the old edge-only branch or local path.

**Step 2: Add a failing organization test for forbidden legacy names**

Create `tenferro/tests/integration/migration_organization.rs` that fails if these names remain in the final public/internal story:

- `Tape`
- `TrackedValue`
- `expert`
- `AdTensor`
- `DynAdTensor`

This test should scan:

- `tenferro/src`
- `internal/tenferro-internal-ad-*`
- `tenferro/tests/integration`

Exclude:

- historical plan docs
- test comments whose only purpose is to assert a forbidden string is absent

**Step 3: Add a failing organization test for the new required names**

The same file should assert that the intended `tidu` vocabulary appears where appropriate:

- `Value`
- `LinearizableOp`
- `LinearizedOp`
- `CheckpointHint`

**Step 4: Run the red tests**

Run:

```bash
cargo test -p tenferro --test integration --release migration_organization
```

Expected: FAIL because the legacy names are still widespread.

**Step 5: Commit the dependency freeze and guards**

```bash
git add Cargo.toml \
        tenferro/Cargo.toml \
        internal/tenferro-internal-ad-core/Cargo.toml \
        internal/tenferro-internal-ad-ops/Cargo.toml \
        internal/tenferro-internal-ad-linalg/Cargo.toml \
        internal/tenferro-internal-ad-surface/Cargo.toml \
        tenferro/tests/integration.rs \
        tenferro/tests/integration/migration_organization.rs
git commit -m "test: lock tenferro linearize-first migration contract"
```

### Task 2: Hard-cut `tenferro-internal-ad-core`

**Files:**
- Modify: `internal/tenferro-internal-ad-core/src/lib.rs`
- Delete: `internal/tenferro-internal-ad-core/src/tensor.rs`
- Delete: `internal/tenferro-internal-ad-core/src/dyn_ad_tensor.rs`
- Delete: any remaining `ad_tensor*` / `dyn_ad_tensor*` source files in this crate
- Create: `internal/tenferro-internal-ad-core/src/value.rs`
- Create: `internal/tenferro-internal-ad-core/src/linearized.rs`
- Create: `internal/tenferro-internal-ad-core/src/reverse.rs`
- Modify: `internal/tenferro-internal-ad-core/src/tests/*`

**Step 1: Write failing core tests for the new carrier**

Add tests that exercise only the new internal core vocabulary:

- constructing a reverse-enabled internal value without `AdTensor`
- reading/writing tangent snapshots without `DynAdTensor`
- creating reverse graph edges through `tidu::Value<DynTensor>`

Expected: FAIL because `AdTensor` is still the carrier.

**Step 2: Introduce the new internal carrier type**

Create a new internal carrier wrapper around:

- primal `DynTensor`
- optional `tidu::Value<DynTensor>` reverse handle
- optional forward tangent payload if still needed for first-order forward support

The exact type name may be `CoreValue`, `AdValue`, or similar, but it must **not** be `AdTensor` or `DynAdTensor`.

**Step 3: Rebuild core constructors and snapshots**

Replace:

- `new_primal`
- `new_forward`
- `new_reverse_leaf`
- `new_reverse_leaf_with_tangent`

with constructors on the new carrier type.

**Step 4: Remove old typed/dynamic wrappers**

Delete `AdTensor<T>` and `DynAdTensor` from this crate entirely. If tests still need them, the tests are wrong and must be rewritten now.

**Step 5: Rewrite core tests**

Replace old `AdTensor`/`DynAdTensor` tests with:

- internal carrier tests
- direct `tidu::Value<DynTensor>` integration tests

**Step 6: Verify Stage B**

Run:

```bash
cargo nextest run --release -p tenferro-internal-ad-core
cargo clippy -p tenferro-internal-ad-core --tests
```

Expected: PASS, and `rg -n "AdTensor|DynAdTensor|Tape|TrackedValue|expert"` in this crate returns no hits outside historical docs.

**Step 7: Commit**

```bash
git add internal/tenferro-internal-ad-core
git commit -m "refactor: hard-cut internal ad core to tidu value carrier"
```

### Task 3: Migrate scalar/reduction/einsum ops to `LinearizableOp`

**Files:**
- Modify: `internal/tenferro-internal-ad-ops/src/lib.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/ad/mod.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/ad/scalar_eager.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/scalar/ad/common.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/reduction/ad.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/einsum/ad.rs`
- Modify: `internal/tenferro-internal-ad-ops/src/ops/einsum/backward.rs`
- Modify: `internal/tenferro-internal-ad-ops/tests/*`

**Step 1: Add failing op tests using only `LinearizableOp`/`LinearizedOp`**

For each op family, write one minimal failing test that expects:

- no `register_rule`
- no `Tape`
- no `AdTensor`
- local `jvp` and `vjp` access through linearization

**Step 2: Port pointwise ops first**

For scalar pointwise ops:

- implement `LinearizableOp<DynTensor>`
- move old backward math into `LinearizedOp::vjp`
- provide `jvp` only where already well-defined in current forward mode

**Step 3: Port reductions**

For reductions:

- linearize from saved shapes / reduction axes / keepdim flags
- expose reverse via `vjp`
- avoid any tape identity plumbing

**Step 4: Port einsum**

Reuse the already separated backward math, but attach it to `LinearizedOp::vjp` instead of any closure/tape registration path.

**Step 5: Delete legacy op registration**

Remove:

- `register_rule`
- closure rule registration
- tape-aware helpers

from this crate.

**Step 6: Verify Stage C (ops half)**

Run:

```bash
cargo nextest run --release -p tenferro-internal-ad-ops
cargo clippy -p tenferro-internal-ad-ops --tests
```

**Step 7: Commit**

```bash
git add internal/tenferro-internal-ad-ops
git commit -m "refactor: migrate internal ad ops to linearizable tidu ops"
```

### Task 4: Migrate linalg to `LinearizableOp` / `LinearizedOp`

**Files:**
- Modify: `internal/tenferro-internal-ad-linalg/src/lib.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/eager_impl.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/svd_qr_impl.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/ops/linalg/results.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/*`

**Step 1: Write failing QR/SVD tests for the new boundary**

Tests should assert:

- no `AdTensor`
- no `DynAdTensor`
- no tape handles
- QR/SVD linearizations expose `vjp`
- QR/SVD may expose `jvp` where supported
- `checkpoint_hint()` is set deliberately on expensive ops

**Step 2: Port easy single-output linalg first**

Start with:

- `cholesky`
- `solve`
- `inv`
- `det`
- `norm`

These should move to dynamic `LinearizableOp<DynTensor>` without result wrapper churn.

**Step 3: Port multi-output QR/SVD**

Introduce linearized residuals that keep:

- input primal
- output primals needed for `vjp`
- static config

Do **not** add HVP hooks.

**Step 4: Delete old eager typed result carriers**

Any remaining typed builder/result helpers that mention `AdTensor` or `DynAdTensor` should be removed now.

**Step 5: Verify Stage C (full)**

Run:

```bash
cargo nextest run --release -p tenferro-internal-ad-ops
cargo nextest run --release -p tenferro-internal-ad-linalg
cargo clippy -p tenferro-internal-ad-linalg --tests
```

**Step 6: Commit**

```bash
git add internal/tenferro-internal-ad-linalg
git commit -m "refactor: migrate linalg ad to linearized tidu runtime"
```

### Task 5: Rebuild `tenferro-internal-ad-surface` and public `tenferro`

**Files:**
- Modify: `internal/tenferro-internal-ad-surface/src/lib.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/autograd_api.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/*`
- Delete: any remaining `dyn_ad_tensor` modules
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/core/value/*`
- Modify: `tenferro/src/ops/**/*`
- Modify: `tenferro/tests/integration/*`

**Step 1: Add failing public-surface tests**

The failing tests should require:

- `tenferro::Tensor` as the only user-facing tensor type
- no public/internal use of `AdTensor`, `DynAdTensor`, `Tape`
- public methods wired through the new `tidu` API

**Step 2: Replace `dyn_ad_tensor` with direct tensor/runtime wiring**

`Tensor` should either:

- hold the internal core carrier directly, or
- hold the minimal metadata needed to bridge to `tidu::Value<DynTensor>`

but it must not wrap `DynAdTensor`.

**Step 3: Rewire public ops**

Update public `Tensor` ops and linalg entrypoints to use the migrated internal ops/linalg crates.

**Step 4: Rewrite integration tests**

Remove or rewrite all integration tests that still construct:

- `Tape`
- `AdTensor`
- `DynAdTensor`

They should test only:

- `Tensor`
- internal core carrier tests inside internal crates
- `tidu` integration at crate boundaries

**Step 5: Verify Stage D**

Run:

```bash
cargo nextest run --release -p tenferro
cargo test -p tenferro --doc --release
cargo clippy -p tenferro --tests
```

**Step 6: Commit**

```bash
git add internal/tenferro-internal-ad-surface tenferro
git commit -m "refactor: reconnect tenferro surface to linearized tidu api"
```

### Task 6: Re-enable `tensor4all-rs` downstream and run full verification

**Files:**
- Modify: `/home/shinaoka/tensor4all/tensor4all-rs/Cargo.toml`
- Modify: any `tensor4all-rs` crates still depending on old tenferro behavior
- Modify: downstream integration tests as needed

**Step 1: Switch downstream to the new `tenferro-rs` branch/revision**

Point `tensor4all-rs` at the updated `tenferro-rs` worktree revision.

**Step 2: Fix downstream breakage crate by crate**

Target order:

- `tensor4all-core`
- `tensor4all-treetn`
- remaining crates that use tenferro-backed AD

Each fix must use public `tenferro::Tensor` only.

**Step 3: Run stage-local verification repeatedly**

Examples:

```bash
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-treetn
```

**Step 4: Run final full verification**

In `tenferro-rs`:

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
```

In `tensor4all-rs`:

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
```

**Step 5: Commit final downstream reconnect**

```bash
git add .
git commit -m "refactor: reconnect tensor4all stack to linearized tidu runtime"
```

## Final Acceptance Checklist

Before considering the branch complete, verify all of the following:

- `tidu-rs` dependency is pinned to the linearize-first revision
- `tenferro-rs` source has zero remaining hits for:
  - `Tape`
  - `TrackedValue`
  - `expert`
  - `AdTensor`
  - `DynAdTensor`
- public `tenferro` docs and examples use only `Tensor`
- HVP APIs are absent from the migrated path
- QR/SVD are migrated with `jvp`/`vjp` only
- all stage-local tests were green before moving forward
- full `tenferro-rs` and `tensor4all-rs` verification passes at Stage E
- PR creation is still deferred until Stage E is complete

## Notes For Execution

- If a stage exposes a large red surface, do **not** paper over it with compatibility wrappers.
- If a crate cannot be made green without reviving legacy types, stop and redesign that stage before continuing.
- If QR/SVD block the linalg stage, land the easy single-output linalg ops first and keep QR/SVD as the last linalg subtask inside Stage C.
- The “disable downstream crates” strategy means “do not include them in the current verification target”, not “break their source on purpose”.
