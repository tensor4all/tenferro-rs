# Final Generic/DRY Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining visible production debt around CPU-only linalg bridging, dyadtensor runtime duplication, and Burn panic-oriented glue while keeping docs aligned.

**Architecture:** First remove the hidden CPU semiring bridge in `tenferro-linalg`, then simplify `dyadtensor` runtime dispatch around a single slot model, then replace `tenferro-burn` `expect(...)` paths with checked helpers, and finally update docs plus structural tests so the cleaned layering is enforced.

**Tech Stack:** Rust workspace crates (`tenferro-linalg`, `tenferro-prims`, `tenferro-dyadtensor`, `tenferro-burn`), rustdoc, workspace CI gates.

---

### Task 1: Lock in the bridge/runtime debt with failing structural tests

**Files:**
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs`
- Modify or create: `extension/tenferro-burn/src/tests/*.rs`

**Step 1: Add structural assertions**

Add tests that:

- reject thread-local `CpuContext` ownership in `tenferro-linalg/src/prims_bridge.rs`
- ensure dyadtensor runtime dispatch keeps concrete runtime names centralized
- ensure `extension/tenferro-burn/src/{lib,backward}.rs` do not contain
  `expect(`

**Step 2: Run targeted tests and watch them fail**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-linalg runtime_capability -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-burn --lib -- --nocapture
```

**Step 3: Commit**

```bash
git add tenferro-linalg/src/tests/runtime_capability.rs extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs extension/tenferro-burn/src/tests
git commit -m "test: lock final cleanup structural debt"
```

### Task 2: Remove the hidden CPU semiring bridge from `tenferro-linalg`

**Files:**
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/ad_helpers/backend_ops.rs`
- Modify: `tenferro-linalg/src/ad_helpers/matrix_exp.rs`
- Modify: call sites under `tenferro-linalg/src/primal/**`, `tenferro-linalg/src/frules/**`, `tenferro-linalg/src/rrules/**` as needed

**Step 1: Replace thread-local context ownership**

Make the bridge require the semiring-core context from the caller instead of
creating or caching `CpuContext` internally.

**Step 2: Update call sites**

Thread the already-available linalg context through helper paths that use the
bridge.

**Step 3: Run targeted tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-linalg --lib -- --nocapture
```

**Step 4: Commit**

```bash
git add tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/ad_helpers tenferro-linalg/src/primal tenferro-linalg/src/frules tenferro-linalg/src/rrules
git commit -m "refactor: remove cpu-only linalg prims bridge"
```

### Task 3: Collapse repeated dyadtensor runtime-slot wiring

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/contracts.rs` if needed
- Modify: dyadtensor API call sites/macros if needed

**Step 1: Extract a single slot-driven dispatch pattern**

Keep CPU/CUDA/ROCm metadata in one place and remove repeated concrete type
aliases in the dispatch helpers/macros.

**Step 2: Keep the public contract unchanged**

Do not widen the API surface. This is an internal cleanup for generic/DRY/KISS.

**Step 3: Run targeted tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
```

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api
git commit -m "refactor: simplify dyadtensor runtime dispatch"
```

### Task 4: Remove library-code `expect(...)` from `tenferro-burn`

**Files:**
- Modify: `extension/tenferro-burn/src/backward.rs`
- Modify: `extension/tenferro-burn/src/lib.rs`
- Modify tests as needed

**Step 1: Introduce checked helpers**

Move parse/tree/input extraction logic into checked helpers returning typed or
internal errors.

**Step 2: Minimize unavoidable panic surface**

Only keep assertions where the Burn trait contract is inherently infallible and
the failure truly indicates an internal invariant violation. Avoid `expect(...)`.

**Step 3: Run targeted tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-target cargo test -p tenferro-burn --lib -- --nocapture
```

**Step 4: Commit**

```bash
git add extension/tenferro-burn/src extension/tenferro-burn/tests
git commit -m "refactor: remove burn expect-based glue"
```

### Task 5: Update active docs and crate docs

**Files:**
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify rustdoc in touched crates as needed

**Step 1: Update active docs**

Describe the cleaned bridge/runtime shape and remaining true debt.

**Step 2: Re-run docs gate**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-doc-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-final-generic-dry-cleanup-doc-target/doc
```

**Step 3: Commit**

```bash
git add docs
git commit -m "docs: align generic cleanup architecture notes"
```

### Task 6: Full verification and PR

**Files:**
- Verify entire workspace

**Step 1: Run full required checks**

```bash
cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-release-target cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-cov-target cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-final-generic-dry-cleanup-doc-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-final-generic-dry-cleanup-doc-target/doc
```

**Step 2: Create PR**

```bash
git push -u origin refactor/final-generic-dry-cleanup
gh pr create --base main --head refactor/final-generic-dry-cleanup
gh pr merge --auto --squash --delete-branch
```
