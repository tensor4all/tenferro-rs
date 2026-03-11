# Remove Legacy TensorPrims Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete `TensorPrims<A>`, `PrimDescriptor`, and `Extension` from the workspace and cut every crate over to family-native primitive execution contracts in one PR.

**Architecture:** Make `tenferro-prims` family-native first, then port `tenferro-tropical`, downstream call sites, tests, and docs. Do not keep a compatibility shim at any stage; the branch may be temporarily broken between commits, but each checkpoint should move toward the final contract rather than preserve the old one.

**Tech Stack:** Rust workspace, `tenferro-prims`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-linalg-prims`, `tenferro-capi`, `tenferro-dyadtensor`, `tenferro-tropical`, `cargo test`, `cargo llvm-cov`, `cargo doc`

---

### Task 1: Freeze the design state in the branch

**Files:**
- Create: `docs/plans/2026-03-12-remove-legacy-tensor-prims-design.md`
- Create: `docs/plans/2026-03-12-remove-legacy-tensor-prims.md`

**Step 1: Save the approved design**

Write the design doc exactly around the accepted end state:

- remove `TensorPrims<A>`
- remove `PrimDescriptor`
- remove `Extension`
- no compatibility adapters
- tropical included in the same cutover

**Step 2: Save this implementation plan**

Keep the plan substrate-first and explicitly name every crate that must be cut
over in the PR.

**Step 3: Commit the planning checkpoint**

Run:

```bash
git add docs/plans/2026-03-12-remove-legacy-tensor-prims-design.md docs/plans/2026-03-12-remove-legacy-tensor-prims.md
git commit -m "docs: plan legacy TensorPrims removal"
```

### Task 2: Replace the legacy semiring substrate in tenferro-prims

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/semiring_core.rs`
- Modify: `tenferro-prims/src/semiring_fast_path.rs`
- Modify: `tenferro-prims/src/cpu.rs`
- Modify: `tenferro-prims/src/cuda.rs`
- Modify: `tenferro-prims/src/gpu_stubs.rs`
- Modify: `tenferro-prims/src/registry.rs`
- Test: `tenferro-prims/src/tests/mod.rs`
- Test: `tenferro-prims/src/tests/analytic_phase1.rs`
- Test: `tenferro-prims/src/tests/scalar_phase1.rs`
- Test: `tenferro-prims/tests/prims_tests.rs`
- Test: `tenferro-prims/tests/inject_tests.rs`

**Step 1: Write failing tests for family-native planning/execution**

Add or rewrite tests so they call:

- `TensorSemiringCore::plan/execute`
- `TensorSemiringFastPath::plan/execute/has_fast_path`
- `TensorScalarPrims::plan/execute/has_scalar_support`
- `TensorAnalyticPrims::plan/execute/has_analytic_support`

and no longer mention `TensorPrims`, `PrimDescriptor`, or `Extension`.

**Step 2: Remove the blanket adapters**

Delete:

- `SemiringCoreDescriptor::to_legacy`
- `SemiringFastPathDescriptor::to_legacy`
- blanket `impl<Alg, B> TensorSemiringCore for B where B: TensorPrims<Alg>`
- blanket `impl<Alg, B> TensorSemiringFastPath for B where B: TensorPrims<Alg>`

**Step 3: Replace the backend dispatcher**

Port CPU/CUDA/stub planning and execution to the family traits directly.

Concrete outcomes:

- `CpuBackend` directly implements all four family traits
- `CudaBackend` directly implements the semiring and scalar/analytic family
  traits it truthfully supports
- `RocmBackend` stubs directly implement those family traits with truthful
  unsupported behavior

**Step 4: Remove the legacy surface from tenferro-prims**

Delete from `tenferro-prims/src/lib.rs`:

- `TensorPrims`
- `PrimDescriptor`
- `Extension`

Update crate docs so examples use family traits only.

**Step 5: Run focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-prims --lib
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-prims --test prims_tests
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-prims --test inject_tests
```

**Step 6: Commit**

```bash
git add tenferro-prims
git commit -m "refactor: remove legacy TensorPrims substrate"
```

### Task 3: Port tenferro-tropical and tropical capi

**Files:**
- Modify: `extension/tenferro-tropical/src/prims.rs`
- Modify: `extension/tenferro-tropical/src/lib.rs`
- Modify: `extension/tenferro-tropical/src/algebra.rs`
- Modify: `extension/tenferro-tropical/src/ad.rs`
- Modify: `extension/tenferro-tropical/tests/tropical_tests.rs`
- Modify: `extension/tenferro-tropical-capi/src/lib.rs`

**Step 1: Rewrite tropical trait impls**

Change tropical execution from `impl TensorPrims<XxxAlgebra> for CpuBackend` to
direct `TensorSemiringCore<XxxAlgebra>` impls. If a fast-path trait is needed,
implement it truthfully with unsupported capabilities where appropriate.

**Step 2: Rewrite tropical AD and capi bounds**

Replace `TensorPrims<Alg>` bounds with `TensorSemiringCore<Alg>` in:

- tropical AD entrypoints
- tropical capi helper generics

**Step 3: Rewrite tropical tests**

Replace every legacy descriptor-based test with family descriptor tests.

**Step 4: Run focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-tropical --release
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-tropical-capi --release
```

**Step 5: Commit**

```bash
git add extension/tenferro-tropical extension/tenferro-tropical-capi
git commit -m "refactor: port tropical execution to semiring families"
```

### Task 4: Port einsum, capi, and dyadtensor call sites

**Files:**
- Modify: `tenferro-einsum/src/**/*.rs`
- Modify: `tenferro-capi/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/contracts.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/structured/einsum.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad_builders/einsum.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs`

**Step 1: Rewrite einsum generic bounds**

`tenferro-einsum` should require only:

- `TensorSemiringCore`
- `TensorSemiringFastPath`

Remove all `TensorPrims` references.

**Step 2: Rewrite capi contiguous materialization**

Replace `PrimDescriptor::MakeContiguous` calls with semiring-core `MakeContiguous`.

**Step 3: Rewrite dyadtensor runtime contracts**

Change `EinsumRuntimeValue` and runtime capability checks so they use semiring
family traits and `has_fast_path`, not `TensorPrims` and `Extension::Contract`.

**Step 4: Run focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-einsum --release
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-capi --release
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-target cargo test -p tenferro-dyadtensor --release
```

**Step 5: Commit**

```bash
git add tenferro-einsum tenferro-capi extension/tenferro-dyadtensor
git commit -m "refactor: cut over downstream runtime contracts"
```

### Task 5: Remove legacy references from tests and docs

**Files:**
- Modify: `docs/api_index.md`
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/linalg-backend-api.md`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/design/linalg.md`
- Modify rustdoc in `tenferro-prims`, `tenferro-tropical`, `tenferro-capi`, `tenferro-einsum`

**Step 1: Remove migration-language from active docs**

Delete phrasing such as:

- "migration layer"
- "backed by legacy TensorPrims"
- "compatibility paths"

from active docs. Keep historical notes only in `docs/plans/`.

**Step 2: Rewrite rustdoc examples**

Any example using `PrimDescriptor`, `Extension`, or `TensorPrims` must be
rewritten to the family traits.

**Step 3: Add one audit grep test pass**

Run:

```bash
rg -n '\\bTensorPrims\\b|\\bPrimDescriptor\\b|\\bExtension\\b' tenferro-prims tenferro-einsum tenferro-linalg tenferro-capi extension extern docs/design docs/api_index.md
```

Expected: no matches outside `docs/plans/` or intentionally historical text not
included in deploy docs.

**Step 4: Commit**

```bash
git add docs tenferro-prims tenferro-einsum tenferro-capi extension/tenferro-tropical extension/tenferro-dyadtensor
git commit -m "docs: remove legacy TensorPrims references"
```

### Task 6: Full verification and PR

**Files:**
- Modify only if verification reveals breakage

**Step 1: Run the full required checks**

Run:

```bash
cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-release cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-cov cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-remove-legacy-tensor-prims-doc cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --root-dir . --doc-root /tmp/tenferro-remove-legacy-tensor-prims-doc/doc
```

**Step 2: Push and create PR**

```bash
git push -u origin feat/remove-legacy-tensor-prims
gh pr create --base main --title "refactor: remove legacy TensorPrims" --body "## Summary\n- remove TensorPrims, PrimDescriptor, and Extension from the workspace\n- cut over prim backends and downstream crates to family-native contracts\n- rewrite tests and docs around the new primitive families\n\nGenerated with Codex"
gh pr merge --auto --squash --delete-branch
```

**Step 3: Monitor CI**

```bash
bash scripts/monitor-pr-checks.sh <pr-number-or-url> --interval 30
```

If a required check fails, fix it locally, rerun the relevant checks, push, and
resume monitoring.
