# Dyadtensor Op-First Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `tenferro-dyadtensor` into an op-first/shared-core layout so operation families are locally readable without changing the crate's execution model.

**Architecture:** Move AD value and dynamic wrapper definitions into `core/`, runtime and reverse tape into small shared modules, and relocate public operation wiring into `ops/` subtrees by family. Keep execution generic by continuing to route through `tenferro-prims` and `tenferro-linalg-prims`.

**Tech Stack:** Rust workspace, `tenferro-dyadtensor`, `tenferro-prims`, `tenferro-linalg`, `tenferro-linalg-prims`, workspace docs, existing structure tests.

---

### Task 1: Create the target module skeleton

**Files:**
- Create: `extension/tenferro-dyadtensor/src/core/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/runtime/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/tape/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/ops/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Test: `extension/tenferro-dyadtensor/src/api/tests/organization.rs`

**Step 1: Write failing structure tests**

Add tests that assert the new top-level modules exist and that the old
top-level `dyn_types/` and broad `api/` shape no longer remain as the primary
layout.

**Step 2: Run the structure tests to verify they fail**

Run:

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

Expected: failures for missing files/directories.

**Step 3: Add the new empty module skeleton**

Create `core`, `runtime`, `tape`, and `ops` modules with only narrow exports and
placeholder `mod` declarations.

**Step 4: Run the structure tests to verify they pass**

Run:

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

Expected: structure tests pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src/core extension/tenferro-dyadtensor/src/runtime extension/tenferro-dyadtensor/src/tape extension/tenferro-dyadtensor/src/ops extension/tenferro-dyadtensor/src/api/tests/organization.rs
git commit -m "refactor(dyadtensor): add op-first module skeleton"
```

### Task 2: Move AD value and dynamic wrapper infrastructure into `core/`

**Files:**
- Create: `extension/tenferro-dyadtensor/src/core/value/*`
- Create: `extension/tenferro-dyadtensor/src/core/node/*`
- Create: `extension/tenferro-dyadtensor/src/core/dyn/*`
- Create: `extension/tenferro-dyadtensor/src/core/convert/*`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: all imports under `extension/tenferro-dyadtensor/src/**`
- Test: `extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs`

**Step 1: Move structure tests first**

Add tests that assert `core/dyn` exists and `dyn_types` is no longer a top-level
entry point.

**Step 2: Run those tests and confirm failure**

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

**Step 3: Move modules without behavior changes**

Relocate `ad_value` and `dyn_types` contents into `core/` subtrees, update all
imports, and keep file sizes small.

**Step 4: Run focused crate tests**

```bash
cargo test -p tenferro-dyadtensor ad_value -- --nocapture
cargo test -p tenferro-dyadtensor dyn_types -- --nocapture
```

Expected: both pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/core extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src
git commit -m "refactor(dyadtensor): move ad values and dyn wrappers into core"
```

### Task 3: Move runtime and reverse-tape infrastructure into shared modules

**Files:**
- Create: `extension/tenferro-dyadtensor/src/runtime/context.rs`
- Create: `extension/tenferro-dyadtensor/src/runtime/dispatch.rs`
- Create: `extension/tenferro-dyadtensor/src/tape/registry.rs`
- Create: `extension/tenferro-dyadtensor/src/tape/scalar.rs`
- Create: `extension/tenferro-dyadtensor/src/tape/tensor.rs`
- Modify: imports across `extension/tenferro-dyadtensor/src/**`
- Test: `extension/tenferro-dyadtensor/src/runtime/tests/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/reverse_tape/tests/mod.rs`

**Step 1: Add structure tests for shared runtime/tape boundaries**

Protect the new directories and forbid direct operation modules from importing
old `context` or `reverse_tape` paths.

**Step 2: Run tests to verify failure**

```bash
cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
```

**Step 3: Move and simplify shared runtime/tape modules**

Keep public helper names small and generic. Do not alter runtime semantics.

**Step 4: Run focused tests**

```bash
cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
cargo test -p tenferro-dyadtensor reverse_tape -- --nocapture
```

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/runtime extension/tenferro-dyadtensor/src/tape extension/tenferro-dyadtensor/src
git commit -m "refactor(dyadtensor): isolate runtime and tape infrastructure"
```

### Task 4: Move scalar, reduction, and einsum operations into `ops/`

**Files:**
- Create: `extension/tenferro-dyadtensor/src/ops/scalar/*`
- Create: `extension/tenferro-dyadtensor/src/ops/reduction/*`
- Create: `extension/tenferro-dyadtensor/src/ops/einsum/*`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/api/ad/tests/*.rs`
- Test: `extension/tenferro-dyadtensor/src/api/tests/*.rs`

**Step 1: Add structure tests for op-first non-linalg families**

Assert that scalar, reduction, and einsum entrypoints live under `ops/`.

**Step 2: Run tests to verify failure**

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

**Step 3: Move builder/eager/primal wiring**

Relocate existing files into op-family directories. Keep re-export behavior
unchanged from the user perspective.

**Step 4: Run focused functional tests**

```bash
cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
cargo test -p tenferro-dyadtensor einsum -- --nocapture
```

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src
git commit -m "refactor(dyadtensor): move scalar reduction and einsum ops under ops"
```

### Task 5: Move linalg families into `ops/linalg/*`

**Files:**
- Create: `extension/tenferro-dyadtensor/src/ops/linalg/{svd,qr,lu,eigen,solve,norm,matrix_functions}/*`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/ops/linalg/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/api/ad/tests/builder_pullbacks.rs`
- Test: `extension/tenferro-dyadtensor/src/api/ad/tests/linalg_finite_difference.rs`
- Test: `extension/tenferro-dyadtensor/src/api/tests/builder_coverage.rs`

**Step 1: Add structure tests for linalg op-first layout**

Protect `ops/linalg/svd`, `ops/linalg/qr`, `ops/linalg/lu`, `ops/linalg/eigen`,
`ops/linalg/solve`, `ops/linalg/norm`, and `ops/linalg/matrix_functions`.

**Step 2: Run tests to verify failure**

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

**Step 3: Move linalg builders and eager AD/primal entrypoints**

Keep each family self-contained. `svd` should be readable from its local
directory plus `tenferro-linalg`.

**Step 4: Run focused linalg tests**

```bash
cargo test -p tenferro-dyadtensor builder_pullbacks -- --nocapture
cargo test -p tenferro-dyadtensor linalg_finite_difference -- --nocapture
cargo test -p tenferro-dyadtensor builder_coverage -- --nocapture
```

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ops extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src
git commit -m "refactor(dyadtensor): move linalg families under ops"
```

### Task 6: Remove obsolete `api/` and `dyn_types/` roots and simplify re-exports

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Delete/move: obsolete `extension/tenferro-dyadtensor/src/api/**`
- Delete/move: obsolete `extension/tenferro-dyadtensor/src/dyn_types/**`
- Test: `extension/tenferro-dyadtensor/src/api/tests/organization.rs`

**Step 1: Update organization tests to forbid legacy roots**

Keep only intentionally preserved compatibility shims, if any remain.

**Step 2: Run tests to verify failure**

```bash
cargo test -p tenferro-dyadtensor organization -- --nocapture
```

**Step 3: Remove obsolete roots**

Finish the cutover and keep the public flat re-exports in `lib.rs` if still
useful.

**Step 4: Run crate-local tests**

```bash
cargo test -p tenferro-dyadtensor --lib -- --nocapture
```

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src
git commit -m "refactor(dyadtensor): complete op-first cutover"
```

### Task 7: Update documentation to explain the new mental model

**Files:**
- Modify: `docs/api_index.md`
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`

**Step 1: Update crate docs and design docs**

Explain `core`, `runtime`, `tape`, `structured`, and `ops` with `ops` as the
primary navigation entry.

**Step 2: Run docs checks**

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

**Step 3: Commit**

```bash
git add docs/api_index.md docs/design/architecture.md docs/design/autodiff.md docs/design/supported-ops.md extension/tenferro-dyadtensor/src/lib.rs
git commit -m "docs: explain dyadtensor op-first layout"
```

### Task 8: Final verification and PR

**Files:**
- Verify all touched files

**Step 1: Run formatting and full tests**

```bash
cargo fmt --all --check
cargo test --workspace --release
```

**Step 2: Run coverage and docs gates**

```bash
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

**Step 3: Re-read the diff for drift**

Confirm there is no accidental reintroduction of:

- top-level `dyn_types`
- broad `api/` buckets as the main implementation home
- CPU-only execution shortcuts
- inline test suites in production files

**Step 4: Commit any final cleanups**

```bash
git add .
git commit -m "refactor: finish dyadtensor op-first layout cleanup"
```

**Step 5: Create PR**

```bash
git push -u origin refactor/dyadtensor-op-first-layout
gh pr create --base main --head refactor/dyadtensor-op-first-layout --title "refactor: reorganize dyadtensor around op-first layout" --body "$(cat <<'EOF'
## Summary
- reorganize tenferro-dyadtensor into an op-first/shared-core layout
- move AD values and dynamic wrappers into core and isolate runtime/tape infrastructure
- group scalar, einsum, reduction, and linalg families under ops

Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
gh pr merge --auto --squash --delete-branch
```
