# Dyadtensor Registry Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace dyadtensor's generic global context and multi-registry reverse-tape storage with explicit runtime-holder and tape-rule-store subsystems.

**Architecture:** Introduce a dedicated runtime holder for `RuntimeContext`, then collapse reverse-tape registration into a single tape-local store abstraction. Migrate callers without changing AD math or public builder semantics.

**Tech Stack:** Rust, `thread_local!`, `HashMap`, typed rule stores, `thiserror`, existing dyadtensor AD/runtime APIs

---

### Task 1: Introduce a runtime-only holder

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/context.rs`
- Modify: `extension/tenferro-dyadtensor/src/runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Test: `extension/tenferro-dyadtensor/src/context/tests/mod.rs`

**Step 1: Write the failing test**

Add assertions that the public API exports only runtime-oriented helpers and
that missing runtime maps directly to `Error::RuntimeNotConfigured`.

**Step 2: Run the targeted test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor context -- --nocapture
```

Expected: fail until runtime-only holder is in place.

**Step 3: Implement the runtime holder**

- replace generic `set_global_context` storage with a `RuntimeContext`-specific
  thread-local slot
- provide a guard that restores the previous runtime on drop
- remove `MissingGlobalContext` / `ContextTypeMismatch` usage from runtime code

**Step 4: Re-run the targeted test**

Run the same command and expect PASS.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/context.rs extension/tenferro-dyadtensor/src/runtime.rs extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src/context/tests/mod.rs
git commit -m "refactor: make dyadtensor runtime holder explicit"
```

### Task 2: Replace reverse-tape registries with a tape-local store

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/reverse_tape/registry.rs`
- Modify: `extension/tenferro-dyadtensor/src/reverse_tape/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/reverse_tape/tests/mod.rs`

**Step 1: Write the failing test**

Add structural tests that assert reverse tape state is stored through one
`TapeRuleStore`-style abstraction instead of multiple parallel registries.

**Step 2: Run the targeted test**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor reverse_tape -- --nocapture
```

Expected: fail before the storage refactor.

**Step 3: Implement the tape-local store**

- create a `TapeRuleStore`
- keep typed rule tables inside the store
- collapse the five thread-local registries into one outer tape registry
- preserve existing registration and lookup helper signatures where possible

**Step 4: Re-run the targeted test**

Run the same command and expect PASS.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/reverse_tape/registry.rs extension/tenferro-dyadtensor/src/reverse_tape/mod.rs extension/tenferro-dyadtensor/src/reverse_tape/tests/mod.rs
git commit -m "refactor: collapse dyadtensor reverse tape registries"
```

### Task 3: Migrate AD call sites and remove stale generic-context errors

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/error.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Modify call sites under:
  - `extension/tenferro-dyadtensor/src/api/**`
  - `extension/tenferro-dyadtensor/src/ad_value/**`
  - `extension/tenferro-dyadtensor/src/dyn_types/**`

**Step 1: Write the failing tests**

Add or adjust runtime-surface and AD regression tests so they no longer expect
generic global-context errors.

**Step 2: Run the targeted tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor runtime_surface -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor builder_pullbacks -- --nocapture
```

Expected: fail until call sites and errors are migrated.

**Step 3: Implement the migration**

- remove stale generic-context errors from `Error`
- update runtime access helpers
- update any direct context helper callers

**Step 4: Re-run the targeted tests**

Expect PASS.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/error.rs extension/tenferro-dyadtensor/src/api/runtime.rs extension/tenferro-dyadtensor/src/api extension/tenferro-dyadtensor/src/ad_value extension/tenferro-dyadtensor/src/dyn_types
git commit -m "refactor: align dyadtensor errors with explicit registries"
```

### Task 4: Add structural guard tests and update docs

**Files:**
- Modify or add tests under:
  - `extension/tenferro-dyadtensor/src/context/tests/`
  - `extension/tenferro-dyadtensor/src/reverse_tape/tests/`
  - `extension/tenferro-dyadtensor/src/api/tests/`
- Modify docs:
  - `extension/tenferro-dyadtensor/src/lib.rs`
  - `docs/design/supported-ops.md`
  - any runtime/AD docs that mention generic global context

**Step 1: Add structural tests**

Add tests that explicitly prevent:

- reintroduction of `set_global_context::<T>`
- reintroduction of multiple thread-local reverse registries

**Step 2: Update docs**

Describe:

- scoped default runtime holder
- tape-local reverse-rule store

**Step 3: Run focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor organization -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src/context extension/tenferro-dyadtensor/src/reverse_tape docs/design/supported-ops.md
git commit -m "docs: document dyadtensor registry redesign"
```

### Task 5: Reread, verify, and prepare PR

**Files:**
- Review all touched files

**Step 1: Reread for similar issues**

Search for:

```bash
rg -n 'set_global_context|with_global_context|try_with_global_context|MissingGlobalContext|ContextTypeMismatch|thread_local!' extension/tenferro-dyadtensor/src -g '*.rs'
```

Fix any similar leftover ad hoc patterns in active production code.

**Step 2: Run full required verification**

```bash
cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-release cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-cov cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-dyadtensor-registry-doc cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-dyadtensor-registry-doc/doc
```

**Step 3: Commit final cleanup**

```bash
git add -A
git commit -m "refactor: redesign dyadtensor registries"
```

**Step 4: Create PR**

```bash
git push -u origin refactor/dyadtensor-registry
gh pr create --base main --head refactor/dyadtensor-registry --title "refactor: redesign dyadtensor registries" --body "## Summary
- replace generic dyadtensor global context with a runtime-only holder
- collapse reverse-tape registries into a tape-local rule store
- update tests and docs around the new registry model

## Verification
- cargo fmt --all --check
- cargo test --workspace --release
- cargo llvm-cov --workspace --json --output-path coverage.json
- python3 scripts/check-coverage.py coverage.json
- cargo doc --workspace --no-deps
- python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-dyadtensor-registry-doc/doc

Generated with [Claude Code](https://claude.com/claude-code)"
gh pr merge --auto --squash --delete-branch
```
