# DyAdTensor Module-Local Test Directories Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move inline unit tests in `tenferro-dyadtensor` out of `src/**` production files into module-local test directories, and codify the rule in `AGENTS.md`.

**Architecture:** Keep runtime behavior unchanged and treat this as a structural refactor. Each affected Rust source file keeps only `#[cfg(test)] mod tests;`, while the test bodies move into sibling `tests/` directories that preserve access to private module items through normal Rust module nesting.

**Tech Stack:** Rust 2021, Cargo test harness, repository conventions in `AGENTS.md`

---

### Task 1: Codify the repository rule

**Files:**
- Modify: `AGENTS.md`

**Step 1: Add the unit test organization rule**

Insert a new `Unit Test Organization` subsection near `File Organization` covering:

- keep production files focused on production code
- avoid inline `#[cfg(test)]` blocks in normal Rust modules
- prefer module-local test directories (`foo/tests/...`)
- reserve crate-root `tests/` for integration tests
- optimize for clean reading context for humans and AI

**Step 2: Run formatting-sensitive checks by inspection**

Confirm the added prose is plain Markdown and does not disturb surrounding sections.

### Task 2: Establish module-local test directories for top-level modules

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ad_value.rs`
- Modify: `extension/tenferro-dyadtensor/src/context.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Modify: `extension/tenferro-dyadtensor/src/reverse_tape.rs`
- Modify: `extension/tenferro-dyadtensor/src/runtime.rs`
- Create: `extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/context/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/reverse_tape/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/runtime/tests/mod.rs`

**Step 1: Move each inline test block**

For each file above:

- replace `#[cfg(test)] mod tests { ... }` with `#[cfg(test)] mod tests;`
- copy the previous test-body contents into the corresponding `tests/mod.rs`
- preserve existing helper functions and `use` statements

**Step 2: Verify module path correctness**

Run:

```bash
cargo test -p tenferro-dyadtensor dyn_ad_value_mode_and_tangent -- --exact
```

Expected: the test compiles and passes from the new module-local location.

### Task 3: Establish module-local test directories for `api`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/chainrules_api.rs`
- Create: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/api/ad/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/api/chainrules_api/tests/mod.rs`

**Step 1: Move the inline test blocks**

Apply the same transformation:

- production file keeps only `#[cfg(test)] mod tests;`
- moved test content lives under the sibling `tests/` directory

**Step 2: Split the largest moved suite if the grouping is obvious**

For `src/api/ad.rs`, prefer a `tests/mod.rs` that re-exports a few concern-based
submodules when the split is mechanically clear, such as runtime checks,
builder smoke coverage, and AD mode propagation. Avoid semantic rewrites.

**Step 3: Run targeted verification**

Run:

```bash
cargo test -p tenferro-dyadtensor run_requires_runtime -- --exact
cargo test -p tenferro-dyadtensor qr_rrule_matches_linalg_backend -- --exact
```

Expected: both tests compile and pass.

### Task 4: Establish module-local test directories for `structured`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/structured/layout.rs`
- Modify: `extension/tenferro-dyadtensor/src/structured/einsum.rs`
- Modify: `extension/tenferro-dyadtensor/src/structured/meta.rs`
- Create: `extension/tenferro-dyadtensor/src/structured/layout/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/structured/einsum/tests/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/structured/meta/tests/mod.rs`

**Step 1: Move the inline test blocks**

Repeat the same extraction pattern for the three `structured` modules.

**Step 2: Run targeted verification**

Run:

```bash
cargo test -p tenferro-dyadtensor structured_output_roundtrips_through_dense -- --exact
cargo test -p tenferro-dyadtensor reverse_subscripts_swaps_input_and_output_labels -- --exact
```

Expected: both tests compile and pass.

### Task 5: Final verification and cleanup

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/**`
- Modify: `AGENTS.md`

**Step 1: Run formatter**

Run:

```bash
cargo fmt --all
```

Expected: formatting completes without errors.

**Step 2: Run crate tests**

Run:

```bash
cargo test -p tenferro-dyadtensor
```

Expected: all `tenferro-dyadtensor` unit and integration tests pass.

**Step 3: Inspect final diff**

Run:

```bash
git diff -- AGENTS.md extension/tenferro-dyadtensor
```

Expected: only test relocation and the new repository rule remain.
