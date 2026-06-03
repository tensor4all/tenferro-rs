# Publish Prep Without Git Dependency Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare tenferro-rs for crates.io publication except for converting remaining git dependencies to registry dependencies.

**Architecture:** Move tenferro workspace packages under `crates/`, keep the workspace facade-free, share the root README across published packages, and document user-facing crates in basic-first order. Add a repository contract script to keep the layout and publish metadata from drifting.

**Tech Stack:** Rust Cargo workspaces, TOML manifests, Python standard library validation script, Markdown docs.

---

### Task 1: Add a publish-layout contract check

**Files:**
- Create: `scripts/check-publish-layout.py`

- [ ] Add a Python script that parses root and crate `Cargo.toml` files with `tomllib`, verifies workspace tenferro members live under `crates/`, rejects a root `tenferro` facade member, requires shared `readme.workspace = true`, `repository.workspace = true`, `homepage.workspace = true`, and confirms git dependency conversion is intentionally not enforced yet.
- [ ] Run it before moving crates; expected failure: workspace members still point at top-level `tenferro-*` directories.

### Task 2: Move tenferro packages under `crates/`

**Files:**
- Move: `tenferro-*` package directories to `crates/tenferro-*`
- Modify: `Cargo.toml`
- Modify: `docs/tutorial-code/Cargo.toml`
- Modify: `ext/tropical/Cargo.toml`

- [ ] Move all tenferro package directories under `crates/`.
- [ ] Update workspace `members` and `default-members` to use `crates/tenferro-*` paths.
- [ ] Update non-crates package path dependencies in docs/tutorial-code and ext/tropical.
- [ ] Keep git dependencies unchanged.

### Task 3: Add publication metadata except git dependency conversion

**Files:**
- Modify: `Cargo.toml`
- Modify: `crates/*/Cargo.toml`

- [ ] Set workspace package `publish = true` so published tenferro crates stop inheriting `publish = false`.
- [ ] Add shared `readme`, `repository`, and `homepage` workspace metadata.
- [ ] Add `readme.workspace = true`, `repository.workspace = true`, and `homepage.workspace = true` to each published tenferro package.
- [ ] Keep `docs/tutorial-code` and `ext/tropical` unpublished.

### Task 4: Update README and path-sensitive docs/scripts

**Files:**
- Modify: `README.md`
- Modify path-sensitive docs/scripts only where current paths would be misleading or broken.

- [ ] Add a crates section ordered as user-facing basic crates first: tensor, CPU/GPU backends together, runtime, AD; then standard operation extensions; then published implementation crates.
- [ ] State explicitly that there is no `tenferro` facade crate.
- [ ] Update current path examples from top-level `tenferro-*` paths to `crates/tenferro-*` where they refer to live source locations.

### Task 5: Verify

**Files:**
- No new files beyond prior tasks.

- [ ] Run `python3 scripts/check-publish-layout.py`.
- [ ] Run `cargo metadata --no-deps --format-version 1`.
- [ ] Run `cargo fmt --all --check`.
- [ ] Run `cargo test -p tenferro-tensor --lib`.
