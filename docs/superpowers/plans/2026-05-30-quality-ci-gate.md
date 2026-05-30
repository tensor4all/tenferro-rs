# Quality CI Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make clippy a required CI quality gate and remove the low-risk lint/dead-code debt that currently prevents the gate from passing.

**Architecture:** Add one isolated GitHub Actions job for clippy, then fix only local mechanical Rust lints and unused crate-private wrappers. Keep numerical behavior, public APIs, backend dispatch, and broad operation-wrapper abstractions unchanged.

**Tech Stack:** Rust, Cargo, Clippy, GitHub Actions, `rg`, existing tenferro workspace crates.

---

## File Structure

- Modify `.github/workflows/ci.yml`: add the `clippy` job for the root workspace and `ext/tropical`.
- Modify `.github/workflows/CI_gpu.yml`: add `clippy` to the non-GPU prerequisite list.
- Modify `tenferro-core-ops/src/catalog.rs`: replace the primitive count macro expression that triggers `unused_unit`.
- Modify `tenferro-tensor-core/src/layout.rs`: replace manual zero-extent scans with `contains`.
- Modify `tenferro-tensor-core/src/lib.rs`: replace manual zero-extent scans with `contains`.
- Modify `tenferro-cpu/src/elementwise.rs`: delete unused local-pool convenience wrappers for Tier2 elementwise helpers.
- Modify `tenferro-cpu/src/gemm/mod.rs`: delete unused uncached GEMM convenience wrappers.
- Modify `tenferro-cpu/src/gemm/tests.rs`: call the retained cached BLAS helper directly.

## Task 1: Verify the Red Clippy Baseline

**Files:**
- No edits.

- [ ] **Step 1: Run workspace clippy and confirm it fails for the known lints**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: FAIL with `unused_unit` in `tenferro-core-ops/src/catalog.rs` and
`manual_contains` in `tenferro-tensor-core/src/layout.rs` and
`tenferro-tensor-core/src/lib.rs`.

## Task 2: Fix Mechanical Clippy Lints

**Files:**
- Modify `tenferro-core-ops/src/catalog.rs`
- Modify `tenferro-tensor-core/src/layout.rs`
- Modify `tenferro-tensor-core/src/lib.rs`

- [ ] **Step 1: Replace the primitive count expression**

In `tenferro-core-ops/src/catalog.rs`, replace:

```rust
pub const COUNT: usize = <[()]>::len(&[$({ let _ = stringify!($variant); () }),*]);
```

with:

```rust
pub const COUNT: usize = [$(PrimitiveOpKind::$variant),*].len();
```

- [ ] **Step 2: Replace manual zero checks in `layout.rs`**

In `tenferro-tensor-core/src/layout.rs`, replace:

```rust
if shape.iter().any(|&extent| extent == 0) {
```

with:

```rust
if shape.contains(&0) {
```

and replace:

```rust
if self.shape().iter().any(|&extent| extent == 0) {
```

with:

```rust
if self.shape().contains(&0) {
```

- [ ] **Step 3: Replace manual zero checks in `lib.rs`**

In `tenferro-tensor-core/src/lib.rs`, replace each:

```rust
if shape.iter().any(|&extent| extent == 0) {
```

with:

```rust
if shape.contains(&0) {
```

and replace:

```rust
self.shape.iter().any(|&extent| extent == 0)
```

with:

```rust
self.shape.contains(&0)
```

- [ ] **Step 4: Run focused clippy for the touched crates**

Run:

```bash
cargo clippy -p tenferro-core-ops -p tenferro-tensor-core --all-targets -- -D warnings
```

Expected: PASS.

## Task 3: Remove Low-Risk Dead Code Wrappers

**Files:**
- Modify `tenferro-cpu/src/elementwise.rs`
- Modify `tenferro-cpu/src/gemm/mod.rs`
- Modify `tenferro-cpu/src/gemm/tests.rs`

- [ ] **Step 1: Confirm elementwise wrappers have no call sites**

Run:

```bash
rg -n '\btyped_abs\b|\btyped_sign\b|\btyped_maximum\b|\btyped_minimum\b|\btyped_compare\b' tenferro-cpu/src
```

Expected: only the wrapper definitions in `tenferro-cpu/src/elementwise.rs`.

- [ ] **Step 2: Delete the unused elementwise wrappers**

Delete these wrapper functions from `tenferro-cpu/src/elementwise.rs`, keeping
the corresponding `_with_pool` functions:

```rust
pub(crate) fn typed_abs<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_abs_with_pool(buffers, input))
}
```

```rust
pub(crate) fn typed_sign<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_sign_with_pool(buffers, input))
}
```

```rust
pub(crate) fn typed_maximum<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_maximum_with_pool(buffers, lhs, rhs))
}
```

```rust
pub(crate) fn typed_minimum<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_minimum_with_pool(buffers, lhs, rhs))
}
```

```rust
pub(crate) fn typed_compare<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: CompareElem,
{
    with_local_pool(|buffers| typed_compare_with_pool(buffers, lhs, rhs, dir))
}
```

- [ ] **Step 3: Confirm uncached GEMM wrappers are unused or test-only**

Run:

```bash
rg -n '\bdot_general_faer\b|\bdot_general_faer_with_conj\b|\bdot_general_blas\b|\bdot_general_blas_with_conj\b' tenferro-cpu/src
```

Expected: faer wrappers and BLAS-with-conj wrapper appear only as definitions;
`dot_general_blas` also appears in `tenferro-cpu/src/gemm/tests.rs`.

- [ ] **Step 4: Delete unused uncached GEMM wrappers**

Delete these wrapper functions from `tenferro-cpu/src/gemm/mod.rs`, keeping the
retained cached variants with the same backend implementation:

- `dot_general_faer`
- `dot_general_faer_with_conj`
- `dot_general_blas`
- `dot_general_blas_with_conj`

- [ ] **Step 5: Update the BLAS GEMM test to call the cached helper**

In `tenferro-cpu/src/gemm/tests.rs`, replace the BLAS import:

```rust
use super::dot_general_blas;
```

with:

```rust
use super::dot_general_blas_cached;
```

Then replace the test call:

```rust
let out = dot_general_blas(&mut buffers, &mut cache, &lhs, &rhs, &config)
    .expect("dot_general should succeed");
```

with:

```rust
let out = dot_general_blas_cached(&mut buffers, &mut cache, None, &lhs, &rhs, &config)
    .expect("dot_general should succeed");
```

- [ ] **Step 6: Run focused CPU clippy**

Run:

```bash
cargo clippy -p tenferro-cpu --all-targets -- -D warnings
```

Expected: PASS. If a feature-conditional helper still needs `#[allow(dead_code)]`,
leave it in place and add no broad cleanup.

## Task 4: Add the CI Clippy Gate

**Files:**
- Modify `.github/workflows/ci.yml`
- Modify `.github/workflows/CI_gpu.yml`

- [ ] **Step 1: Add a clippy job to the main CI workflow**

Add this job after `fmt` in `.github/workflows/ci.yml`:

```yaml
  clippy:
    name: clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - name: Run workspace clippy
        run: cargo clippy --workspace --all-targets -- -D warnings
      - name: Run tropical extension clippy
        run: cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
```

- [ ] **Step 2: Make the GPU workflow wait for clippy**

In `.github/workflows/CI_gpu.yml`, update:

```javascript
const required = [
  "rustfmt",
  "cargo test (blas inject)",
  "tensor core dependency boundary",
  "coverage",
  "docs-site",
  "CI gate (PR workspace tests)",
];
```

to:

```javascript
const required = [
  "rustfmt",
  "clippy",
  "cargo test (blas inject)",
  "tensor core dependency boundary",
  "coverage",
  "docs-site",
  "CI gate (PR workspace tests)",
];
```

## Task 5: Full Local Verification

**Files:**
- No edits unless verification exposes a real regression.

- [ ] **Step 1: Check formatting**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS.

- [ ] **Step 2: Run root workspace clippy**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 3: Run tropical extension clippy**

Run:

```bash
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
```

Expected: PASS.

- [ ] **Step 4: Run root workspace tests**

Run:

```bash
cargo test --workspace
```

Expected: PASS.

- [ ] **Step 5: Run tropical extension tests**

Run:

```bash
cargo test --manifest-path ext/tropical/Cargo.toml
```

Expected: PASS.

## Task 6: Commit the Implementation

**Files:**
- All modified source and workflow files from Tasks 2-4.

- [ ] **Step 1: Review the diff**

Run:

```bash
git diff --stat
git diff --check
git status --short --branch
```

Expected: no whitespace errors, and only the planned files are modified.

- [ ] **Step 2: Commit**

Run:

```bash
git add .github/workflows/ci.yml .github/workflows/CI_gpu.yml \
  tenferro-core-ops/src/catalog.rs \
  tenferro-tensor-core/src/layout.rs \
  tenferro-tensor-core/src/lib.rs \
  tenferro-cpu/src/elementwise.rs \
  tenferro-cpu/src/gemm/mod.rs \
  tenferro-cpu/src/gemm/tests.rs
git commit -m "ci: gate rust clippy warnings"
```

Expected: one implementation commit after the spec and plan commits.
