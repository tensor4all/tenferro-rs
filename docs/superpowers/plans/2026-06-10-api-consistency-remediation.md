# API Consistency Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the first API consistency audit from candidate findings to enforceable release-freeze checks backed by a fixed API convention specification under `docs/spec/`.

**Architecture:** Put durable inspection rules in `docs/spec/api-conventions.md`, keep release-cleanup rationale in `docs/design/api-and-convention-freeze.md`, and make `scripts/check-api-consistency.py` implement the spec. Treat `traced_tensor` as an intentional surface namespace paired with `eager_tensor`; then clean hidden AD bridge names and replace dtype-specific internal op constructors with one sealed generic constructor.

**Tech Stack:** Rust workspace crates, `docs/spec/`, `scripts/check-api-consistency.py`, `cargo test`, `cargo doc`, repository design docs, and release-freeze public API rules.

---

## Recommended Order

1. Create `docs/spec/api-conventions.md` as the fixed source of truth for API inspection rules.
2. Keep `traced_tensor` as a release API namespace because it names the traced surface paired with `eager_tensor`.
3. Refine `scripts/check-api-consistency.py` so `traced_tensor` module paths stop masking real function-name findings.
4. Rename the `tenferro_runtime::ad_support` bridge helpers that start with `traced_`.
5. Replace `StdTensorOp::constant_f64` and peer constructors with `StdTensorOp::constant<T>()`.
6. Turn `scripts/check-api-consistency.py --fail-on-findings` into a release gate once the report reaches zero convention findings.

This order keeps user-facing namespace churn out of the first cleanup batch and focuses code changes on places where the current names are redundant or copy dtype plumbing.

### Task 1: Establish Fixed API Convention Spec

**Files:**
- Create: `docs/spec/api-conventions.md`
- Modify: `docs/spec/index.md`
- Modify: `docs/design/api-and-convention-freeze.md`

- [ ] **Step 1: Create the normative API convention spec**

Create `docs/spec/api-conventions.md` with:

```markdown
# API Convention Specification

**Date:** 2026-06-10
**Parent:** [`../index.md`](../index.md)
**Related:** [`tensor-semantics.md`](tensor-semantics.md), [`backend-contract.md`](backend-contract.md), [`extension-op.md`](extension-op.md), [`../design/api-and-convention-freeze.md`](../design/api-and-convention-freeze.md)

---

## Purpose

This document is the normative specification for public API naming, module
shape, feature naming, and documentation-surface checks. It owns the fixed
rules implemented by `scripts/check-api-consistency.py`.

Release cleanup rationale and migration order live in
`../design/api-and-convention-freeze.md`. This file owns the rules that remain
after the cleanup is complete.

---

## Public API Strata

Every exported item belongs to one stratum:

| Stratum | Meaning | Requirement |
| --- | --- | --- |
| Release API | Intended for downstream users. | Public rustdoc contract and runnable example when the item is callable. |
| Owner-scoped bridge | Required only because Rust crate boundaries separate owning implementation units. | Narrow API, no guide promotion, and documented owner. |
| Experimental or unsupported | Visible because an incomplete capability is intentionally exposed. | Explicit unsupported behavior and error contract. |
| Internal | Tests, planning, lowering, dispatch, cache plumbing, provider selection, or backend glue. | Private or `pub(crate)` unless a documented owner-scoped bridge is required. |

The default stratum is internal.

## Naming Rules

1. There is no root `tenferro` facade crate. User-facing docs and examples must
   import direct crates such as `tenferro_runtime`, `tenferro_ad`,
   `tenferro_gpu`, `tenferro_einsum`, `tenferro_linalg`, and `tenferro_fft`.
2. Tensor operation names use unsuffixed names for owned compact tensor inputs.
3. `_read` is reserved for APIs that explicitly accept borrowed read-oriented
   inputs such as `TensorRead`.
4. `_view` is reserved for metadata-only layout operations. Operations that
   allocate, canonicalize, execute kernels, transfer data, or materialize data
   must not use `_view`.
5. Scalar constructors use generic `TensorScalar`-bounded entry points instead
   of dtype-specific public functions such as `constant_f64`.
6. Traced tensor method and free-function names do not use a `traced_` prefix.
7. Public module namespaces may use `traced_tensor` when they identify the
   traced tensor surface as a peer of `eager_tensor`. The namespace is a
   surface stratum, not an operation-name prefix.
8. User-facing backend features use concrete backend family names such as
   `cuda` and `rocm`. Public crates must not expose a vague `gpu` feature.
9. Optional operation-specific AD support belongs behind an `autodiff` feature
   in the owning operation crate.

## API Shape Rules

1. Unary single-output traced ops are methods on `TracedTensor`.
2. Binary single-output ops use operator overloads when the operator is natural;
   otherwise they use methods.
3. Multi-output linalg and decomposition ops are free functions.
4. Einsum is a free function owned by `tenferro-einsum`.
5. Standard operation families are first-class crates, not modules under a
   broad facade.

## Documentation Checks

1. `README.md`, `docs/guides/`, and `docs/getting-started/` must not reference
   internal crates, internal graph/IR vocabulary, or deleted public paths.
2. User-facing examples must use direct public crates and must not rely on a
   root facade path.
3. Flat-buffer constructors, exports, examples, and FFI contracts must state or
   encode column-major layout expectations.
4. Rustdoc examples for release APIs must compile and run as doctests unless
   they intentionally use `compile_fail`.

## Checker Mapping

`scripts/check-api-consistency.py` implements these automated checks:

| Check category | Spec rule |
| --- | --- |
| `traced_prefix` | Naming rule 6, limited to public function and method names. |
| `read_suffix_without_read_input` | Naming rule 3. |
| `per_dtype_constructor` | Naming rule 5. |
| `public_gpu_feature` | Naming rule 8, limited to published crates. |
| `facade_path_in_user_docs` | Naming rule 1 and documentation check 2. |
| `internal_jargon_in_user_docs` | Documentation check 1. |

The concept-family matrices emitted by the checker are review aids. A matrix
difference becomes a finding only when the relevant spec, design doc, or
rustdoc contract does not explain the difference.
```

- [ ] **Step 2: Register the spec in the spec index**

In `docs/spec/index.md`, add this row to the table:

```markdown
| [api-conventions.md](./api-conventions.md) | Public API naming, module shape, feature naming, documentation-surface checks, and checker mapping |
```

- [ ] **Step 3: Point the design doc at the fixed spec**

In `docs/design/api-and-convention-freeze.md`, under `## Source Of Truth`, add:

```markdown
API convention rules that should remain stable after the release cleanup live
in `docs/spec/api-conventions.md`. This design document records the cleanup
posture, triage workflow, and migration rationale.
```

- [ ] **Step 4: Commit the fixed spec**

Run:

```bash
git add docs/spec/api-conventions.md docs/spec/index.md docs/design/api-and-convention-freeze.md
git commit -m "docs: specify api convention rules"
```

Expected: commit succeeds.

### Task 2: Fix Traced Namespace Audit Semantics

**Files:**
- Modify: `scripts/check-api-consistency.py`

- [ ] **Step 1: Include getting-started docs in user-facing scans**

In `scripts/check-api-consistency.py`, replace:

```python
def user_facing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    docs = [root / "README.md"]
    guides = root / "docs" / "guides"
    if guides.exists():
        docs.extend(sorted(guides.rglob("*.md")))
    return [path for path in docs if path.exists()]
```

with:

```python
def user_facing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    docs = [root / "README.md"]
    for relative in ("docs/guides", "docs/getting-started"):
        directory = root / relative
        if directory.exists():
            docs.extend(sorted(directory.rglob("*.md")))
    return [path for path in docs if path.exists()]
```

- [ ] **Step 2: Narrow the checker to function names**

In `scripts/check-api-consistency.py`, replace:

```python
        if item.name.startswith("traced_"):
            findings.append(
                Finding(
                    "traced_prefix",
                    location,
                    f"`{item.name}` is public",
                    "Traced tensor APIs should not use a `traced_` prefix.",
                )
            )
```

with:

```python
        if item.kind == "fn" and item.name.startswith("traced_"):
            findings.append(
                Finding(
                    "traced_prefix",
                    location,
                    f"`{item.name}` is public",
                    "Traced tensor function and method names should not use a `traced_` prefix.",
                )
            )
```

- [ ] **Step 3: Verify the checker now reports only function-level traced prefixes**

Run:

```bash
python3 -m py_compile scripts/check-api-consistency.py
python3 scripts/check-api-consistency.py --output /tmp/tenferro-api-consistency.md
```

Expected:

```text
api-consistency-report: 12 crates, 1382 lexically public items, 14 convention findings
```

The four removed findings are the `pub mod traced_tensor` entries in `tenferro-runtime`, `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft`.

- [ ] **Step 4: Commit the checker precision change**

Run:

```bash
git add scripts/check-api-consistency.py
git commit -m "tools: refine traced namespace audit"
```

Expected: commit succeeds.

### Task 3: Rename AD Support Bridge Helpers

**Files:**
- Modify: `crates/tenferro-runtime/src/ad_support.rs`
- Modify: `crates/tenferro-ad/src/traced.rs`

- [ ] **Step 1: Rename bridge functions at the owning boundary**

In `crates/tenferro-runtime/src/ad_support.rs`, rename these public bridge helpers:

```rust
pub fn traced_tensor_from_parts(parts: TracedTensorParts) -> TracedTensor
pub fn traced_shape_hint(tensor: &TracedTensor) -> Option<Vec<SymDim>>
pub fn traced_inputs_map(tensor: &TracedTensor) -> Arc<HashMap<TensorInputKey, Arc<Tensor>>>
pub fn traced_extra_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>>
pub fn traced_checkpoint_chain(tensor: &TracedTensor) -> Option<Arc<CheckpointNode>>
pub fn traced_metadata_scopes(tensor: &TracedTensor) -> &[Arc<RuntimeMetadataScope>]
pub fn traced_resolve_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>>
pub fn checkpoint_traced_tensor(tensor: &mut TracedTensor, data: Arc<Tensor>)
```

to:

```rust
pub fn tensor_from_parts(parts: TracedTensorParts) -> TracedTensor
pub fn shape_hint(tensor: &TracedTensor) -> Option<Vec<SymDim>>
pub fn inputs_map(tensor: &TracedTensor) -> Arc<HashMap<TensorInputKey, Arc<Tensor>>>
pub fn extra_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>>
pub fn checkpoint_chain(tensor: &TracedTensor) -> Option<Arc<CheckpointNode>>
pub fn metadata_scopes(tensor: &TracedTensor) -> &[Arc<RuntimeMetadataScope>]
pub fn resolve_roots(tensor: &TracedTensor) -> Vec<Arc<Graph<StdTensorOp>>>
pub fn checkpoint_tensor(tensor: &mut TracedTensor, data: Arc<Tensor>)
```

Keep the existing bodies unchanged. The `ad_support` module name already supplies the bridge context, so each helper name can describe the borrowed part.

- [ ] **Step 2: Update `tenferro-ad` imports**

In `crates/tenferro-ad/src/traced.rs`, replace the `ad_support` import group with:

```rust
use tenferro_runtime::ad_support::{
    checkpoint_chain, checkpoint_tensor, extra_roots, inputs_map, leaf_input_key,
    linear_input_key, metadata_scopes, metadata_scopes_with_new, resolve_roots, shape_hint,
    tensor_from_parts, tensor_meta_from_tensor, TracedTensorParts,
};
```

- [ ] **Step 3: Update `tenferro-ad` call sites**

In `crates/tenferro-ad/src/traced.rs`, make these exact replacements:

```text
checkpoint_traced_tensor( -> checkpoint_tensor(
traced_checkpoint_chain( -> checkpoint_chain(
traced_resolve_roots( -> resolve_roots(
traced_inputs_map( -> inputs_map(
traced_extra_roots( -> extra_roots(
traced_tensor_from_parts( -> tensor_from_parts(
traced_shape_hint( -> shape_hint(
traced_metadata_scopes( -> metadata_scopes(
```

- [ ] **Step 4: Verify the AD bridge rename**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-ad traced
python3 scripts/check-api-consistency.py --output /tmp/tenferro-api-consistency.md
```

Expected:

```text
api-consistency-report: 12 crates, 1382 lexically public items, 7 convention findings
```

The remaining convention findings are the seven `StdTensorOp::constant_*` constructors.

- [ ] **Step 5: Commit the bridge rename**

Run:

```bash
git add crates/tenferro-runtime/src/ad_support.rs crates/tenferro-ad/src/traced.rs
git commit -m "refactor: rename ad support bridge helpers"
```

Expected: commit succeeds.

### Task 4: Replace Dtype-Specific Constant Constructors

**Files:**
- Modify: `crates/tenferro-internal-ops/src/std_tensor_op.rs`
- Modify: `crates/tenferro-internal-ops/src/lib.rs`
- Modify: `crates/tenferro-internal-ops/src/tests/std_tensor_op_tests.rs`
- Modify: `crates/tenferro-runtime/src/traced.rs`
- Modify: `crates/tenferro-linalg/src/ad/rules/support.rs`
- Modify: `crates/tenferro-ad/tests/compiler_wiring.rs`

- [ ] **Step 1: Add the sealed scalar encoding trait**

In `crates/tenferro-internal-ops/src/std_tensor_op.rs`, after the imports and before `tenferro_core_ops::define_std_tensor_op!();`, add:

```rust
/// Scalar values that can be encoded as tensor constant operations.
pub trait ConstantScalar: tenferro_tensor::TensorScalar + private::Sealed {
    /// Encode the scalar value as little-endian constant bytes.
    fn constant_bytes(self) -> Vec<u8>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i64 {}
    impl Sealed for i32 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

impl ConstantScalar for f64 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for f32 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for i64 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for i32 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for bool {
    fn constant_bytes(self) -> Vec<u8> {
        vec![u8::from(self)]
    }
}

impl ConstantScalar for Complex64 {
    fn constant_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&self.re.to_le_bytes());
        bytes.extend_from_slice(&self.im.to_le_bytes());
        bytes
    }
}

impl ConstantScalar for Complex32 {
    fn constant_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&self.re.to_le_bytes());
        bytes.extend_from_slice(&self.im.to_le_bytes());
        bytes
    }
}
```

- [ ] **Step 2: Add the generic constructor**

In the existing `impl StdTensorOp` block, replace all seven `constant_f64`, `constant_f32`, `constant_i64`, `constant_i32`, `constant_bool`, `constant_c64`, and `constant_c32` functions with:

```rust
    /// Create a scalar constant op from any supported tensor scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_tensor::DType;
    ///
    /// let real = StdTensorOp::constant(1.5_f64);
    /// let complex = StdTensorOp::constant(Complex64::new(1.0, -2.0));
    ///
    /// assert_eq!(real.output_dtype(&[]).unwrap(), DType::F64);
    /// assert_eq!(complex.output_dtype(&[]).unwrap(), DType::C64);
    /// ```
    pub fn constant<T: ConstantScalar>(value: T) -> Self {
        Self::Constant {
            dtype: T::dtype(),
            bytes: value.constant_bytes(),
        }
    }
```

- [ ] **Step 3: Update source call sites**

Apply these replacements:

```text
StdTensorOp::constant_f64(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_f32(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_i64(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_i32(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_bool(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_c64(x) -> StdTensorOp::constant(x)
StdTensorOp::constant_c32(x) -> StdTensorOp::constant(x)
```

When a literal would become ambiguous, add an explicit suffix, for example:

```rust
StdTensorOp::constant(1.25_f64)
StdTensorOp::constant(1.25_f32)
StdTensorOp::constant(7_i64)
StdTensorOp::constant(7_i32)
```

- [ ] **Step 4: Update crate-level docs**

In `crates/tenferro-internal-ops/src/lib.rs`, replace:

```rust
//! let op = StdTensorOp::constant_f64(2.0);
```

with:

```rust
//! let op = StdTensorOp::constant(2.0_f64);
```

- [ ] **Step 5: Update internal-op tests**

In `crates/tenferro-internal-ops/src/tests/std_tensor_op_tests.rs`, update each constant constructor call to `StdTensorOp::constant(...)`. Add this test near the existing constant byte tests:

```rust
#[test]
fn generic_constant_constructor_sets_dtype_for_each_scalar() {
    assert_eq!(StdTensorOp::constant(1.0_f64).output_dtype(&[]).unwrap(), DType::F64);
    assert_eq!(StdTensorOp::constant(1.0_f32).output_dtype(&[]).unwrap(), DType::F32);
    assert_eq!(StdTensorOp::constant(1_i64).output_dtype(&[]).unwrap(), DType::I64);
    assert_eq!(StdTensorOp::constant(1_i32).output_dtype(&[]).unwrap(), DType::I32);
    assert_eq!(StdTensorOp::constant(true).output_dtype(&[]).unwrap(), DType::Bool);
    assert_eq!(
        StdTensorOp::constant(Complex64::new(1.0, -2.0))
            .output_dtype(&[])
            .unwrap(),
        DType::C64
    );
    assert_eq!(
        StdTensorOp::constant(Complex32::new(1.0, -2.0))
            .output_dtype(&[])
            .unwrap(),
        DType::C32
    );
}
```

- [ ] **Step 6: Verify constructor cleanup**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-internal-ops
cargo test -p tenferro-runtime traced_tensor
cargo test -p tenferro-ad compiler_wiring
python3 scripts/check-api-consistency.py --fail-on-findings
```

Expected: all commands exit `0`, and the final script report contains:

```text
api-consistency-report: 12 crates, 1376 lexically public items, 0 convention findings
```

The public item count may differ by one or two if rustdoc-visible trait items are counted differently after formatting; the required invariant is zero convention findings.

- [ ] **Step 7: Commit the constructor cleanup**

Run:

```bash
git add \
  crates/tenferro-internal-ops/src/std_tensor_op.rs \
  crates/tenferro-internal-ops/src/lib.rs \
  crates/tenferro-internal-ops/src/tests/std_tensor_op_tests.rs \
  crates/tenferro-runtime/src/traced.rs \
  crates/tenferro-linalg/src/ad/rules/support.rs \
  crates/tenferro-ad/tests/compiler_wiring.rs
git commit -m "refactor: use generic tensor constant constructor"
```

Expected: commit succeeds.

### Task 5: Add Release-Gate Documentation

**Files:**
- Modify: `docs/design/api-and-convention-freeze.md`
- Modify: `docs/spec/api-conventions.md`
- Create: `docs/worklogs/2026-06-10-api-consistency-remediation.md`

- [ ] **Step 1: Add the concrete gate command to the design doc**

In `docs/design/api-and-convention-freeze.md`, under `## Enforcement Targets`, add this bullet:

```markdown
- release-freeze API convention findings via
  `python3 scripts/check-api-consistency.py --fail-on-findings`;
```

- [ ] **Step 2: Add the gate command to the fixed spec**

In `docs/spec/api-conventions.md`, under `## Checker Mapping`, after the table, add:

````markdown
The release-freeze convention gate is:

```bash
python3 scripts/check-api-consistency.py --fail-on-findings
```

This command must exit `0` before the API convention freeze is considered
green. Concept-family matrices remain review aids until a specific matrix rule
is promoted into this spec.
````

- [ ] **Step 3: Add the remediation work log**

Create `docs/worklogs/2026-06-10-api-consistency-remediation.md` with:

```markdown
# API Consistency Remediation Work Log

## Summary

This cleanup made the release-freeze API convention audit enforceable for the
first detected naming findings.

## Context Read

- `REPOSITORY_RULES.md`
- `docs/spec/api-conventions.md`
- `docs/design/api-and-convention-freeze.md`
- `scripts/check-api-consistency.py`
- `/tmp/tenferro-api-consistency.md`
- `crates/tenferro-runtime/src/ad_support.rs`
- `crates/tenferro-ad/src/traced.rs`
- `crates/tenferro-internal-ops/src/std_tensor_op.rs`

## Decisions

- Kept `traced_tensor` as a documented namespace paired with `eager_tensor`.
- Treated `traced_` prefixes on public function names as findings.
- Renamed `ad_support` bridge helpers because the module name already supplies
  the traced AD bridge context.
- Replaced dtype-specific `StdTensorOp::constant_*` constructors with one
  sealed generic constructor.

## Verification

- `python3 -m py_compile scripts/check-api-consistency.py`
- `python3 scripts/check-api-consistency.py --fail-on-findings`
- `cargo fmt --all --check`
- `cargo test -p tenferro-internal-ops`
- `cargo test -p tenferro-runtime traced_tensor`
- `cargo test -p tenferro-ad traced`
- `cargo test -p tenferro-ad compiler_wiring`

## Residual Risks

The concept-family matrices still need human triage for behavior-level
differences such as eager vs traced error surfaces, CPU vs GPU unsupported
paths, and view vs materializing operation boundaries.
```

- [ ] **Step 4: Verify docs**

Run:

```bash
python3 scripts/check-doc-snippets.py --root-dir . --check
cargo doc --workspace --no-deps
```

Expected: both commands exit `0`.

- [ ] **Step 5: Commit release-gate documentation**

Run:

```bash
git add docs/design/api-and-convention-freeze.md docs/spec/api-conventions.md docs/worklogs/2026-06-10-api-consistency-remediation.md
git commit -m "docs: record api consistency remediation gate"
```

Expected: commit succeeds.

### Task 6: Final Release-Freeze Verification

**Files:**
- No source edits in this task.

- [ ] **Step 1: Run fast release-freeze checks**

Run:

```bash
cargo fmt --all --check
python3 -m py_compile scripts/check-api-consistency.py
python3 scripts/check-api-consistency.py --fail-on-findings
python3 scripts/check-doc-snippets.py --root-dir . --check
```

Expected: all commands exit `0`.

- [ ] **Step 2: Run workspace checks**

Run:

```bash
cargo test --workspace
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands exit `0`.

- [ ] **Step 3: Review remaining concept matrices**

Run:

```bash
python3 scripts/check-api-consistency.py --output /tmp/tenferro-api-consistency.md
sed -n '/## Concept-Family Matrices/,$p' /tmp/tenferro-api-consistency.md
```

Expected: the convention findings section is empty, and concept-family matrices remain review aids for the next cleanup stream.

- [ ] **Step 4: Commit no-op verification state**

Run:

```bash
git status --short
```

Expected: no uncommitted files.

Do not create an empty commit.
