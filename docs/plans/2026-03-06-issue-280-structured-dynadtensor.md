# Structured DynAdTensor Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the dense-only root `AdTensor` / `DynAdTensor` with a canonical structured carrier that supports at least `Dense` and `Diag`, removes the public `partial_diag` family, and preserves AD metadata and reverse correctness in the structured payload space.

**Architecture:** Introduce `StructuredTensor<T>` as the root payload type, move reusable `partial_diag` logic into internal `structured::*` modules, and rebase root `AdTensor<T>` / `DynAdTensor` plus reverse tape plumbing on top of `AdValue<StructuredTensor<T>>`. Keep native structured support for eager tensor ops and use explicit dense fallback only inside unsupported linalg wrappers.

**Tech Stack:** Rust, `tenferro-dyadtensor`, `tenferro-einsum`, `tenferro-linalg`, `tenferro-tensor`, existing reverse tape registry and AD wrappers

---

### Task 1: Lock Root StructuredTensor API with Failing Tests

**Files:**
- Create: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn root_structured_tensor_supports_dense_and_diag_layouts() {
    let dense = StructuredTensor::from_dense(tensor2(&[1.0, 2.0, 3.0, 4.0], 2, 2));
    assert_eq!(dense.logical_dims(), &[2, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert!(dense.is_dense());

    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    assert_eq!(diag.logical_dims(), &[2, 2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert!(diag.is_diag());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: FAIL because `StructuredTensor` does not exist in the root API yet.

**Step 3: Write minimal implementation**

Create a new internal module layout:

```rust
pub struct StructuredTensor<T: Scalar> {
    payload: Tensor<T>,
    logical_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}
```

Export it from the crate root without yet rebasing `AdTensor`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/lib.rs \
        extension/tenferro-dyadtensor/src/structured \
        extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs
git commit -m "feat(dyadtensor): add root structured tensor payload"
```

### Task 2: Move partial_diag Layout and Metadata Internals Under structured/*

**Files:**
- Create: `extension/tenferro-dyadtensor/src/structured/mod.rs`
- Create: `extension/tenferro-dyadtensor/src/structured/layout.rs`
- Create: `extension/tenferro-dyadtensor/src/structured/meta.rs`
- Modify: `extension/tenferro-dyadtensor/src/partial_diag/meta.rs`
- Modify: `extension/tenferro-dyadtensor/src/partial_diag/typed.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`

**Step 1: Write the failing test**

Add validation and canonicalization coverage:

```rust
#[test]
fn structured_tensor_new_canonicalizes_axis_classes() {
    let payload = vector(&[1.0, 2.0]);
    let x = StructuredTensor::new(vec![2, 2], vec![4, 4], payload).unwrap();
    assert_eq!(x.axis_classes(), &[0, 0]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests structured_tensor_new_canonicalizes_axis_classes`
Expected: FAIL because root `StructuredTensor::new` does not yet own the old `partial_diag` layout helpers.

**Step 3: Write minimal implementation**

Move reusable functions out of `partial_diag` into internal structured files:

```rust
pub(crate) fn canonicalize_axis_classes(classes: &[usize]) -> Vec<usize> { ... }
pub(crate) fn validate_layout<T: Scalar>(...) -> Result<()> { ... }
pub(crate) fn plan_axis_classes_for_subscripts(...) -> Result<AxisClassMergePlan> { ... }
```

Keep `partial_diag` compiling temporarily by delegating to the new internal modules.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/structured \
        extension/tenferro-dyadtensor/src/partial_diag/meta.rs \
        extension/tenferro-dyadtensor/src/partial_diag/typed.rs \
        extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs
git commit -m "refactor(dyadtensor): move structured layout helpers under root modules"
```

### Task 3: Rebase Root AdTensor on AdValue<StructuredTensor<T>>

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ad_value.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`

**Step 1: Write the failing test**

Add root AD construction coverage:

```rust
#[test]
fn ad_tensor_wraps_structured_payload_and_reports_logical_dims() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let x = AdTensor::new_primal(diag);
    assert_eq!(x.dims(), &[2, 2]);
    assert!(x.primal().is_diag());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests ad_tensor_wraps_structured_payload_and_reports_logical_dims`
Expected: FAIL because `AdTensor<T>` still wraps `Tensor<T>`.

**Step 3: Write minimal implementation**

Change the root carrier:

```rust
pub struct AdTensor<T: Scalar>(pub AdValue<StructuredTensor<T>>);
```

Update the accessor surface so existing callers can still ask for:

- logical dims
- payload metadata
- mode / tangent / node / tape

Adjust root API wrappers to consume `StructuredTensor<T>` instead of `Tensor<T>`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ad_value.rs \
        extension/tenferro-dyadtensor/src/lib.rs \
        extension/tenferro-dyadtensor/src/api/mod.rs \
        extension/tenferro-dyadtensor/src/api/ad.rs \
        extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs
git commit -m "refactor(dyadtensor): rebase root ad tensor on structured payloads"
```

### Task 4: Rebase DynAdTensor and Root Constructors on Structured Payloads

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`

**Step 1: Write the failing test**

Add dynamic root coverage:

```rust
#[test]
fn dyn_ad_tensor_carries_diag_payload_without_dense_materialization() {
    let diag = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let dyn_x: DynAdTensor = AdTensor::new_primal(diag).into();
    assert_eq!(dyn_x.dims(), &[2, 2]);
    assert!(dyn_x.primal().unwrap().is_diag());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests dyn_ad_tensor_carries_diag_payload_without_dense_materialization`
Expected: FAIL because `DynAdTensor` still assumes dense `Tensor<T>` payloads.

**Step 3: Write minimal implementation**

Update typed and dynamic dispatch:

```rust
pub enum DynAdTensor {
    F32(AdTensor<f32>),
    F64(AdTensor<f64>),
    C32(AdTensor<Complex32>),
    C64(AdTensor<Complex64>),
}
```

but make each variant carry `AdValue<StructuredTensor<T>>` through the new root
`AdTensor<T>` implementation.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/dyn_types.rs \
        extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs
git commit -m "refactor(dyadtensor): rebase dynamic ad tensors on structured payloads"
```

### Task 5: Port Structured Native Eager Ops and Reverse Rules

**Files:**
- Create: `extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Modify: `extension/tenferro-dyadtensor/src/ad_value.rs`
- Modify: `extension/tenferro-dyadtensor/src/reverse_tape.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs`

**Step 1: Write the failing test**

Cover dense and diag in root structured space:

```rust
#[test]
fn diag_scale_reverse_keeps_diag_cotangent_space() {
    let x = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(vector(&[2.0, 3.0]), 2).unwrap(),
        NodeId(1),
        TapeId(7),
        None,
    );
    let a = DynAdScalar::from(2.0_f64);
    let y = DynAdTensor::from(x).scale(&a).unwrap();
    let grads = ad::pullback_wrt_mixed(&y, &diag_cotangent(&[1.0, 1.0])).unwrap();
    assert!(grads.input(0).unwrap().is_diag());
}
```

Add similar tests for:

- `axpby`
- `real_part`
- `imag_part`
- `compose_complex`
- `conj`
- `sum`

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_reverse_tests`
Expected: FAIL because reverse rules still operate on plain `Tensor<T>` cotangents.

**Step 3: Write minimal implementation**

Introduce structured-aware helpers:

```rust
fn map_ad_tensor_same_type_linear_structured<T>(...) -> Result<AdTensor<T>> { ... }
fn map_ad_tensor_mixed_linear_structured<TIn, TOut>(...) -> Result<AdTensor<TOut>> { ... }
```

Update reverse tape registration to store and traverse `StructuredTensor<T>`
cotangents rather than dense `Tensor<T>` values for root tensor nodes.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_reverse_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ad_value.rs \
        extension/tenferro-dyadtensor/src/dyn_types.rs \
        extension/tenferro-dyadtensor/src/reverse_tape.rs \
        extension/tenferro-dyadtensor/src/api/mod.rs \
        extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs
git commit -m "fix(dyadtensor): preserve structured cotangent space in eager ops"
```

### Task 6: Port Structured Einsum and Sum Through Root Carrier

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/structured/einsum.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs`

**Step 1: Write the failing test**

Add root-einsum coverage:

```rust
#[test]
fn diag_einsum_stays_structured_in_root_api() {
    let a = AdTensor::new_primal(StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap());
    let b = AdTensor::new_primal(StructuredTensor::from_diagonal_vector(vector(&[3.0, 4.0]), 2).unwrap());
    let out = tenferro_dyadtensor::ad::einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert!(out.primal().is_diag());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_reverse_tests diag_einsum_stays_structured_in_root_api`
Expected: FAIL because root einsum still assumes dense payloads and `partial_diag` remains separate.

**Step 3: Write minimal implementation**

Move `partial_diag` contraction logic into `structured/einsum.rs` and call it
from root AD builders when operand payloads are structured. Ensure `sum` also
reduces from `StructuredTensor<T>` and returns a scalar with correct reverse
bridges.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_reverse_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/structured/einsum.rs \
        extension/tenferro-dyadtensor/src/api/mod.rs \
        extension/tenferro-dyadtensor/src/api/ad.rs \
        extension/tenferro-dyadtensor/src/dyn_types.rs \
        extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs
git commit -m "feat(dyadtensor): route root einsum through structured carriers"
```

### Task 7: Add Explicit Dense Fallback for Linalg Wrappers

**Files:**
- Create: `extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn structured_diag_input_can_flow_through_qr_via_internal_dense_fallback() {
    let x = AdTensor::new_primal(StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap());
    let out = tenferro_dyadtensor::ad::qr(&x).unwrap();
    assert!(out.q.primal().is_dense());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_linalg_fallback_tests`
Expected: FAIL because linalg wrappers take dense payloads directly.

**Step 3: Write minimal implementation**

For unsupported structured linalg inputs:

```rust
let dense_primal = structured.to_dense()?;
let dense_tangent = tangent.map(|t| t.to_dense()).transpose()?;
```

Call the existing dense backend and wrap outputs with
`StructuredTensor::from_dense(...)` before re-entering the root AD carrier.

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_linalg_fallback_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/mod.rs \
        extension/tenferro-dyadtensor/src/api/ad.rs \
        extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs
git commit -m "refactor(dyadtensor): add explicit dense fallback for structured linalg"
```

### Task 8: Remove public partial_diag and Migrate Docs/Tests to Root API

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Delete: `extension/tenferro-dyadtensor/src/partial_diag/mod.rs`
- Delete: `extension/tenferro-dyadtensor/src/partial_diag/typed.rs`
- Delete: `extension/tenferro-dyadtensor/src/partial_diag/dyn_tensor.rs`
- Delete: `extension/tenferro-dyadtensor/src/partial_diag/meta.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/ad_value.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs`
- Test: `extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs`

**Step 1: Write the failing test**

Add doc-oriented root usage coverage that would replace old `partial_diag`
examples:

```rust
#[test]
fn root_api_exposes_structured_diag_without_partial_diag_module() {
    let x = StructuredTensor::from_diagonal_vector(vector(&[1.0, 2.0]), 2).unwrap();
    let ad = AdTensor::new_primal(x);
    assert!(ad.primal().is_diag());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests root_api_exposes_structured_diag_without_partial_diag_module`
Expected: FAIL until root docs/examples are fully migrated.

**Step 3: Write minimal implementation**

- remove `pub mod partial_diag;`
- remove re-exports that mention `partial_diag::*`
- move any remaining useful helper exports to root or `structured::*`
- rewrite public docs and examples to use root `StructuredTensor` and root
  `AdTensor`

**Step 4: Run test to verify it passes**

Run: `cargo test -p tenferro-dyadtensor --test structured_tensor_root_tests`
Expected: PASS

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/lib.rs \
        extension/tenferro-dyadtensor/src/api/ad.rs \
        extension/tenferro-dyadtensor/src/api/mod.rs \
        extension/tenferro-dyadtensor/src/ad_value.rs \
        extension/tenferro-dyadtensor/src/dyn_types.rs \
        extension/tenferro-dyadtensor/tests/structured_tensor_root_tests.rs \
        extension/tenferro-dyadtensor/tests/structured_reverse_tests.rs \
        extension/tenferro-dyadtensor/tests/structured_linalg_fallback_tests.rs
git add -u extension/tenferro-dyadtensor/src/partial_diag
git commit -m "refactor(dyadtensor): remove public partial diag api"
```

### Task 9: Full Verification and PR Readiness

**Files:**
- Modify: `docs/plans/2026-03-06-issue-280-structured-dynadtensor-design.md`
- Modify: `docs/plans/2026-03-06-issue-280-structured-dynadtensor.md`
- Test: workspace verification commands

**Step 1: Run crate-focused regression suite**

Run: `CARGO_BUILD_JOBS=1 cargo nextest run --release -p tenferro-dyadtensor`
Expected: PASS

**Step 2: Run repository-required verification**

Run: `cargo fmt --all --check`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS

Run: `cargo llvm-cov --workspace --json --output-path coverage.json`
Expected: PASS

Run: `python3 scripts/check-coverage.py coverage.json`
Expected: PASS

**Step 3: Confirm worktree is clean except intended changes**

Run: `git status --short --branch`
Expected: only the intended issue #280 files are modified.

**Step 4: Commit final cleanup if needed**

```bash
git add -A
git commit -m "test(dyadtensor): finalize structured root migration coverage"
```

**Step 5: Request review**

Use `@superpowers/requesting-code-review` before opening the PR.
