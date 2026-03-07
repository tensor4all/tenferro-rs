# Homogeneous Tape and PyTorch-Style Scalar AD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `DynTape` and the heterogeneous AD query surface, keep `Tape<V>` as the only reverse-mode tape API, and update current docs to describe homogeneous custom-type graphs plus PyTorch-style rank-0 tensor scalar semantics.

**Architecture:** Keep the generic `Differentiable` / `Tape<V>` model and delete only the heterogeneous runtime-erased path. Preserve custom downstream graphs such as `Tape<MyType>`, keep implicit reverse seeds tied to `num_elements() == 1`, and document tensor-operation scalars as rank-0 tensors (`shape=[]`) rather than shape `[1]`.

**Tech Stack:** Rust, `extern/chainrules`, workspace docs under `docs/design`, `docs/AD`, and `docs/api_index.md`

---

### Task 1: Lock the Remaining Monomorphic Contract with Regression Tests

**Files:**
- Modify: `extern/chainrules/tests/chainrules_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs`
- Test: `extern/chainrules/tests/chainrules_tests.rs`
- Test: `extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs`

**Step 1: Write the regression tests**

Add a homogeneous custom-type `Tape<V>` test in `extern/chainrules/tests/chainrules_tests.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarBox(f64);

impl Differentiable for ScalarBox {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent { Self(0.0) }
    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }
    fn num_elements(&self) -> usize { 1 }
    fn seed_cotangent(&self) -> Self::Tangent { Self(1.0) }
}

#[test]
fn tape_pullback_supports_homogeneous_custom_type() {
    let tape = Tape::<ScalarBox>::new();
    let x = tape.leaf(ScalarBox(2.0));
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), ScalarBox(1.0));
}
```

Add a single-element custom output contract test in
`extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
struct SingleSlot(f64);

impl Add for SingleSlot { /* fieldwise add */ }
impl Mul for SingleSlot { /* fieldwise mul */ }

impl Differentiable for SingleSlot {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent { Self(0.0) }
    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }
    fn num_elements(&self) -> usize { 1 }
    fn seed_cotangent(&self) -> Self::Tangent { Self(1.0) }
}

#[test]
fn single_element_custom_output_can_omit_seed_grad() {
    let x = Variable::new(SingleSlot(3.0)).requires_grad_(true).unwrap();
    let y = autograd::square(&x).unwrap();

    y.backward(BackwardOptions::default()).unwrap();
    assert_eq!(x.grad(), Some(SingleSlot(6.0)));
}
```

**Step 2: Run the focused tests**

Run:

```bash
cargo test -p chainrules --test chainrules_tests --test autodiff_next_seed_and_output_contract_tests
```

Expected: PASS before the `DynTape` removal starts, and keep passing throughout the refactor. These tests are the guardrails that prove `Tape<V>` still supports homogeneous custom types and that implicit seeding remains tied to `num_elements() == 1`.

**Step 3: Commit**

```bash
git add extern/chainrules/tests/chainrules_tests.rs \
        extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs
git commit -m "test(chainrules): lock homogeneous tape contract"
```

### Task 2: Remove the Heterogeneous DynTape Surface from `chainrules`

**Files:**
- Modify: `extern/chainrules/src/lib.rs`
- Modify: `extern/chainrules/tests/autodiff_next_api_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_contract_errors.rs`
- Test: `extern/chainrules/src/lib.rs`
- Test: `extern/chainrules/tests/autodiff_next_api_tests.rs`

**Step 1: Record the current unsupported surface with a search check**

Run:

```bash
rg -n "DynTape|DynVariable|DynTangent|DynBackwardOptions|DynHvpOptions|grad_dyn_tangent|grad_dyn_variable" \
  extern/chainrules/src/lib.rs \
  extern/chainrules/tests/autodiff_next_api_tests.rs \
  extern/chainrules/tests/autodiff_next_contract_errors.rs
```

Expected: multiple matches. This is the removal checklist.

**Step 2: Remove the public dyn types and dyn query APIs**

Delete from `extern/chainrules/src/lib.rs`:

- the heterogeneous tape registry and `DynTapeInner`
- `DynTape`
- `DynVariable`
- `DynTangent`
- `DynGradients`
- `DynHvpResult`
- `DynBackwardOptions`
- `DynHvpOptions`
- `autograd::grad_dyn_tangent`
- `autograd::grad_dyn_variable`

Also remove now-unused imports such as:

```rust
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
```

Preserve the monomorphic API:

```rust
pub struct Tape<V: Differentiable> { ... }
pub struct TrackedTensor<V: Differentiable> { ... }
pub struct DualTensor<V: Differentiable> { ... }
pub struct Variable<V: Differentiable> { ... }
pub struct BackwardOptions<V: Differentiable> { ... }
```

Update `extern/chainrules/tests/autodiff_next_api_tests.rs` to assert only the
monomorphic public surface:

```rust
use chainrules::{autograd, AutodiffError, AutogradContext, BackwardOptions, Tape, Variable};

#[test]
fn monomorphic_api_surface_exists() {
    let _ = BackwardOptions::<f64>::default();
    let _ = Tape::<f64>::new();
    let v = Variable::new(1.0_f64);
    assert!(!v.requires_grad());
}
```

Update `extern/chainrules/tests/autodiff_next_contract_errors.rs` by removing
the `DynTape::hvp` freed-graph case and keeping only the monomorphic contract
tests.

**Step 3: Run targeted tests**

Run:

```bash
cargo test -p chainrules --test autodiff_next_api_tests --test autodiff_next_contract_errors
```

Expected: PASS.

**Step 4: Re-run the removal search**

Run:

```bash
rg -n "DynTape|DynVariable|DynTangent|DynBackwardOptions|DynHvpOptions|grad_dyn_tangent|grad_dyn_variable" \
  extern/chainrules/src/lib.rs \
  extern/chainrules/tests/autodiff_next_api_tests.rs \
  extern/chainrules/tests/autodiff_next_contract_errors.rs
```

Expected: no matches.

**Step 5: Commit**

```bash
git add extern/chainrules/src/lib.rs \
        extern/chainrules/tests/autodiff_next_api_tests.rs \
        extern/chainrules/tests/autodiff_next_contract_errors.rs
git commit -m "refactor(chainrules): remove dyn tape surface"
```

### Task 3: Delete DynTape-Only Tests and Replace Them with Monomorphic Coverage

**Files:**
- Delete: `extern/chainrules/tests/autodiff_next_dyn_api_tests.rs`
- Delete: `extern/chainrules/tests/autodiff_next_dyn_query_tests.rs`
- Delete: `extern/chainrules/tests/autodiff_next_contract_dyn.rs`
- Modify: `extern/chainrules/tests/autodiff_next_api_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_contract_errors.rs`
- Test: `extern/chainrules/tests`

**Step 1: Confirm which tests are dyn-only**

Run:

```bash
rg -l "DynTape|DynVariable|DynTangent|DynBackwardOptions|DynHvpOptions|grad_dyn_" extern/chainrules/tests
```

Expected: the dyn-only test files plus any mixed API test file still referencing the removed surface.

**Step 2: Delete or rewrite the dyn-only tests**

Delete:

- `extern/chainrules/tests/autodiff_next_dyn_api_tests.rs`
- `extern/chainrules/tests/autodiff_next_dyn_query_tests.rs`
- `extern/chainrules/tests/autodiff_next_contract_dyn.rs`

Strengthen the remaining monomorphic tests instead of keeping empty gaps. For
example, extend `extern/chainrules/tests/autodiff_next_api_tests.rs` with:

```rust
#[test]
fn tape_and_variable_surfaces_cover_remaining_public_entry_points() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0_f64);
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}
```

**Step 3: Run the full `chainrules` test target**

Run:

```bash
cargo test -p chainrules
```

Expected: PASS with no dyn-specific tests remaining.

**Step 4: Commit**

```bash
git add extern/chainrules/tests/autodiff_next_api_tests.rs \
        extern/chainrules/tests/autodiff_next_contract_errors.rs \
        extern/chainrules/tests/chainrules_tests.rs \
        extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs
git rm extern/chainrules/tests/autodiff_next_dyn_api_tests.rs \
       extern/chainrules/tests/autodiff_next_dyn_query_tests.rs \
       extern/chainrules/tests/autodiff_next_contract_dyn.rs
git commit -m "test(chainrules): remove dyn tape coverage"
```

### Task 4: Update Current Documentation to the Homogeneous-Tape Model

**Files:**
- Modify: `extern/chainrules/src/lib.rs`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/design/index.md`
- Modify: `docs/design/einsum-dyadtensor.md`
- Modify: `docs/AD/index.md`
- Modify: `docs/api_index.md`
- Test: `docs/design/autodiff.md`
- Test: `docs/design/index.md`
- Test: `docs/design/einsum-dyadtensor.md`
- Test: `docs/AD/index.md`
- Test: `docs/api_index.md`

**Step 1: Capture the old documentation language**

Run:

```bash
rg -n "DynTape|DynVariable|heterogeneous|mixed custom types|grad_dyn_tangent|grad_dyn_variable|coexistence" \
  extern/chainrules/src/lib.rs \
  docs/design/autodiff.md \
  docs/design/index.md \
  docs/design/einsum-dyadtensor.md \
  docs/AD/index.md \
  docs/api_index.md
```

Expected: matches showing the stale heterogeneous narrative.

**Step 2: Rewrite the docs to the new model**

Apply the approved design:

- `extern/chainrules/src/lib.rs`
  - crate docs describe only `Tape<V>`, `TrackedTensor<V>`, `DualTensor<V>`, and `Variable<V>`
  - examples include `Tape<MyType>` or equivalent generic custom-type usage
  - tensor examples use rank-0 scalar losses for tensor-operation scalar meaning
- `docs/design/autodiff.md`
  - remove the `Tape<V>` + `DynTape` coexistence narrative
  - remove `DynTape` as the higher-order fallback for `V::Tangent != V`
  - state homogeneous-graph-only explicitly
- `docs/design/index.md`
  - update the AD architecture summary to list only `Tape<V>`
- `docs/design/einsum-dyadtensor.md`
  - remove heterogeneous graph references and `DynTape` examples
- `docs/AD/index.md`
  - update the role distinction row to remove `DynTape` coexistence
- `docs/api_index.md`
  - describe `chainrules` as the monomorphic `Tape<V>` AD engine

Use wording like:

```md
`chainrules` provides a single reverse-mode tape surface, `Tape<V>`, for homogeneous graphs.
External users extend AD with `Differentiable` and operation-specific rules on their own value type `V`.
```

For tensor scalar semantics, write:

```md
Tensor-operation scalars follow PyTorch semantics: a scalar tensor is rank-0 (`shape=[]`), not shape `[1]`.
Implicit reverse seed creation remains tied to `num_elements() == 1`.
```

**Step 3: Re-run the doc search**

Run:

```bash
rg -n "DynTape|DynVariable|grad_dyn_tangent|grad_dyn_variable" \
  extern/chainrules/src/lib.rs \
  docs/design/autodiff.md \
  docs/design/index.md \
  docs/design/einsum-dyadtensor.md \
  docs/AD/index.md \
  docs/api_index.md
```

Expected: no matches.

**Step 4: Commit**

```bash
git add extern/chainrules/src/lib.rs \
        docs/design/autodiff.md \
        docs/design/index.md \
        docs/design/einsum-dyadtensor.md \
        docs/AD/index.md \
        docs/api_index.md
git commit -m "docs: update autodiff docs for homogeneous tape model"
```

### Task 5: Run Final Verification Across Formatting, Unit Tests, and Workspace Integration

**Files:**
- Modify: `extern/chainrules/src/lib.rs`
- Modify: `extern/chainrules/tests/chainrules_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_api_tests.rs`
- Modify: `extern/chainrules/tests/autodiff_next_contract_errors.rs`
- Modify: `docs/design/autodiff.md`
- Modify: `docs/design/index.md`
- Modify: `docs/design/einsum-dyadtensor.md`
- Modify: `docs/AD/index.md`
- Modify: `docs/api_index.md`

**Step 1: Format the workspace**

Run:

```bash
cargo fmt --all
```

Expected: succeeds with the changed Rust files formatted.

**Step 2: Run the local crate tests**

Run:

```bash
cargo test -p chainrules
```

Expected: PASS.

**Step 3: Run the workspace regression suite**

Run:

```bash
cargo test --workspace
```

Expected: PASS. This catches downstream crates that still referenced the
removed dyn surface in docs or code.

**Step 4: Run the final static search**

Run:

```bash
rg -n "DynTape|DynVariable|DynTangent|DynBackwardOptions|DynHvpOptions|grad_dyn_tangent|grad_dyn_variable" \
  extern/chainrules \
  docs/design \
  docs/AD \
  docs/api_index.md
```

Expected: no matches outside historical planning records.

**Step 5: Commit**

```bash
git add extern/chainrules/src/lib.rs \
        extern/chainrules/tests/chainrules_tests.rs \
        extern/chainrules/tests/autodiff_next_seed_and_output_contract_tests.rs \
        extern/chainrules/tests/autodiff_next_api_tests.rs \
        extern/chainrules/tests/autodiff_next_contract_errors.rs \
        docs/design/autodiff.md \
        docs/design/index.md \
        docs/design/einsum-dyadtensor.md \
        docs/AD/index.md \
        docs/api_index.md
git commit -m "refactor(chainrules): unify on homogeneous tape model"
```
