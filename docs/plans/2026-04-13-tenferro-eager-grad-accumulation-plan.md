# Tenferro Eager Gradient Accumulation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Change `tenferro::EagerTensor` to PyTorch-style gradient accumulation semantics, add explicit eager gradient reset APIs, and update public docs so eager reverse-mode AD is documented honestly.

**Architecture:** Keep the current `Arc<EagerContext<B>>` + `Mutex<B>` runtime model. The root change is inside eager gradient-slot storage: `backward()` must stop clearing or overwriting slots and instead merge fresh cotangents into persistent per-leaf storage. Public docs should then position eager AD as scalar-loss reverse mode, while traced AD remains the transform-oriented path for `grad`, `vjp`, `jvp`, and `hvp`.

**Tech Stack:** Rust 2021, `tenferro` facade crate, `tenferro-einsum`, `tenferro-tensor`, rustdoc doctests, mdBook/quarto docs, `cargo fmt`, `cargo test --release`, `cargo nextest`, `cargo llvm-cov`

---

### Task 1: Lock the eager accumulation contract with failing tests

**Files:**
- Modify: `tenferro/tests/eager_tensor.rs`
- Modify: `tenferro/tests/eager_einsum_ad.rs`

**Step 1: Add failing repeated-backward accumulation tests**

Add a leaf test like:

```rust
#[test]
fn eager_repeated_backward_accumulates_across_calls() {
    let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]));

    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 8.0, 12.0]);
}
```

Also add:

- `eager_clear_grad_resets_only_one_leaf`
- `eager_context_clear_grads_resets_all_live_leaves`
- `eager_unrelated_backward_keeps_existing_leaf_grad`
- `eager_tracks_grad_reports_leaf_state`
- a compile-time `assert_send_sync::<EagerTensor<CpuBackend>>()`
- a compile-time `assert_send_sync::<EagerContext<CpuBackend>>()`

**Step 2: Add failing eager einsum accumulation tests**

Add a test like:

```rust
#[test]
fn eager_einsum_ad_backward_accumulates_across_calls() {
    let a = EagerTensor::requires_grad(Tensor::new(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = EagerTensor::requires_grad(Tensor::new(
        vec![3, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let c = eager_einsum_ad(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    let grad_a_once = a.grad().unwrap();

    let c = eager_einsum_ad(&[&a, &b], "ij,jk->ik").unwrap();
    let loss = c.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    let grad_a_twice = a.grad().unwrap();
    assert_eq!(
        grad_a_twice.as_slice::<f64>().unwrap(),
        &grad_a_once
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|x| x * 2.0)
            .collect::<Vec<_>>()
    );
}
```

Also add a context-level clear test that uses `EagerContext::with_backend(...)`
plus `requires_grad_in(...)`.

**Step 3: Run the focused tests to confirm they fail**

Run:

```bash
cargo test -p tenferro --test eager_tensor --release
cargo test -p tenferro --test eager_einsum_ad --release
```

Expected:

- compile errors for missing `clear_grad`, `clear_grads`, or `tracks_grad`
- assertion failures showing repeated `backward()` still overwrites instead of accumulating

**Step 4: Commit the failing contract**

```bash
git add tenferro/tests/eager_tensor.rs \
        tenferro/tests/eager_einsum_ad.rs
git commit -m "test: lock eager grad accumulation contract"
```

### Task 2: Implement accumulation-aware gradient storage in `EagerContext`

**Files:**
- Modify: `tenferro/src/eager.rs`

**Step 1: Replace overwrite storage with accumulation storage**

Delete the old overwrite helper shape:

```rust
fn store_grads(&self, cotangents: &HashMap<...>) { ... }
```

Replace it with an accumulation-aware helper that:

- retains only live grad slots
- ignores tracked keys absent from `cotangents`
- writes `Some(incoming.clone())` into empty slots
- adds `incoming` into the existing gradient when the slot is already `Some(...)`

Use backend-backed tensor addition rather than hard-coded host-side dtype
special cases.

**Step 2: Keep the backend lock through accumulation**

Change `backward()` from:

```rust
self.ctx.clear_grads();
...
let cotangents = backward_dag(...);
self.ctx.store_grads(&cotangents);
```

to:

```rust
let cotangents = backward_dag(...);
self.ctx.accumulate_grads(&cotangents, &mut *backend)?;
```

Do not clear gradients inside `backward()`.

**Step 3: Preserve the scalar-output guard and return value**

Keep:

- the existing non-scalar error path
- the returned fresh `cotangents` map from the current reverse pass

Do not change `backward()` into a pure side-effect API.

**Step 4: Run the core eager tensor tests**

Run:

```bash
cargo test -p tenferro --test eager_tensor --release
```

Expected: the accumulation semantics tests now pass.

**Step 5: Commit**

```bash
git add tenferro/src/eager.rs
git commit -m "feat: accumulate eager gradients across backward calls"
```

### Task 3: Add explicit reset and tracking APIs

**Files:**
- Modify: `tenferro/src/eager.rs`

**Step 1: Expose `EagerContext::clear_grads()` publicly**

Promote `clear_grads` to a documented public method:

```rust
impl<B: TensorBackend> EagerContext<B> {
    pub fn clear_grads(&self) { ... }
}
```

It should:

- clear every live registered slot in the context
- retain only live weak references
- never panic on empty slots

**Step 2: Add `EagerTensor::clear_grad()`**

Add:

```rust
pub fn clear_grad(&self) {
    *self.grad_slot.lock().unwrap() = None;
}
```

This should be a no-op for untracked tensors and detached tensors with empty
slots.

**Step 3: Add `EagerTensor::tracks_grad()`**

Add:

```rust
pub fn tracks_grad(&self) -> bool {
    self.requires_grad
}
```

Do not rename the existing constructors. The getter exists because a method
named `requires_grad()` would collide with the constructor family.

**Step 4: Update eager rustdoc examples in the same file**

Revise `tenferro/src/eager.rs` docs so examples show:

- `backward()` accumulation across two calls
- `clear_grad()` on a single tensor
- `clear_grads()` on a shared context
- `tracks_grad()` returning `true` only for tracked leaves

**Step 5: Run the full eager tests again**

Run:

```bash
cargo test -p tenferro --test eager_tensor --release
cargo test -p tenferro --test eager_einsum_ad --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/eager.rs
git commit -m "feat: add eager grad reset and tracking apis"
```

### Task 4: Clean up the eager facade surface for user-facing docs

**Files:**
- Modify: `tenferro/src/eager_einsum.rs`

**Step 1: Re-export eager primal einsum from the facade module**

Add:

```rust
pub use tenferro_einsum::eager_einsum;
```

inside `tenferro::eager_einsum`, alongside `eager_einsum_ad`.

This lets user-facing docs stay on `use tenferro::...` imports, which is
required by `REPOSITORY_RULES.md`.

**Step 2: Update the module-level rustdoc example**

Show both eager execution and eager reverse-mode in `tenferro/src/eager_einsum.rs`.
Keep examples small and runnable.

**Step 3: Run focused docs/tests**

Run:

```bash
cargo test -p tenferro --test eager_einsum_ad --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro/src/eager_einsum.rs
git commit -m "feat: expose eager einsum from the tenferro facade"
```

### Task 5: Update README, guides, and mapping docs

**Files:**
- Modify: `README.md`
- Modify: `tenferro/README.md`
- Modify: `docs/index.md`
- Modify: `docs/guides/eager-operations.md`
- Modify: `docs/guides/autodiff.md`
- Modify: `docs/getting-started/pytorch-jax-mapping.md`

**Step 1: Fix stale capability claims**

Update all public wording that still says eager mode is "without automatic
differentiation" or that "need gradients" always means traced mode.

The new public positioning should be:

- eager mode supports immediate execution plus scalar-loss reverse-mode via
  `backward()`
- traced mode remains the public transform-oriented AD surface for `grad`,
  `vjp`, `jvp`, and `hvp`

**Step 2: Add one canonical eager accumulation example**

In `docs/guides/eager-operations.md`, include a compact example like:

```rust
use tenferro::{EagerTensor, Tensor};

let x = EagerTensor::requires_grad(Tensor::new(vec![2], vec![1.0_f64, 2.0]));
let loss = (&x * &x).reduce_sum(&[0]).unwrap();
let _ = loss.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);

let loss = (&x * &x).reduce_sum(&[0]).unwrap();
let _ = loss.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[4.0, 8.0]);

x.clear_grad();
assert!(x.grad().is_none());
```

Also update the "When to use eager vs lazy" table so it distinguishes:

- eager `backward()` for scalar-loss reverse mode
- traced `grad` / `vjp` / `jvp` / `hvp` for transform-oriented AD

**Step 3: Fix user-facing imports**

Replace internal-crate imports such as:

```rust
use tenferro_tensor::{...}
use tenferro_einsum::eager_einsum;
```

with facade imports such as:

```rust
use tenferro::{CpuBackend, Tensor, TensorBackend, TypedTensor};
use tenferro::eager_einsum::{eager_einsum, eager_einsum_ad};
```

Do not leave any user-facing doc page importing internal crates.

**Step 4: Update the PyTorch/JAX mapping table**

Make the mapping explicit:

- PyTorch `loss.backward()` -> eager `loss.backward()` with accumulation
- PyTorch `torch.autograd.grad(...)` -> traced `loss.grad(&x)`
- JAX `jax.grad`, `jax.vjp`, `jax.jvp` -> traced transform APIs

**Step 5: Run docs verification**

Run:

```bash
cargo test -p tenferro --doc --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 6: Commit**

```bash
git add README.md \
        tenferro/README.md \
        docs/index.md \
        docs/guides/eager-operations.md \
        docs/guides/autodiff.md \
        docs/getting-started/pytorch-jax-mapping.md
git commit -m "docs: describe eager gradient accumulation semantics"
```

### Task 6: Final verification and PR-readiness checks

**Files:**
- No intended source changes; only fix fallout if a verification step exposes one

**Step 1: Format the workspace**

Run:

```bash
cargo fmt --all
cargo fmt --all --check
```

Expected: PASS.

**Step 2: Run targeted eager verification one more time**

Run:

```bash
cargo test -p tenferro --test eager_tensor --release
cargo test -p tenferro --test eager_einsum_ad --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 3: Run the repository-wide verification gates**

Run:

```bash
cargo nextest run --workspace --release --no-fail-fast
cargo test --doc --workspace --release
cargo llvm-cov nextest --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 4: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected:

- only intended eager API / test / docs files are modified
- no unrelated files are touched

**Step 5: If verification fallout requires fixes, make one final focused commit**

```bash
git add -A
git commit -m "fix: address eager grad accumulation verification fallout"
```

Only do this if a verification failure required code or docs changes.
