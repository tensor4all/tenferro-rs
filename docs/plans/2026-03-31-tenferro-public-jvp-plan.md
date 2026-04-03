# Tenferro Public JVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a public `tenferro::jvp(...)` transform for `Tensor` without exposing dual-carrier internals, while preserving the `linearize-first` internal seam.

**Architecture:** Keep `Tensor` as a facade over the upstream `Value<DynTensor>` carrier. Public forward AD is a functional transform layered over existing `LinearizableOp` / `LinearizedOp` machinery. The public API exposes `JvpResult` and `jvp(...)`; the implementation remains free to delegate to `frule` for simple rules or use optimized `LinearizedOp::jvp` paths where residual reuse matters.

**Tech Stack:** Rust 2021, `tenferro` public crate, `tenferro-internal-ad-surface`, `tidu-rs` linearize-first API, `cargo fmt`, `cargo test --release`, rustdoc

---

### Task 1: Lock the public contract with failing tests

**Files:**
- Modify: `tenferro/tests/integration.rs`
- Create: `tenferro/tests/integration/public_jvp.rs`

**Step 1: Write the failing public API tests**

Add tests for:

- unary JVP over `exp().sum()`
- binary JVP over `add().sum()`
- `None` tangent input
- multi-output JVP over `qr()`
- runtime-missing error for `qr()` without installed runtime

The tests should call a free `tenferro::jvp(...)` function and assert on:

- output count
- tangent count
- tangent presence/absence
- basic numeric correctness

**Step 2: Run the tests to confirm they fail**

Run:

```bash
cargo test -p tenferro --test integration --release public_jvp
```

Expected: FAIL because `tenferro::jvp` and `JvpResult` do not exist yet.

**Step 3: Commit the failing contract**

```bash
git add tenferro/tests/integration.rs \
        tenferro/tests/integration/public_jvp.rs
git commit -m "test: lock tenferro public jvp contract"
```

### Task 2: Add the public `JvpResult` and `jvp(...)` surface

**Files:**
- Modify: `tenferro/src/lib.rs`
- Create: `tenferro/src/jvp.rs`

**Step 1: Add the new public result type**

Create `JvpResult` with:

- `outputs: Vec<Tensor>`
- `output_tangents: Vec<Option<Tensor>>`

Keep the type minimal. Do not add HVP or higher-order options.

**Step 2: Add the new public function**

Add:

```rust
pub fn jvp<F>(
    f: F,
    primals: &[Tensor],
    tangents: &[Option<Tensor>],
) -> Result<JvpResult>
where
    F: FnOnce(&[Tensor]) -> Result<Vec<Tensor>>;
```

**Step 3: Re-export the new surface from `lib.rs`**

Public users should discover `jvp` and `JvpResult` directly from the crate
root.

**Step 4: Run the public API tests**

Run:

```bash
cargo test -p tenferro --test integration --release public_jvp
```

Expected: compile now succeeds, but tests still fail at runtime until the
implementation exists.

**Step 5: Commit**

```bash
git add tenferro/src/lib.rs \
        tenferro/src/jvp.rs
git commit -m "feat: add tenferro public jvp surface"
```

### Task 3: Implement the transform in `tenferro-internal-ad-surface`

**Files:**
- Modify: `internal/tenferro-internal-ad-surface/src/lib.rs`
- Create: `internal/tenferro-internal-ad-surface/src/jvp.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`

**Step 1: Add internal validation helpers**

Validate:

- `primals.len() == tangents.len()`
- each `Some(tangent)` matches the corresponding primal in dtype, shape, and
  layout

Return ordinary `tenferro` errors through the existing error translation layer.

**Step 2: Add the internal `jvp(...)` executor**

Implement the internal transform executor that:

- receives public `Tensor` primals and optional tangents
- runs `f`
- obtains primal outputs and output tangents through the `linearize-first`
  internal seam
- returns detached tangent outputs

Do not expose public dual-builder concepts here.

**Step 3: Keep `Tensor` as a facade**

Do not add forward state or dual state to `Tensor`. Any helpers added to
`tensor.rs` must preserve the existing facade-only architecture.

**Step 4: Run focused internal tests if needed**

Add or update crate-local tests so the internal executor can be verified without
going through the public integration crate for every case.

**Step 5: Run verification**

Run:

```bash
cargo test -p tenferro-internal-ad-surface --release
cargo test -p tenferro --test integration --release public_jvp
```

Expected: PASS for the public JVP tests.

**Step 6: Commit**

```bash
git add internal/tenferro-internal-ad-surface/src/lib.rs \
        internal/tenferro-internal-ad-surface/src/jvp.rs \
        internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs
git commit -m "feat: implement tenferro public jvp transform"
```

### Task 4: Add optimized seam coverage where JVP is not thin delegation

**Files:**
- Modify: `internal/tenferro-internal-ad-ops/tests/linearized_ops.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

**Step 1: Identify non-thin `LinearizedOp::jvp` implementations**

Focus on ops whose `jvp` path is not just a trivial wrapper over an existing
`frule`, especially where the implementation depends on:

- cached outputs
- saved factorization state
- multi-output packaging

Candidates likely include:

- `exp`
- `qr`
- `svd`

**Step 2: Add focused seam tests**

For each such op, add tests that compare the runtime `LinearizedOp::jvp`
behavior against expected semantics while explicitly exercising:

- saved linearization state
- optional tangent handling
- output ordering for multi-output ops

**Step 3: Run the focused seam tests**

Run:

```bash
cargo test -p tenferro-internal-ad-ops --release linearized_ops
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops
```

Expected: PASS.

**Step 4: Commit**

```bash
git add internal/tenferro-internal-ad-ops/tests/linearized_ops.rs \
        internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs
git commit -m "test: cover optimized jvp seam behavior"
```

### Task 5: Update docs and examples to match the real public surface

**Files:**
- Modify: `tenferro/README.upstream.md`
- Modify: `tenferro/src/lib.rs`
- Modify: any rustdoc examples that discuss AD or public transforms

**Step 1: Document the new public transform**

Update README and rustdoc to show:

- `tenferro::jvp(...)`
- the `JvpResult` shape
- that reverse-mode remains available separately

**Step 2: Remove or avoid misleading claims**

Confirm docs do not imply:

- public dual builders
- public HVP/FoR
- more forward-mode coverage than actually exists

This is required by `REPOSITORY_RULES.md`.

**Step 3: Run rustdoc tests**

Run:

```bash
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro/README.upstream.md \
        tenferro/src/lib.rs
git commit -m "docs: add tenferro public jvp examples"
```

### Task 6: Final verification

**Files:**
- No new files

**Step 1: Format**

Run:

```bash
cargo fmt --all
```

**Step 2: Run the relevant crate tests**

Run:

```bash
cargo test -p tenferro-internal-ad-surface --release
cargo test -p tenferro-internal-ad-ops --release
cargo test -p tenferro-internal-ad-linalg --release
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
cargo check -p tenferro --tests --release
```

Expected: PASS.

**Step 3: Review docs and repository rules**

Before considering the work complete, verify:

- README / rustdoc / examples do not exceed the current public surface
- non-thin `LinearizedOp::jvp/vjp` implementations have seam coverage
- no DRY/KISS/layering-breaking ad hoc workaround was introduced

**Step 4: Commit final cleanup if needed**

```bash
git add -A
git commit -m "chore: finish tenferro public jvp rollout"
```
