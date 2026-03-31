# Tenferro Complex Linalg AD Rollout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Roll out complex AD support for the remaining linalg operations in `tenferro::Tensor`, using oracle replay as the first gate and enabling public Tensor seams in three batches.

**Architecture:** Use an oracle-first rollout. Each batch closes four layers in order: `tenferro-linalg` stateless rules, oracle replay, internal `LinearizedOp` seam, and finally the public `Tensor` seam. Batch A covers same-domain complex outputs, Batch B covers scalar or mixed structured outputs, and Batch C isolates `eig`.

**Tech Stack:** Rust 2021, `tenferro-linalg`, `tenferro-internal-ad-linalg`, `tenferro-internal-ad-surface`, vendored `tensor-ad-oracles`, `cargo fmt`, `cargo clippy`, `cargo test --release`

---

### Task 1: Lock the oracle-first rollout invariants in docs

**Files:**
- Create: `docs/plans/2026-03-31-tenferro-complex-linalg-ad-design.md`
- Create: `docs/plans/2026-03-31-tenferro-complex-linalg-ad-plan.md`

**Step 1: Save the approved design**

Write the approved batching/layering design into the new design document.

**Step 2: Save this implementation plan**

Write the implementation plan with exact batch boundaries and verification
gates.

**Step 3: Commit the docs**

```bash
git add docs/plans/2026-03-31-tenferro-complex-linalg-ad-design.md \
        docs/plans/2026-03-31-tenferro-complex-linalg-ad-plan.md
git commit -m "docs: add complex linalg ad rollout design"
```

### Task 2: Add Batch A oracle replay coverage tests first

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/decode.rs`
- Test: vendored oracle cases under `third_party/tensor-ad-oracles/cases/{inv,cholesky,pinv,matrix_exp}/`

**Step 1: Extend support classification for Batch A complex cases**

Update record support classification so the complex oracle cases for:

- `inv`
- `cholesky`
- `pinv`
- `matrix_exp`

are treated as supported replay targets.

**Step 2: Add failing replay coverage**

Add or tighten tests so replay is required to process the new complex cases.

**Step 3: Run replay tests and confirm failure**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db
```

Expected: FAIL before the complex `_frule` / `_rrule` work is complete.

**Step 4: Commit the failing replay gate**

```bash
git add tenferro-linalg/tests/oracle_db/support.rs \
        tenferro-linalg/tests/oracle_db/replay.rs \
        tenferro-linalg/tests/oracle_db/decode.rs
git commit -m "test: require batch a complex oracle replay"
```

### Task 3: Implement Batch A complex stateless rules

**Files:**
- Modify: `tenferro-linalg/src/frules/linear_systems.rs`
- Modify: `tenferro-linalg/src/rrules/linear_systems.rs`
- Modify: `tenferro-linalg/src/frules/least_squares.rs`
- Modify: `tenferro-linalg/src/rrules/least_squares.rs`
- Modify: `tenferro-linalg/src/frules/spectral.rs`
- Modify: `tenferro-linalg/src/rrules/spectral.rs`
- Modify: `tenferro-linalg/src/frules/matrix_functions.rs`
- Modify: `tenferro-linalg/src/rrules/matrix_functions.rs`

**Step 1: Implement complex `inv_frule` / `inv_rrule`**

Use the oracle-backed complex formulas and keep the implementation
solve-style rather than introducing ad hoc explicit inverse logic.

**Step 2: Implement complex `cholesky_frule` / `cholesky_rrule`**

Preserve the Hermitian triangular-solve structure already used in the real
path.

**Step 3: Implement complex `pinv_frule` / `pinv_rrule`**

Keep threshold semantics aligned with the existing primal pseudoinverse.

**Step 4: Implement complex `matrix_exp_frule` / `matrix_exp_rrule`**

Use the oracle-supported block-exponential / differential path rather than
introducing a separate ad hoc formulation.

**Step 5: Run Batch A replay**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db
```

Expected: Batch A complex replay records PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg/src/frules/linear_systems.rs \
        tenferro-linalg/src/rrules/linear_systems.rs \
        tenferro-linalg/src/frules/least_squares.rs \
        tenferro-linalg/src/rrules/least_squares.rs \
        tenferro-linalg/src/frules/spectral.rs \
        tenferro-linalg/src/rrules/spectral.rs \
        tenferro-linalg/src/frules/matrix_functions.rs \
        tenferro-linalg/src/rrules/matrix_functions.rs
git commit -m "feat: add batch a complex linalg ad rules"
```

### Task 4: Lift Batch A through the internal linearized seam

**Files:**
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

**Step 1: Remove Batch A real-only gates**

Open complex dtype handling for:

- `inv`
- `cholesky`
- `pinv`
- `matrix_exp`

inside the internal linearized seam.

**Step 2: Add focused seam tests where delegation is not thin**

If the implementation is not just a straight `frule` / `rrule` call, add
focused seam tests per `REPOSITORY_RULES.md`.

**Step 3: Run internal linalg seam tests**

Run:

```bash
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops
```

Expected: PASS.

**Step 4: Commit**

```bash
git add internal/tenferro-internal-ad-linalg/src/linearized.rs \
        internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs
git commit -m "feat: lift batch a complex linalg through linearized seam"
```

### Task 5: Expose Batch A at the public Tensor seam

**Files:**
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/tests/integration/autograd_surface_tests.rs`
- Modify: `tenferro/README.md`

**Step 1: Remove public Batch A real-only rejections**

Enable complex `Tensor` usage for:

- `inv`
- `cholesky`
- `pinv`
- `matrix_exp`

for both reverse AD and public `jvp(...)`.

**Step 2: Add public complex integration tests**

Cover both:

- `Tensor` primal + reverse path
- `Tensor + jvp(...)`

for the Batch A ops.

**Step 3: Update the support table**

Update `tenferro/README.md` so Batch A is described as public complex AD,
not only primal complex support.

**Step 4: Run verification**

Run:

```bash
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 5: Commit**

```bash
git add internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs \
        tenferro/tests/integration/linalg_surface_tests.rs \
        tenferro/tests/integration/autograd_surface_tests.rs \
        tenferro/README.md
git commit -m "feat: expose batch a complex tensor ad support"
```

### Task 6: Add Batch B oracle replay coverage tests first

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/decode.rs`
- Test: vendored oracle cases under `third_party/tensor-ad-oracles/cases/{det,slogdet,norm,lstsq_grad_oriented,eigh}/`

**Step 1: Extend support classification for Batch B**

Mark the Batch B complex families as supported replay targets.

**Step 2: Add failing replay coverage**

Require replay to process Batch B complex records before implementation.

**Step 3: Run replay and confirm failure**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db
```

Expected: FAIL before Batch B stateless rules are updated.

**Step 4: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/support.rs \
        tenferro-linalg/tests/oracle_db/replay.rs \
        tenferro-linalg/tests/oracle_db/decode.rs
git commit -m "test: require batch b complex oracle replay"
```

### Task 7: Implement Batch B complex stateless rules

**Files:**
- Modify: `tenferro-linalg/src/frules/linear_systems.rs`
- Modify: `tenferro-linalg/src/rrules/linear_systems.rs`
- Modify: `tenferro-linalg/src/frules/norms.rs`
- Modify: `tenferro-linalg/src/rrules/norms.rs`
- Modify: `tenferro-linalg/src/frules/least_squares.rs`
- Modify: `tenferro-linalg/src/rrules/least_squares.rs`
- Modify: `tenferro-linalg/src/frules/lu_eigen.rs`
- Modify: `tenferro-linalg/src/rrules/lu_eigen.rs`

**Step 1: Implement complex `det` / `slogdet` rules**

Preserve the mixed output semantics already pinned by oracle notes.

**Step 2: Implement complex `norm` rules**

Keep the real-output / complex-gradient bridge explicit and tested.

**Step 3: Implement complex `lstsq` rules**

Keep result-struct packaging aligned with the oracle family
`lstsq_grad_oriented`.

**Step 4: Implement complex `eigen` rules**

Follow the Hermitian complex strategy used by the oracle families.

**Step 5: Run Batch B replay**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db
```

Expected: Batch B complex replay records PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg/src/frules/linear_systems.rs \
        tenferro-linalg/src/rrules/linear_systems.rs \
        tenferro-linalg/src/frules/norms.rs \
        tenferro-linalg/src/rrules/norms.rs \
        tenferro-linalg/src/frules/least_squares.rs \
        tenferro-linalg/src/rrules/least_squares.rs \
        tenferro-linalg/src/frules/lu_eigen.rs \
        tenferro-linalg/src/rrules/lu_eigen.rs
git commit -m "feat: add batch b complex linalg ad rules"
```

### Task 8: Lift Batch B through the internal seam and public Tensor seam

**Files:**
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/tests/integration/autograd_surface_tests.rs`
- Modify: `tenferro/README.md`

**Step 1: Open Batch B complex gates internally**

Enable Batch B complex JVP/VJP in the linearized seam.

**Step 2: Add focused seam tests**

Add seam tests for mixed output / mixed cotangent packaging where delegation is
not trivially thin.

**Step 3: Open Batch B at the public Tensor seam**

Enable public complex support for:

- `det`
- `slogdet`
- `norm`
- `lstsq`
- `eigen`

**Step 4: Update docs**

Update the public support table so Batch B is accurately described.

**Step 5: Run verification**

Run:

```bash
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add internal/tenferro-internal-ad-linalg/src/linearized.rs \
        internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs \
        internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs \
        tenferro/tests/integration/linalg_surface_tests.rs \
        tenferro/tests/integration/autograd_surface_tests.rs \
        tenferro/README.md
git commit -m "feat: expose batch b complex tensor ad support"
```

### Task 9: Design-review and implement Batch C (`eig`)

**Files:**
- Modify: `tenferro-linalg/src/primal/spectral.rs`
- Modify: `tenferro-linalg/src/frules/spectral.rs`
- Modify: `tenferro-linalg/src/rrules/spectral.rs`
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `internal/tenferro-internal-ad-linalg/src/linearized.rs`
- Modify: `internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs`
- Modify: `tenferro/tests/integration/linalg_surface_tests.rs`
- Modify: `tenferro/README.md`

**Step 1: Re-check the oracle strategy for `eig`**

Before implementation, confirm the current oracle-side strategy for:

- complex outputs
- gauge handling
- replay expectations

Do not assume `eig` can be implemented like Batch A or Batch B.

**Step 2: Add failing replay coverage for the chosen `eig` strategy**

Require replay to validate the intended `eig` path.

**Step 3: Implement only the supported `eig` contract**

Match the oracle contract exactly. Do not widen beyond the oracle strategy.

**Step 4: Add seam and public tests**

Cover:

- the supported `eig` observable contract
- gauge-sensitive expected-error behavior where relevant

**Step 5: Update docs**

Describe exactly what `eig` supports and what remains intentionally deferred.

**Step 6: Run verification**

Run:

```bash
cargo test -p tenferro-linalg --release oracle_db
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro-linalg/src/primal/spectral.rs \
        tenferro-linalg/src/frules/spectral.rs \
        tenferro-linalg/src/rrules/spectral.rs \
        tenferro-linalg/tests/oracle_db/support.rs \
        tenferro-linalg/tests/oracle_db/replay.rs \
        internal/tenferro-internal-ad-linalg/src/linearized.rs \
        internal/tenferro-internal-ad-surface/src/core/dynamic/tensor.rs \
        tenferro/tests/integration/linalg_surface_tests.rs \
        tenferro/README.md
git commit -m "feat: expose supported complex eig ad path"
```

### Task 10: Final verification and cleanup

**Files:**
- Modify if needed: `tenferro/README.md`
- Modify if needed: `README.md`

**Step 1: Run formatting**

```bash
cargo fmt --all
```

**Step 2: Run lint**

```bash
cargo clippy -p tenferro --tests --release -- -D warnings
```

**Step 3: Run focused tests**

```bash
cargo test -p tenferro-linalg --release oracle_db
cargo test -p tenferro-internal-ad-linalg --release dyn_linalg_ops
cargo test -p tenferro --test integration --release
cargo test -p tenferro --doc --release
```

**Step 4: Run final support-table review**

Manually confirm README / rustdoc / examples do not claim support beyond the
current public surface.

**Step 5: Commit final cleanup**

```bash
git add README.md tenferro/README.md
git commit -m "docs: finalize complex linalg ad support matrix"
```
