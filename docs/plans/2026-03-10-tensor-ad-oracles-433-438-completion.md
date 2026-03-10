# Tensor AD Oracles Issues 433-438 Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete issues #433 through #438 in one integrated PR with tenferro-native public APIs, CPU implementations, required docs/tests, dyadtensor support, and oracle replay promotion.

**Architecture:** Land the work in four layers: public surface, CPU implementation, dyadtensor/AD integration, and oracle replay promotion. Internally sequence the work as Batch A (`*_ex`, LU factorization/solve, `cond`, `matrix_power`), Batch B (`cross`, `householder_product`, `vander`, `tensorinv`, `tensorsolve`), then Batch C (dyadtensor and replay support for everything new).

**Tech Stack:** Rust workspace crates (`tenferro-linalg`, `tenferro-tensor`, `tenferro-einsum`, `extension/tenferro-dyadtensor`), oracle replay integration tests, rustdoc, GitHub issue-driven scope.

---

### Task 1: Freeze the integrated feature worktree and scope

**Files:**
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-433-438-completion-design.md`

**Step 1: Verify the feature branch starts from current `origin/main`**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/main
```

Expected: both SHAs match at the start of implementation.

**Step 2: Re-read the issue scope and design doc**

Check:

- `docs/plans/2026-03-10-tensor-ad-oracles-433-438-completion-design.md`
- GitHub issues `#433` through `#438`

Expected: the exact family inventory and AD policy are fixed before code edits.

**Step 3: Commit any design-only clarification**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-433-438-completion-design.md
git commit -m "docs: finalize tensor-ad-oracles completion design"
```

### Task 2: Add failing contract tests for structured `*_ex` and LU factor APIs

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Test: `tenferro-linalg/src/tests/` or existing module-local test files for linalg public API

**Step 1: Write failing primal tests for new public result types and functions**

Cover:

- `cholesky_ex`
- `inv_ex`
- `solve_ex`
- `lu_factor`
- `lu_factor_ex`
- `lu_solve`

Include assertions for:

- result field shapes
- `info` semantics
- `Err` on contract violations

**Step 2: Run only the new failing tests**

Run:

```bash
cargo test -p tenferro-linalg ex
cargo test -p tenferro-linalg lu_factor
```

Expected: FAIL because the public APIs do not exist yet.

**Step 3: Add minimal public result structs and signatures**

Implement in `tenferro-linalg/src/lib.rs`:

- new result structs
- new public functions
- rustdoc examples for each new public item

Do not implement full CPU behavior yet beyond enough stubs to compile.

**Step 4: Re-run the targeted tests**

Expected: compile or runtime failures now isolate CPU implementation gaps rather
than missing symbols.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs
git commit -m "feat: add structured ex and LU factor public contracts"
```

### Task 3: Implement CPU behavior for `*_ex`, LU factorization, and LU solve

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg/src/backend/cpu.rs`
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Test: linalg module-local tests for these operations

**Step 1: Write failing execution tests**

Add tests for:

- successful `cholesky_ex`, `inv_ex`, `solve_ex`
- singular or numerically failing cases that set `info`
- packed LU output from `lu_factor`
- structured status from `lu_factor_ex`
- `lu_solve` against direct `solve`

**Step 2: Run the targeted execution tests**

Run:

```bash
cargo test -p tenferro-linalg cholesky_ex
cargo test -p tenferro-linalg inv_ex
cargo test -p tenferro-linalg solve_ex
cargo test -p tenferro-linalg lu_factor
cargo test -p tenferro-linalg lu_solve
```

Expected: FAIL with missing backend implementation details.

**Step 3: Implement CPU behavior**

Use existing CPU helpers where possible:

- reuse `lu_factor` backend internals for packed factorization
- define clear conversion from numerical status to `info`
- keep `Err` for contract violations only

**Step 4: Re-run the targeted tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend/tensor_api.rs \
  tenferro-linalg/src/backend/cpu.rs \
  tenferro-linalg/src/backend/cpu_tensor_impl.rs \
  tenferro-linalg/src/lib.rs
git commit -m "feat: implement structured ex and LU factor CPU paths"
```

### Task 4: Add failing primal tests for `cond` and `matrix_power`

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Test: linalg module-local tests for scalar-output utilities

**Step 1: Add failing tests**

Cover:

- `cond` for supported norm choices and simple diagonal matrices
- `matrix_power` for exponent `0`, positive exponents, and negative powers on
  invertible matrices

**Step 2: Run the targeted tests**

Run:

```bash
cargo test -p tenferro-linalg cond
cargo test -p tenferro-linalg matrix_power
```

Expected: FAIL because the APIs do not exist yet.

**Step 3: Implement minimal public APIs and CPU behavior**

Prefer composition through existing public operations:

- `cond` via norms / inverse / singular values as appropriate
- `matrix_power` via repeated squaring and inverse where needed

Add rustdoc examples.

**Step 4: Re-run the targeted tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs
git commit -m "feat: add cond and matrix_power APIs"
```

### Task 5: Add failing primal tests for tensor construction and tensorized solve

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Test: linalg module-local tests covering:
  - `cross`
  - `householder_product`
  - `vander`
  - `tensorinv`
  - `tensorsolve`

**Step 1: Write failing tests for each public contract**

Cover:

- shape validation
- axis semantics
- small deterministic numeric examples

**Step 2: Run the targeted tests**

Run:

```bash
cargo test -p tenferro-linalg cross
cargo test -p tenferro-linalg householder_product
cargo test -p tenferro-linalg vander
cargo test -p tenferro-linalg tensorinv
cargo test -p tenferro-linalg tensorsolve
```

Expected: FAIL because the APIs are missing.

**Step 3: Implement the public APIs and CPU paths**

Use:

- explicit shape and axis validation in `tenferro-linalg`
- `tenferro-einsum` where it simplifies tensorized contractions
- matricization / reshape composition for `tensorinv` and `tensorsolve`

Add rustdoc examples.

**Step 4: Re-run the targeted tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs
git commit -m "feat: add tensor construction and tensorized solve APIs"
```

### Task 6: Add dyadtensor primal surface for all new operations

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Test: `extension/tenferro-dyadtensor/src/api/ad/tests/`

**Step 1: Add failing primal-surface tests**

Cover builder or eager entrypoints for:

- `*_ex`
- `lu_factor`, `lu_factor_ex`, `lu_solve`
- `cond`
- `matrix_power`
- `cross`, `householder_product`, `vander`
- `tensorinv`, `tensorsolve`

**Step 2: Run the targeted tests**

Run:

```bash
cargo test -p tenferro-dyadtensor ex
cargo test -p tenferro-dyadtensor lu_factor
cargo test -p tenferro-dyadtensor tensorinv
```

Expected: FAIL because the surface is missing.

**Step 3: Add the primal dyadtensor surface**

Expose builder/eager wrappers for all new public APIs.

Keep primal-only ops primal-only:

- `cholesky_ex`
- `inv_ex`
- `solve_ex`
- `lu_factor`
- `lu_factor_ex`

**Step 4: Re-run the targeted tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/mod.rs \
  extension/tenferro-dyadtensor/src/api/ad.rs \
  extension/tenferro-dyadtensor/src/api/ad/tests
git commit -m "feat: expose new oracle-completion ops in dyadtensor"
```

### Task 7: Add AD support where the contract is meaningful

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Test: `extension/tenferro-dyadtensor/src/api/ad/tests/`

**Step 1: Add failing AD tests**

Cover:

- `cond`
- `matrix_power`
- `tensorinv`
- `tensorsolve`
- `cross`
- `householder_product`
- `vander`
- `lu_solve`

Prefer rule reuse via composition where possible.

**Step 2: Run the targeted tests**

Run:

```bash
cargo test -p tenferro-dyadtensor cond
cargo test -p tenferro-dyadtensor matrix_power
cargo test -p tenferro-dyadtensor tensorinv
cargo test -p tenferro-dyadtensor lu_solve
```

Expected: FAIL because the AD wiring is incomplete.

**Step 3: Implement the minimal AD path**

Order:

1. composition from existing differentiable ops
2. explicit frule/rrule only where composition is inadequate

Document any intentionally unsupported tangent paths.

**Step 4: Re-run the targeted tests**

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs \
  extension/tenferro-dyadtensor/src/api/mod.rs \
  extension/tenferro-dyadtensor/src/api/ad.rs \
  extension/tenferro-dyadtensor/src/api/ad/tests
git commit -m "feat: add AD support for oracle-completion ops"
```

### Task 8: Promote oracle replay support for all newly covered families

**Files:**
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/hvp.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`
- Update if needed: `docs/generated/tensor-ad-oracles-support.md`

**Step 1: Add failing replay classification expectations**

Move the new families from unsupported to supported in tests.

Include:

- #433 through #438 families
- `multi_dot`
- `pinv_hermitian`
- `vecdot`

**Step 2: Run the oracle replay test to verify it fails**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db
```

Expected: FAIL on unsupported-classification drift or missing replay logic.

**Step 3: Implement replay mappings**

Add support classifications and observable execution using the new public APIs.

Do not call backend-only helpers from replay.

**Step 4: Regenerate or update the support report if classification changed**

Use the existing report test as the source of truth.

**Step 5: Re-run the oracle replay test**

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg/tests/oracle_db/support.rs \
  tenferro-linalg/tests/oracle_db/replay.rs \
  tenferro-linalg/tests/oracle_db/hvp.rs \
  tenferro-linalg/tests/oracle_db/main.rs \
  docs/generated/tensor-ad-oracles-support.md
git commit -m "feat: replay the remaining oracle-completion families"
```

### Task 9: Update design docs and public documentation

**Files:**
- Modify: `docs/design/linalg.md`
- Modify: relevant crate-level docs and rustdoc comments touched above

**Step 1: Add failing docs consistency check if needed**

If the current docs do not mention the new surface, update the docs first and
use `cargo doc` plus docs-site checks as verification.

**Step 2: Update docs**

Describe:

- new result structs
- primal-only vs differentiable new APIs
- tensorized solve and tensor construction surface
- oracle replay coverage changes

**Step 3: Run docs verification**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 4: Commit**

```bash
git add docs/design/linalg.md
git commit -m "docs: describe completed oracle surface"
```

### Task 10: Run the repository verification gates before PR work

**Files:**
- No new files; verification only

**Step 1: Run formatting**

```bash
cargo fmt --all --check
```

Expected: PASS. If it fails, run `cargo fmt --all` and re-check.

**Step 2: Run oracle replay and high-value targeted tests again**

```bash
cargo test -p tenferro-linalg --test oracle_db
cargo test -p tenferro-linalg
cargo test -p tenferro-dyadtensor
```

Expected: PASS.

**Step 3: Run PR-readiness checks**

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify oracle completion branch readiness"
```
