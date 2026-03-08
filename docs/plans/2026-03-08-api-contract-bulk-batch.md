# API Contract Bulk Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `#377`, `#380`, `#386`, and `#387` by aligning dyadtensor/prims public contracts with actual runtime behavior.

**Architecture:** Dyadtensor scalar operator sugar will become fallible so mixed reverse tapes no longer panic. In `tenferro-prims`, unsupported scalar/op combinations will be rejected during planning, while the generic contract fallback and real-valued `Max/Min` reductions will gain missing execution behavior.

**Tech Stack:** Rust workspace crates, `tenferro-prims`, `tenferro-dyadtensor`, `thiserror`, `cargo test`, `cargo llvm-cov`

---

### Task 1: Add failing dyadtensor tests for fallible operator overloads

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/tests/dyn_ad_scalar_reverse_tests.rs`

**Step 1: Write the failing tests**

Add tests that assert:

- `AdScalar::new_reverse(..., TapeId(7)) + AdScalar::new_reverse(..., TapeId(8))` returns `Err(Error::MixedReverseTape { .. })`
- same for `DynAdScalar`
- same for one mixed scalar overload path such as `DynAdScalar + f64`

**Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cargo test -p tenferro-dyadtensor ad_scalar_binary_op_panics_on_mixed_reverse_tapes
cargo test -p tenferro-dyadtensor dyn_ad_scalar_try_mul_rejects_mixed_reverse_tapes
```

Expected: existing operator tests still panic or type signatures do not yet match the new assertions.

**Step 3: Commit the failing-test checkpoint if useful locally**

```bash
git add extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs extension/tenferro-dyadtensor/tests/dyn_ad_scalar_reverse_tests.rs
git commit -m "test: add dyadtensor mixed-tape operator regressions"
```

Only commit here if the tree is in a coherent failing-test checkpoint; otherwise continue to green first.

### Task 2: Make dyadtensor scalar binary operators return `Result`

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/ad_value.rs`
- Modify: `extension/tenferro-dyadtensor/src/dyn_types.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`
- Test: `extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs`
- Test: `extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs`
- Test: `extension/tenferro-dyadtensor/tests/dyn_ad_scalar_reverse_tests.rs`

**Step 1: Change the operator overload signatures**

Update:

- `impl Add/Sub/Mul/Div for AdScalar<T>` to `type Output = Result<Self>`
- `impl Add/Sub/Mul/Div for DynAdScalar` to `type Output = Result<DynAdScalar>`
- scalar-mixed overload macros in `dyn_types.rs` to propagate `Result<DynAdScalar>`

Keep `Neg` unchanged.

**Step 2: Route all overloads through the checked helpers**

Delete the panic wrapper helper usage and return the checked result directly from the trait methods.

**Step 3: Update docs/examples to use `.unwrap()` where operator syntax is still shown**

Fix rustdoc examples affected by the new `Result` output.

**Step 4: Run targeted tests**

Run:

```bash
cargo test -p tenferro-dyadtensor ad_scalar
cargo test -p tenferro-dyadtensor dyn_ad_scalar
```

Expected: mixed-tape operator regressions pass and no panics remain on those paths.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/ad_value.rs extension/tenferro-dyadtensor/src/dyn_types.rs extension/tenferro-dyadtensor/src/lib.rs extension/tenferro-dyadtensor/src/ad_value/tests/mod.rs extension/tenferro-dyadtensor/src/dyn_types/tests/mod.rs extension/tenferro-dyadtensor/tests/dyn_ad_scalar_reverse_tests.rs
git commit -m "fix(dyadtensor): make scalar operators fallible"
```

### Task 3: Add failing prims regressions for contract/reduce/gemm contracts

**Files:**
- Modify: `tenferro-prims/tests/prims_tests.rs`

**Step 1: Write the failing tests**

Add tests for:

- contract fallback where an `A`-only mode is fully contracted
- contract fallback where a `B`-only mode is fully contracted
- `ReduceOp::Max` and `ReduceOp::Min` on `f64`
- `cpu_plan::<i32>(..., PrimDescriptor::BatchedGemm { .. }, ...)` returning an early error
- `cpu_plan::<Complex64>(..., PrimDescriptor::Reduce { op: ReduceOp::Max, .. }, ...)` returning an early error

Replace the current regression that encodes the broken `A`-only-mode-fixed-at-zero behavior.

**Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims contract_generic_fallback_with_a_only_mode
cargo test -p tenferro-prims reduce_max_returns_error
```

Expected: old fallback semantics or execute-time failure is still observed.

### Task 4: Fix prims contract fallback and move contract checks to plan time

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`
- Modify: `tenferro-prims/src/lib.rs`
- Test: `tenferro-prims/tests/prims_tests.rs`

**Step 1: Fix fallback contraction indexing**

In `execute_contract`, treat every mode absent from `modes_c` as a reduction mode, not just modes shared by `A` and `B`.

**Step 2: Add scalar-support helpers for plan-time validation**

Add internal helpers in `cpu.rs` for:

- batched GEMM supported scalar set: `f32`, `f64`, `Complex32`, `Complex64`
- ordered real reduction scalar set for `Max/Min`: `f32`, `f64`

Call them from `CpuBackend::build_plan`.

**Step 3: Implement `ReduceOp::Max/Min` execution for `f32`/`f64`**

Add reduction helpers parallel to `execute_reduce_sum`, using strided loops and reuse buffers outside hot loops.

**Step 4: Update crate docs/comments**

Clarify in `tenferro-prims/src/lib.rs` that:

- `BatchedGemm` on CPU is only implemented for the four numeric GEMM scalar types
- `ReduceOp::Max/Min` require ordered real scalars on CPU

**Step 5: Run targeted tests**

Run:

```bash
cargo test -p tenferro-prims contract_
cargo test -p tenferro-prims reduce_
cargo test -p tenferro-prims batched_gemm
```

Expected: new regressions pass and old execute-time contract failures are gone.

**Step 6: Commit**

```bash
git add tenferro-prims/src/cpu.rs tenferro-prims/src/lib.rs tenferro-prims/tests/prims_tests.rs
git commit -m "fix(prims): align contract and reduction APIs"
```

### Task 5: Full verification, PR, and auto-merge

**Files:**
- Review: full branch diff

**Step 1: Run formatting and full verification**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands exit `0`.

**Step 2: Inspect the final diff**

Run:

```bash
git status --short
git diff --stat origin/main..HEAD
```

Expected: only files relevant to `#377 #380 #386 #387` plus the saved design/plan docs are included.

**Step 3: Create PR and enable auto-merge**

Run:

```bash
bash scripts/create-pr.sh --title "Fix API contract bugs in dyadtensor and prims" --ai-tool-name "Claude Code" --ai-tool-url "https://claude.com/claude-code"
```

If the branch is behind `main`, update it and re-enable auto-merge:

```bash
gh pr update-branch <pr-number> --rebase
gh pr merge --auto --squash --delete-branch <pr-number>
```

**Step 4: Monitor until merged**

Run:

```bash
bash scripts/monitor-pr-checks.sh <pr-number> --interval 30
```

Expected: required checks pass and GitHub auto-merges the PR.
