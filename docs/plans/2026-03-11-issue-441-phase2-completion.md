# Issue #441 Phase 2 Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish `#441` without custom CUDA kernels by making the scalar/analytic and linalg-facing surfaces consistently CPU/GPU-generic, then landing the remaining non-CUDA op families and AD wiring in one branch.

**Architecture:** Treat the rest of `#441` as a single cleanup-and-completion stream. First remove the remaining `with_cpu_runtime(...)` and `ensure_cpu_backend(...)` shortcuts from high-level surfaces, replacing them with centralized runtime dispatch plus honest capability queries. Then widen the non-CUDA op vocabulary only where it is still missing (`Where`, remaining analytic unary ops, `Var`, `Std`) and wire the same family surface through prims, dyadtensor, AD, docs, and parity tracking.

**Tech Stack:** Rust workspace crates, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`, `extension/tenferro-dyadtensor`, `extern/chainrules-scalarops`, `tensor-ad-oracles` replay tests, workspace verification commands.

---

### Task 1: Add failing regression coverage for the runtime-generic contract

**Files:**
- Create: `extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`
- Create: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Write the failing dyadtensor runtime-dispatch tests**

Add tests for representative families that currently still route through `with_cpu_runtime(...)`:

```rust
#[test]
fn unary_and_reduction_entrypoints_route_through_runtime_dispatch() {}

#[test]
fn linalg_entrypoints_report_runtime_capability_failures() {}

#[test]
fn structured_einsum_uses_shared_runtime_dispatch_path() {}
```

Cover at least:

- scalar/analytic representative ops: `exp`, `add`, `mean`
- one semiring path: `einsum`
- one linalg primal path: `svd` or `solve`
- one structured path: structured `einsum`

The expected failure mode before implementation is that these paths still depend
on direct `with_cpu_runtime(...)` shortcuts or CPU-only helpers.

**Step 2: Write the failing linalg capability tests**

Add tests that pin the intended replacement for `ensure_cpu_backend(...)`:

```rust
#[test]
fn capability_checked_composite_paths_do_not_require_cpu_type_checks() {}

#[test]
fn cpu_only_kernel_paths_fail_through_capability_not_backend_name() {}
```

Use representative functions such as `lu_solve`, `lstsq`, `matrix_power`, and
`cond`.

**Step 3: Record the baseline debt in the parity audit**

Update the follow-up list in
`docs/design/reference/pytorch-dense-cpu-parity.md` so the phase-2 target is
explicit:

- remove high-level `with_cpu_runtime(...)` use from `dyadtensor`
- replace removable `ensure_cpu_backend(...)` sites in `tenferro-linalg`
- keep CPU-only kernels honest through capability failures

**Step 4: Run the targeted tests and audits**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg runtime_capability -- --nocapture
rg -n 'with_cpu_runtime\(' extension/tenferro-dyadtensor/src -g '*.rs'
rg -n 'ensure_cpu_backend\(' tenferro-linalg/src -g '*.rs'
```

Expected:

- the new tests fail
- the `rg` baselines show many remaining call sites

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs extension/tenferro-dyadtensor/src/api/tests/mod.rs tenferro-linalg/src/tests/runtime_capability.rs tenferro-linalg/src/tests/mod.rs docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "test: expose issue 441 phase 2 runtime cleanup gaps"
```

### Task 2: Introduce a single dyadtensor runtime-dispatch core

**Files:**
- Create: `extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/lib.rs`

**Step 1: Write the failing dispatch-core unit tests if needed**

Add focused tests in `runtime_dispatch.rs` or the new `runtime_dispatch`
test module that prove the helper can:

- route by `RuntimeContext`
- preserve the operation name in errors
- distinguish capability failures from shape/argument failures

**Step 2: Add the shared runtime-dispatch module**

Implement a small runtime-dispatch layer with helpers such as:

```rust
pub(crate) fn with_runtime_unary<...>(...) -> Result<...> { ... }
pub(crate) fn with_runtime_binary<...>(...) -> Result<...> { ... }
pub(crate) fn with_runtime_linalg<...>(...) -> Result<...> { ... }
pub(crate) fn unsupported_runtime_capability(op: &'static str, runtime: &str) -> Error { ... }
```

Rules:

- the helper accepts `RuntimeContext`, not `CpuContext`
- the helper asks family capability first when the backend supports that query
- unsupported runtime/op combinations surface one consistent error style
- no new public API mentions `CpuBackend` or `CpuContext`

**Step 3: Demote `with_cpu_runtime(...)`**

Keep `with_cpu_runtime(...)` only as a temporary private compatibility helper
inside `runtime.rs`, or delete it entirely if all call sites are removed by the
end of this task series. Do not re-export it from `mod.rs`.

**Step 4: Re-run the targeted dispatch tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
```

Expected: the dispatch-core tests pass, while the broader call-site tests still
fail until later tasks migrate callers.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/runtime_dispatch.rs extension/tenferro-dyadtensor/src/api/mod.rs extension/tenferro-dyadtensor/src/api/runtime.rs extension/tenferro-dyadtensor/src/api/scalar_runtime.rs extension/tenferro-dyadtensor/src/lib.rs
git commit -m "refactor: add shared dyadtensor runtime dispatch core"
```

### Task 3: Migrate dyadtensor primal builders and structured einsum to runtime dispatch

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/primal_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/linalg_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/structured/einsum.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`

**Step 1: Route primal builder entrypoints through `runtime_dispatch`**

Replace direct `with_cpu_runtime(...)` usage in:

- `einsum`
- linalg builder entrypoints (`svd`, `qr`, `lu`, `lu_factor(_ex)`, `eigen`,
  `lstsq`, `cholesky(_ex)`, `solve(_ex)`, `lu_solve`, `inv(_ex)`, `det`,
  `slogdet`, `eig`, `pinv`, `matrix_exp`, `matrix_power`, `solve_triangular`,
  `norm`, `cond`, `cross`, `householder_product`, `vander`, `tensorinv`,
  `tensorsolve`)

Use the shared runtime helper rather than open-coding `match RuntimeContext`.

**Step 2: Route structured einsum through the same helper**

Update `structured/einsum.rs` so:

- `structured_to_dense`
- `structured_einsum`

share the same runtime-dispatch story as the rest of dyadtensor.

**Step 3: Run the focused tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor runtime_dispatch -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor primal -- --nocapture
```

Expected: the representative primal and structured runtime tests now pass.

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/primal_builders.rs extension/tenferro-dyadtensor/src/api/linalg_builders.rs extension/tenferro-dyadtensor/src/structured/einsum.rs extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs extension/tenferro-dyadtensor/src/api/tests/mod.rs
git commit -m "refactor: route dyadtensor primal entrypoints through runtime dispatch"
```

### Task 4: Migrate dyadtensor AD entrypoints and generic scalar builders

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs`

**Step 1: Replace generic scalar AD call sites**

Convert the phase-1 generic scalar builders away from direct CPU access:

- unary generic builders
- binary generic builders
- reduction generic builders

No builder should mention `CpuContext` directly after this task.

**Step 2: Replace legacy AD call sites in `ad.rs` and `ad_builders.rs`**

Migrate the representative high-level AD entrypoints:

- `einsum_rrule`, `einsum_frule`, `einsum_hvp`
- `solve_triangular_rrule`
- linalg AD builders for `svd`, `qr`, `lu`, `eigen`, `lstsq`, `slogdet`,
  `eig`, `solve_triangular`

The rule math stays unchanged. Only the runtime plumbing changes.

**Step 3: Update the AD tests**

Extend the existing AD test modules so they assert:

- runtime-dispatched scalar AD paths still compute the same primal/VJP/JVP
- linalg AD paths now fail via runtime capability rather than CPU-name guards

**Step 4: Run the focused AD tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor ad:: -- --nocapture
```

Expected: targeted scalar and linalg AD tests pass.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/ad.rs extension/tenferro-dyadtensor/src/api/ad_builders.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders.rs extension/tenferro-dyadtensor/src/api/tests/runtime_dispatch.rs extension/tenferro-dyadtensor/src/api/ad/tests/mod.rs extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs
git commit -m "refactor: make dyadtensor ad paths runtime generic"
```

### Task 5: Eliminate remaining mid-level `with_cpu_runtime(...)` shortcuts

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/mod.rs`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Remove or confine the helper**

By the end of this task:

- no production call site under `extension/tenferro-dyadtensor/src/api/**`
  may call `with_cpu_runtime(...)`
- if the helper still exists, it must be private to legacy test support only

**Step 2: Replace test-only CPU shortcuts**

Move `expected` calculations in tests away from `with_cpu_runtime(...)` and
either:

- use the new runtime-dispatch helper, or
- build typed CPU expectations directly in the specific test

**Step 3: Verify with static audit**

Run:

```bash
rg -n 'with_cpu_runtime\(' extension/tenferro-dyadtensor/src -g '*.rs'
```

Expected: either zero production matches or a single private helper definition
with no production callers.

**Step 4: Update the parity audit**

Mark the dyadtensor runtime story as moved from `Partial` toward `Yes/Partial`
with the remaining debt limited to actual backend coverage, not CPU-only API
plumbing.

**Step 5: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/runtime.rs extension/tenferro-dyadtensor/src/api/tests/mod.rs extension/tenferro-dyadtensor/src/api/ad/tests/mod.rs docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "refactor: remove dyadtensor cpu runtime shortcuts"
```

### Task 6: Add linalg capability queries and replace removable `ensure_cpu_backend(...)` sites

**Files:**
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`
- Modify: `tenferro-linalg/src/primal.rs`
- Modify: `tenferro-linalg/src/frules.rs`
- Modify: `tenferro-linalg/src/rrules.rs`
- Modify: `tenferro-linalg/src/ad_helpers.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Add family-level linalg capability queries**

Extend `TensorLinalgPrims` with a family-level capability query. The exact API
can be small, for example:

```rust
fn has_linalg_support(op: LinalgKernelOp) -> bool;
```

or a more structured equivalent if the existing result-type families make that
clearer. The important rule is that higher layers stop using context-type
checks as the main gate.

**Step 2: Classify current `ensure_cpu_backend(...)` call sites**

Use the A/B/C split from the design:

- `A`: pure wrapper guards
- `B`: composite but helper-backed
- `C`: genuinely CPU-only execution

Write the classification directly into comments or the test module so the plan
survives code review.

**Step 3: Replace all `A` and `B` sites**

Representative targets:

- `lu_solve`
- `lstsq`
- `inv_frule`
- `det_frule`
- `slogdet_frule`
- `pinv_frule`
- `matrix_exp_frule`
- `norm_frule`
- the corresponding `rrule` sites

Replace `ensure_cpu_backend(...)` with capability-driven branching plus normal
shape/contract validation.

**Step 4: Keep `C` sites honest**

For genuinely CPU-only paths, keep the public API generic but fail through the
new capability contract rather than backend-name checks.

**Step 5: Run the targeted linalg tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg runtime_capability -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg batch_a_contracts -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg batch_b_contracts -- --nocapture
rg -n 'ensure_cpu_backend\(' tenferro-linalg/src -g '*.rs'
```

Expected: tests pass and only genuinely CPU-only islands remain, if any.

**Step 6: Commit**

```bash
git add tenferro-linalg-prims/src/lib.rs tenferro-linalg/src/prims_bridge.rs tenferro-linalg/src/primal.rs tenferro-linalg/src/frules.rs tenferro-linalg/src/rrules.rs tenferro-linalg/src/ad_helpers.rs tenferro-linalg/src/tests/runtime_capability.rs tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: make linalg backend checks capability driven"
```

### Task 7: Land the remaining non-CUDA scalar and analytic ops

**Files:**
- Modify: `tenferro-prims/src/scalar_prims.rs`
- Modify: `tenferro-prims/src/scalar_cpu.rs`
- Modify: `tenferro-prims/src/analytic_prims.rs`
- Modify: `tenferro-prims/src/analytic_cpu.rs`
- Modify: `tenferro-prims/src/family_cpu_common.rs`
- Modify: `tenferro-prims/src/tests/scalar_phase1.rs`
- Modify: `tenferro-prims/src/tests/analytic_phase1.rs`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Expand the public family vocabulary**

Add the remaining planned non-CUDA ops:

- scalar: `Where` if it is still missing from the scalar family surface
- analytic unary: `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`, `Acosh`,
  `Atanh`
- analytic reductions: `Var`, `Std`

Do not add custom CUDA execution. GPU capability remains truthful and narrow.

**Step 2: Write the failing tests first**

Add representative primal tests for:

```rust
#[test]
fn scalar_where_executes_on_cpu() {}

#[test]
fn analytic_extra_unary_ops_execute_on_cpu() {}

#[test]
fn analytic_var_and_std_execute_on_cpu() {}
```

Use small deterministic tensors and hard-coded expected values.

**Step 3: Implement the CPU execution paths**

Keep the same rules as phase 1:

- specialize at plan time
- no per-element enum matching in hot loops
- no new CPU-only public API

For `Var` and `Std`, decide the accumulation and normalization policy once in
planning, then reuse the same executor path.

**Step 4: Run the focused prim tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-prims scalar_phase1 -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-prims analytic_phase1 -- --nocapture
```

Expected: the expanded scalar/analytic suites pass on CPU; GPU capability tests
remain honest.

**Step 5: Commit**

```bash
git add tenferro-prims/src/scalar_prims.rs tenferro-prims/src/scalar_cpu.rs tenferro-prims/src/analytic_prims.rs tenferro-prims/src/analytic_cpu.rs tenferro-prims/src/family_cpu_common.rs tenferro-prims/src/tests/scalar_phase1.rs tenferro-prims/src/tests/analytic_phase1.rs docs/design/tensor-prims.md docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "feat: complete non-cuda scalar and analytic phase 2 ops"
```

### Task 8: Expand scalar AD rules and dyadtensor family wiring for the remaining ops

**Files:**
- Modify: `extern/chainrules-scalarops/src/lib.rs`
- Modify: `extern/chainrules-scalarops/src/tests/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_runtime.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/scalar_ad_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs`
- Modify: `docs/AD/scalar_ops.md`

**Step 1: Add failing rule tests**

Add representative tests for the newly added families:

```rust
#[test]
fn asin_and_atanh_rules_match_known_derivatives() {}

#[test]
fn var_and_std_rules_match_manual_jvp_vjp_checks() {}

#[test]
fn where_builder_preserves_primal_and_ad_contracts() {}
```

**Step 2: Extend backend-independent scalar rules**

Add the math helpers and VJP/JVP logic for:

- the remaining analytic unary families
- `Var`
- `Std`
- `Where` if it is admitted into the scalar family surface

Keep all rule math backend-independent.

**Step 3: Extend dyadtensor wiring**

Route the same new ops through:

- runtime-dispatched primal builder path
- runtime-dispatched VJP path
- runtime-dispatched JVP path

Do not add any `CpuContext`-typed API.

**Step 4: Run the targeted AD tests**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p chainrules-scalarops --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor scalar_generic -- --nocapture
```

Expected: the expanded scalar AD surface passes with the same runtime-generic
contract as the phase-1 families.

**Step 5: Commit**

```bash
git add extern/chainrules-scalarops/src/lib.rs extern/chainrules-scalarops/src/tests/mod.rs extension/tenferro-dyadtensor/src/api/scalar_runtime.rs extension/tenferro-dyadtensor/src/api/scalar_ad_builders.rs extension/tenferro-dyadtensor/src/api/ad/tests/scalar_generic.rs docs/AD/scalar_ops.md
git commit -m "feat: extend scalar ad coverage for issue 441 completion"
```

### Task 9: Close the docs, parity, and oracle bookkeeping

**Files:**
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/api_index.md`
- Modify: `docs/generated/tensor-ad-oracles-support.md` if replay support changes
- Modify: `tenferro-linalg/tests/oracle_db/support.rs` if newly supported scalar-output rows are enabled

**Step 1: Update the design docs**

Reflect the post-phase2 state:

- dyadtensor runtime is dispatch-based rather than CPU-shortcut-based
- linalg now uses capability checks where possible
- scalar/analytic phase-2 inventory is complete on CPU
- custom CUDA kernels are still intentionally deferred

**Step 2: Update parity tracking**

Revisit the coverage matrix in
`docs/design/reference/pytorch-dense-cpu-parity.md` and move the relevant
families upward only where the implementation now deserves it.

**Step 3: Update oracle support bookkeeping if any rows changed**

If phase-2 work enables additional scalar-output or solver-family replay rows,
regenerate the support table and keep the docs in sync.

**Step 4: Commit**

```bash
git add docs/design/architecture.md docs/design/tensor-prims.md docs/design/reference/pytorch-dense-cpu-parity.md docs/api_index.md docs/generated/tensor-ad-oracles-support.md tenferro-linalg/tests/oracle_db/support.rs
git commit -m "docs: update issue 441 completion status"
```

### Task 10: Run the full verification gate, request review, and create the PR

**Files:**
- Modify: none unless verification fails

**Step 1: Run formatting and targeted crate tests one last time**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo fmt --all --check
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-prims --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-linalg --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p tenferro-dyadtensor --lib -- --nocapture
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test -p chainrules-scalarops --lib -- --nocapture
```

Fix any failures before continuing.

**Step 2: Run the workspace PR gate**

Run:

```bash
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo test --workspace --release
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
env CARGO_TARGET_DIR=/tmp/tenferro-implement-441-target cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py --doc-root /tmp/tenferro-implement-441-target/doc
```

If any check fails, fix it before opening the PR.

**Step 3: Run downstream performance sanity only after correctness gates**

Run:

```bash
cd ../tenferro-einsum-benchmark
env CARGO_TARGET_DIR=/tmp/tenferro-einsum-benchmark-target cargo run --release
```

This is a regression sanity check only. Do not widen the scope into custom CUDA
kernel work.

**Step 4: Request code review**

Use the review workflow after all local checks pass.

**Step 5: Push and create the PR**

Use:

```bash
git push -u origin implement-441
gh pr create --title "feat: complete issue 441 cpu-generic substrate cleanup" --body-file <prepared-body>
gh pr merge --auto --squash --delete-branch <pr-number>
bash scripts/monitor-pr-checks.sh <pr-number> --interval 30
```

Make sure the PR body includes:

- a short summary of runtime/layer cleanup
- the completed non-CUDA scalar/analytic inventory
- the explicit note that custom CUDA kernels remain follow-up work
- `Generated with [Claude Code](https://claude.com/claude-code)`
