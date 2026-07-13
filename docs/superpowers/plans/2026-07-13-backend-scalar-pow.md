# Backend Scalar `pow` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native rank-0 operand support to direct CPU and CUDA `pow` execution without dense scalar materialization.

**Architecture:** General broadcasting remains in runtime/eager/traced layers. CPU gains a scalar-aware mapping helper for owned and read-view paths; CUDA reuses its scalar launch boundary with pow-specific float and checked-integer kernels. Existing equal-shape behavior remains unchanged.

**Tech Stack:** Rust, strided-kernel views, CubeCL CUDA kernels, cargo tests, A100 ignored tests.

---

### Task 1: Establish CPU RED coverage

**Files:**
- Modify: `crates/tenferro-cpu/src/tests/analytic_tests.rs`
- Modify: `crates/tenferro-cpu/src/analytic.rs` only after RED

- [ ] Add direct `CpuBackend::pow` tests for F32/F64/I32/I64 with tensor-base/scalar-exponent and scalar-base/tensor-exponent inputs. Assert output shape and exact representative values.
- [ ] Add negative integer scalar and tensor exponent cases asserting the existing typed error.
- [ ] Add unequal non-scalar shape rejection and empty non-scalar output cases.
- [ ] Run the exact new test target and confirm failure is `ShapeMismatch` for a scalar case, not a test setup error.
- [ ] Commit the RED tests separately.

### Task 2: Implement CPU scalar mapping

**Files:**
- Modify: `crates/tenferro-cpu/src/analytic.rs`
- Test: `crates/tenferro-cpu/src/tests/analytic_tests.rs`

- [ ] Add a private helper that accepts two typed views, permits exact shape or exactly one rank-0 operand, selects the non-scalar output shape, reads the scalar once, and writes one pooled output in operand order.
- [ ] Route owned `typed_pow_with_pool` and view-based `typed_pow_view_with_pool` through the helper without allocating a broadcast tensor.
- [ ] Keep integer exponent validation before mapping and retain exact error fields.
- [ ] Run the new CPU tests and relevant analytic crate tests; confirm GREEN.
- [ ] Run formatting and commit the CPU implementation.

### Task 3: Establish CUDA RED and source contracts

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`
- Modify: `crates/tenferro-gpu/tests/cubecl_launch_contract.rs`

- [ ] Replace the scalar-pow rejection assertions with CPU-oracle parity for both operand positions and F32/F64/I32/I64.
- [ ] Include empty tensors, unequal non-scalar rejection, negative integer exponents, and floating exceptional-class checks.
- [ ] Change the source contract to require a scalar launch branch for `pow`, forbid `broadcast_typed`, and preserve dtype-before-shape validation.
- [ ] Run the source contract and confirm RED because `pow` lacks the scalar launcher.
- [ ] On A100, run the focused ignored test and confirm RED with current `ShapeMismatch`.
- [ ] Commit the RED CUDA tests separately.

### Task 4: Implement CUDA scalar pow kernels

**Files:**
- Modify: `crates/tenferro-gpu/src/cubecl/kernels/elementwise.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Test: `crates/tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`
- Test: `crates/tenferro-gpu/tests/cubecl_launch_contract.rs`

- [ ] Add `scalar_pow_float`, reading index zero from the rank-0 operand and the output index from the tensor operand.
- [ ] Add `scalar_pow_int_checked` with the existing negative-exponent device flag contract.
- [ ] Route only unequal-shape, exactly-one-rank-0 pow inputs through existing scalar launch helpers; leave equal-shape kernels untouched.
- [ ] Preserve residency validation and empty-output ordering in the shared launch helpers.
- [ ] Run source contracts, CUDA feature no-run compilation, and the focused A100 test; confirm GREEN.
- [ ] Run `cargo fmt --all --check` and commit the CUDA implementation.

### Task 5: Neighborhood scan and documentation

**Files:**
- Modify if needed: `docs/guides/devices-and-gpu.md`
- Create: `docs/worklogs/2026-07-13-backend-scalar-pow.md`

- [ ] Search current active docs and capability tables for claims that scalar `pow` is rejected; update only active statements.
- [ ] Confirm public/runtime/eager/traced pow already broadcast explicitly and require no production changes.
- [ ] Record issue context, external semantic references, RED/GREEN evidence, design choice, rejected backend-general broadcasting, and residual risks in the worklog.
- [ ] Run docs/source contract checks and commit documentation.

### Task 6: Full verification, review, and PR

**Files:**
- Modify if findings require: files already in scope

- [ ] Run full ignored CUDA suite on A100 with the supported CUDA environment.
- [ ] Run `cargo fmt --all --check` and CI-exact workspace/tropical clippy.
- [ ] Run `cargo test --workspace --release`.
- [ ] Run release llvm-cov plus `scripts/check-coverage.py`.
- [ ] Run workspace docs plus `scripts/check-docs-site.py`.
- [ ] Run committed-head `scripts/repository-rules-review.py` and resolve findings.
- [ ] Perform a final specification and code-quality review against issue #1371 and this plan.
- [ ] Push, create one PR with `Closes #1371`, monitor review and all CI including RunPod GPU gate, resolve actionable feedback, enable normal squash auto-merge, and verify the merged commit and closed issue.
