# Dense CPU Generic Parity Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close `#443`, `#444`, and `#445` and deliver the dense CPU audit required by `#446` in one coherent PR, while enforcing CPU/GPU-generic architectural rules.

**Architecture:** Treat this PR as an audit-and-foundation bundle, not a broad feature bundle. The PR should add a durable dense CPU parity audit, align the human/AI architecture references with the current crate split, finish rustdoc on the new prim families, extract LAPACK-specific eigen helpers out of `LinalgScalar`, and perform the minimum structural refactors required to stop `tenferro-linalg` and `tenferro-dyadtensor` from accumulating more CPU-only and monolithic code.

**Tech Stack:** Rust workspace docs, rustdoc, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`, `AGENTS.md`, GitHub issues, workspace verification commands.

---

### Task 1: Split `tenferro-linalg/src/lib.rs` before functional edits

**Files:**
- Create: `tenferro-linalg/src/result_types.rs`
- Create: `tenferro-linalg/src/primal.rs`
- Create: `tenferro-linalg/src/ad_helpers.rs`
- Create: `tenferro-linalg/src/rrules.rs`
- Create: `tenferro-linalg/src/frules.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Capture the failing structure checks**

Run:

```bash
wc -l tenferro-linalg/src/lib.rs
rg -n '#\\[cfg\\(test\\)\\]|mod eig_scalar_tests' tenferro-linalg/src/lib.rs
```

Expected:

- `tenferro-linalg/src/lib.rs` is still extremely large
- inline test modules are present in a non-leaf file

**Step 2: Perform the no-behavior-change module split**

Move code by concern:

- result structs, cotangents, and options into `result_types.rs`
- primal public APIs into `primal.rs`
- shared AD helpers into `ad_helpers.rs`
- reverse-mode rules into `rrules.rs`
- forward-mode rules into `frules.rs`

Keep `lib.rs` as crate docs, module declarations, re-exports, and thin glue
only.

**Step 3: Move non-leaf inline tests out of `lib.rs`**

- keep `#[cfg(test)] mod tests;` if needed
- move `eig_scalar_tests` into crate-local test modules under
  `tenferro-linalg/src/tests/`
- keep the later `tenferro-linalg-prims` relocation for Task 9, where the
  LAPACK-specific trait split lands

**Step 4: Re-run the structure checks**

Run the same `wc` and `rg` commands from Step 1.
Expected:

- `lib.rs` is reduced to a thin entrypoint
- no inline non-leaf test module remains there

**Step 5: Commit**

```bash
git add tenferro-linalg/src/result_types.rs tenferro-linalg/src/primal.rs tenferro-linalg/src/ad_helpers.rs tenferro-linalg/src/rrules.rs tenferro-linalg/src/frules.rs tenferro-linalg/src/lib.rs tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: split monolithic tenferro-linalg lib module"
```

### Task 2: Split `tenferro-dyadtensor/src/api/mod.rs` by concern

**Files:**
- Create: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Create: `extension/tenferro-dyadtensor/src/api/primal_builders.rs`
- Create: `extension/tenferro-dyadtensor/src/api/linalg_builders.rs`
- Create: `extension/tenferro-dyadtensor/src/api/ad_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`

**Step 1: Capture the failing structure check**

Run:

```bash
wc -l extension/tenferro-dyadtensor/src/api/mod.rs
```

Expected: the file is still multi-thousand-line and mixes runtime, builder, and
AD concerns.

**Step 2: Perform the module split**

Move code by concern:

- runtime helpers and default runtime accessors into `runtime.rs`
- primal builder surface into `primal_builders.rs`
- linalg builder wrappers into `linalg_builders.rs`
- AD builder helpers and wrappers into `ad_builders.rs`

Keep `mod.rs` focused on module wiring, shared exports, and small shared types.

**Step 3: Re-run the structure check**

Run the same `wc` command as Step 1.
Expected: `mod.rs` is reduced to a thin orchestration file.

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/runtime.rs extension/tenferro-dyadtensor/src/api/primal_builders.rs extension/tenferro-dyadtensor/src/api/linalg_builders.rs extension/tenferro-dyadtensor/src/api/ad_builders.rs extension/tenferro-dyadtensor/src/api/mod.rs
git commit -m "refactor: split dyadtensor api module by concern"
```

### Task 3: Remove obvious CPU-only API leaks from touched linalg AD paths

**Files:**
- Modify: `tenferro-linalg/src/rrules.rs`
- Modify: `tenferro-linalg/src/frules.rs`
- Modify: `tenferro-linalg/src/ad_helpers.rs`
- Modify: `tenferro-linalg/src/primal.rs`

**Step 1: Capture the current CPU-only surface**

Run:

```bash
rg -n 'ctx: &mut tenferro_prims::CpuContext' tenferro-linalg/src/rrules.rs tenferro-linalg/src/frules.rs tenferro-linalg/src/ad_helpers.rs
rg -n 'type_name::<|expect\\(' tenferro-linalg/src/primal.rs tenferro-linalg/src/ad_helpers.rs
```

Expected:

- multiple AD entry points still hard-code `CpuContext`
- `type_name::<...>()` appears in backend checks
- `expect(...)` appears in library code

**Step 2: Tighten the CPU/GPU-generic contract**

- change touched AD entry points from `&mut CpuContext` to backend-parametric
  context bounds where the math already supports it
- keep unsupported backends as explicit runtime errors through trait/capability
  checks rather than CPU-specific function signatures
- replace backend comparison by `TypeId::of::<...>()`
- remove `expect(...)` from library code paths and propagate checked values
  explicitly

**Step 3: Re-run targeted tests**

Run:

```bash
cargo test -p tenferro-linalg
```

Expected: PASS

**Step 4: Re-run the grep audit**

Run the same `rg` command as Step 1.
Expected:

- touched AD APIs no longer hard-code `CpuContext`
- no `type_name::<...>()` backend comparison remains
- the targeted `expect(...)` is gone

**Step 5: Commit**

```bash
git add tenferro-linalg/src/rrules.rs tenferro-linalg/src/frules.rs tenferro-linalg/src/ad_helpers.rs tenferro-linalg/src/primal.rs
git commit -m "refactor: make linalg AD surface less CPU-specific"
```

### Task 4: Add the dense CPU audit document skeleton

**Files:**
- Create: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/design/index.md`
- Modify: `docs/design/reference/libtorch.md`

**Step 1: Write the failing docs-site condition**

Add a link to `docs/design/index.md` that references the new audit document
before the file exists.

**Step 2: Run docs-site verification to confirm the link is unresolved**

Run: `python3 scripts/check-docs-site.py`
Expected: FAIL because the referenced design doc does not exist yet.

**Step 3: Write the minimal document skeleton**

Create `docs/design/reference/pytorch-dense-cpu-parity.md` with these sections:

- Scope
- Audit method
- Coverage matrix
- PyTorch-to-tenferro mapping
- Layer findings
- Follow-up backlog

Also add a short pointer from `docs/design/reference/libtorch.md` to the new
audit document.

**Step 4: Re-run docs-site verification**

Run: `python3 scripts/check-docs-site.py`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/design/reference/pytorch-dense-cpu-parity.md docs/design/index.md docs/design/reference/libtorch.md
git commit -m "docs: add dense CPU parity audit skeleton"
```

### Task 5: Populate the family coverage matrix and mapping appendix

**Files:**
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Write the failing audit checklist**

Create a temporary checklist in the new audit doc with placeholder markers such
as `TODO(primal)` / `TODO(vjp)` / `TODO(jvp)` / `TODO(hvp)`.

**Step 2: Verify placeholders remain**

Run: `rg -n 'TODO\\((primal|vjp|jvp|hvp)' docs/design/reference/pytorch-dense-cpu-parity.md`
Expected: matches are found.

**Step 3: Replace placeholders with the actual matrix and mapping**

Fill the document with:

- family rows for `Structural`, `Semiring core/fast path`, `Scalar`,
  `Analytic`, `Linalg kernel`, `Linalg composite`, and `Dyadtensor/AD surface`
- columns for `primal`, `VJP`, `JVP`, `oracle-HVP`, `CPU-generic`, and
  `layer-clean`
- explicit notes for current structure problems such as monolithic modules and
  CPU-only AD/runtime assumptions
- a PyTorch-to-tenferro appendix that groups PyTorch APIs by tenferro family
  rather than by exact surface spelling

**Step 4: Verify placeholders are gone**

Run: `rg -n 'TODO\\((primal|vjp|jvp|hvp)' docs/design/reference/pytorch-dense-cpu-parity.md`
Expected: no matches

**Step 5: Commit**

```bash
git add docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "docs: fill dense CPU parity matrix and mapping"
```

### Task 6: Record layer findings and backlog categories in the audit

**Files:**
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/linalg-prims.md`
- Modify: `docs/design/linalg.md`
- Modify: `docs/design/autodiff.md`

**Step 1: Write the failing architecture consistency check**

Run:

```bash
rg -n 'tenferro-linalg-prims|TensorScalarPrims|TensorAnalyticPrims|CPU-only|with_cpu_runtime|ensure_cpu_backend|Permute' \
  docs/design/architecture.md docs/design/tensor-prims.md docs/design/linalg-prims.md docs/design/linalg.md docs/design/autodiff.md
```

Expected: current references are incomplete or do not explicitly classify the
CPU-only runtime assumptions and pending `Permute` cleanup.

**Step 2: Update the design docs**

Add or tighten the following:

- `tenferro-linalg-prims` in the architecture narrative where missing
- explicit note that `with_cpu_runtime(...)`, `CpuContext`, and
  `ensure_cpu_backend(...)` are current debt, not desired final architecture
- explicit note that `PrimDescriptor::Permute` remains as legacy debt tracked by
  `#441`, not part of this bundle
- backlog categories separating substrate gaps from layer gaps

**Step 3: Re-run the consistency check**

Run the same `rg` command as Step 1.
Expected: all intended design docs now mention the relevant layers/debt areas.

**Step 4: Commit**

```bash
git add docs/design/architecture.md docs/design/tensor-prims.md docs/design/linalg-prims.md docs/design/linalg.md docs/design/autodiff.md docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "docs: record dense CPU layer findings and backlog"
```

### Task 7: Update `AGENTS.md` architecture diagrams for `tenferro-linalg-prims`

**Files:**
- Modify: `AGENTS.md`

**Step 1: Write the failing grep check**

Run: `rg -n 'tenferro-linalg-prims' AGENTS.md`
Expected: no matches.

**Step 2: Update the layered and dependency diagrams**

Edit both architecture blocks in `AGENTS.md` so they reflect:

- `tenferro-prims` and `tenferro-linalg-prims` as separate layer-3 crates
- `tenferro-linalg` depending on `tenferro-linalg-prims`
- the crate split introduced by PR `#442`

**Step 3: Re-run the grep check**

Run: `rg -n 'tenferro-linalg-prims' AGENTS.md`
Expected: matches in both the layered design and dependency graph sections.

**Step 4: Commit**

```bash
git add AGENTS.md
git commit -m "docs: align AGENTS architecture with linalg prim split"
```

### Task 8: Add missing rustdoc to `ScalarPrims` and `AnalyticPrims`

**Files:**
- Modify: `tenferro-prims/src/scalar_prims.rs`
- Modify: `tenferro-prims/src/analytic_prims.rs`

**Step 1: Add the failing documentation expectation**

Record the missing method/field docs in comments or a temporary checklist, then
run:

```bash
cargo doc -p tenferro-prims --no-deps
```

Expected: PASS build-wise, but inspection should show missing detail on
descriptor fields and trait methods.

**Step 2: Add rustdoc**

For both files:

- add field-level docs on descriptor variants
- add method docs on `plan`, `execute`, and support-query methods
- mark forward-declared but currently unsupported variants as reserved / not yet
  wired
- keep examples short and aligned with the semiring-family docs added in PR
  `#442`

**Step 3: Verify rustdoc output**

Run:

```bash
cargo test -p tenferro-prims --doc
cargo doc -p tenferro-prims --no-deps
```

Expected: both pass.

**Step 4: Commit**

```bash
git add tenferro-prims/src/scalar_prims.rs tenferro-prims/src/analytic_prims.rs
git commit -m "docs: complete scalar and analytic prim rustdoc"
```

### Task 9: Extract LAPACK eigen helpers out of `LinalgScalar`

**Files:**
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`
- Create: `tenferro-linalg-prims/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests that assume a new `LapackEigScalar`-style separation exists.

At minimum:

- a `tenferro-linalg-prims` unit test that calls the LAPACK eigen helper trait
  directly for `f32`, `f64`, `Complex32`, and `Complex64`
- an updated `tenferro-linalg` unit test that no longer expects
  `eig_buffer_sizes` / `eig_ri_to_complex` on `LinalgScalar` itself

Run:

```bash
cargo test -p tenferro-linalg-prims
cargo test -p tenferro-linalg private_scalar_and_validation_helpers_are_covered_in_crate_unit_tests
```

Expected: FAIL because the trait split does not exist yet.

**Step 2: Implement the trait split**

In `tenferro-linalg-prims/src/lib.rs`:

- remove `eig_buffer_sizes` and `eig_ri_to_complex` from `LinalgScalar`
- introduce a new trait, tentatively `LapackEigScalar: LinalgScalar`
- move the four scalar-type implementations of the eigen helpers to the new
  trait

In `tenferro-linalg/src/backend/cpu_tensor_impl.rs`:

- tighten only the CPU eig path to require `LapackEigScalar`

In tests:

- stop exercising those methods through `LinalgScalar`
- move direct helper coverage to the new trait test module

**Step 3: Re-run targeted tests**

Run:

```bash
cargo test -p tenferro-linalg-prims
cargo test -p tenferro-linalg
```

Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-linalg-prims/src/lib.rs tenferro-linalg-prims/src/tests/mod.rs tenferro-linalg/src/backend/cpu_tensor_impl.rs tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: extract LAPACK eigen helpers from LinalgScalar"
```

### Task 10: Tie the audit back to the GitHub issue bundle

**Files:**
- Modify: `docs/design/reference/pytorch-dense-cpu-parity.md`

**Step 1: Add the failing issue traceability check**

Run: `rg -n '#443|#444|#445|#446|#441' docs/design/reference/pytorch-dense-cpu-parity.md`
Expected: incomplete issue coverage.

**Step 2: Add explicit traceability**

Document:

- which sections close `#443`
- which sections close `#444`
- which trait refactor closes `#445`
- how the audit satisfies `#446`
- why `#441` remains a follow-up substrate issue rather than being closed here

**Step 3: Re-run the traceability check**

Run: `rg -n '#443|#444|#445|#446|#441' docs/design/reference/pytorch-dense-cpu-parity.md`
Expected: all five issue numbers appear.

**Step 4: Commit**

```bash
git add docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "docs: link dense CPU audit to issue bundle"
```

### Task 11: Run full verification and prepare the single PR

**Files:**
- Verify all files touched above

**Step 1: Run formatting**

Run: `cargo fmt --all --check`
Expected: PASS

**Step 2: Run correctness and coverage gates**

Run:

```bash
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: PASS

**Step 3: Run docs gates**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS

**Step 4: Create one PR for the whole bundle**

Open a single PR that explicitly closes:

- `#443`
- `#444`
- `#445`
- `#446`

and states that `#441` remains open as the substrate-expansion follow-up.

**Step 5: Commit if needed and push**

```bash
git status --short
git push -u origin <branch-name>
gh pr create
```
