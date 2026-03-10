# Dense CPU Generic Parity Bundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close `#443`, `#444`, and `#445` and deliver the dense CPU audit required by `#446` in one coherent PR, while enforcing CPU/GPU-generic architectural rules.

**Architecture:** Treat this PR as an audit-and-foundation bundle, not a broad feature bundle. The PR should add a durable dense CPU parity audit, align the human/AI architecture references with the current crate split, finish rustdoc on the new prim families, and extract LAPACK-specific eigen helpers out of `LinalgScalar` so the linalg kernel layer stays backend-generic.

**Tech Stack:** Rust workspace docs, rustdoc, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`, `AGENTS.md`, GitHub issues, workspace verification commands.

---

### Task 1: Add the dense CPU audit document skeleton

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

### Task 2: Populate the family coverage matrix and mapping appendix

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

### Task 3: Record layer findings and backlog categories in the audit

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
rg -n 'tenferro-linalg-prims|TensorScalarPrims|TensorAnalyticPrims|CPU-only|with_cpu_runtime|ensure_cpu_backend' \
  docs/design/architecture.md docs/design/tensor-prims.md docs/design/linalg-prims.md docs/design/linalg.md docs/design/autodiff.md
```

Expected: current references are incomplete or do not explicitly classify the
CPU-only runtime assumptions.

**Step 2: Update the design docs**

Add or tighten the following:

- `tenferro-linalg-prims` in the architecture narrative where missing
- explicit note that `with_cpu_runtime(...)`, `CpuContext`, and
  `ensure_cpu_backend(...)` are current debt, not desired final architecture
- backlog categories separating substrate gaps from layer gaps

**Step 3: Re-run the consistency check**

Run the same `rg` command as Step 1.
Expected: all intended design docs now mention the relevant layers/debt areas.

**Step 4: Commit**

```bash
git add docs/design/architecture.md docs/design/tensor-prims.md docs/design/linalg-prims.md docs/design/linalg.md docs/design/autodiff.md docs/design/reference/pytorch-dense-cpu-parity.md
git commit -m "docs: record dense CPU layer findings and backlog"
```

### Task 4: Update `AGENTS.md` architecture diagrams for `tenferro-linalg-prims`

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

### Task 5: Add missing rustdoc to `ScalarPrims` and `AnalyticPrims`

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

### Task 6: Extract LAPACK eigen helpers out of `LinalgScalar`

**Files:**
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/lib.rs`
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

In `tenferro-linalg/src/lib.rs` and tests:

- stop exercising those methods through `LinalgScalar`
- move any direct helper tests to the new trait

**Step 3: Re-run targeted tests**

Run:

```bash
cargo test -p tenferro-linalg-prims
cargo test -p tenferro-linalg
```

Expected: PASS

**Step 4: Commit**

```bash
git add tenferro-linalg-prims/src/lib.rs tenferro-linalg-prims/src/tests/mod.rs tenferro-linalg/src/backend/cpu_tensor_impl.rs tenferro-linalg/src/lib.rs tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: extract LAPACK eigen helpers from LinalgScalar"
```

### Task 7: Tie the audit back to the GitHub issue bundle

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

### Task 8: Run full verification and prepare the single PR

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
