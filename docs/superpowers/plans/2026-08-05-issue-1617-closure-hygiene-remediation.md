# Issue #1617 Closure Hygiene Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every valid finding in Issue #1617, bind all P8–P13 evidence to one corrected product candidate, prove Metal/WebGPU on the Apple M5 Max, and merge one remediation PR.

**Architecture:** Build all Rust, checker, test, CI, and durable-design changes first and freeze their last commit as product candidate C. Generate only allowlisted reports after C, combine Linux/CUDA and Apple lane captures into one required-mode matrix, then produce a recorded closure, a fresh reproduced closure, and a final receipt on evidence HEAD E. Ordinary CI remains receipt/report-only; only final closure runs the bounded reproduction set.

**Tech Stack:** Rust 2021, Cargo, `trybuild`, Python 3.11+, `unittest`, GitHub Actions YAML, Git, Criterion, cargo-llvm-cov, WebGPU/Metal, CUDA.

## Global Constraints

- Preserve one move-only owner per physical allocation.
- Keep duplicate, upload, download, synchronization, and materialization explicit.
- Preserve CUDA, WebGPU, and Apple provider namespaces.
- Do not add hashes, signatures, attestations, nonces, hostile-runner defenses, a new generic runner, or a new dependency.
- Add checks only for failures demonstrated by #1617: macOS compilation, stale candidate binding, missing/duplicate/non-passing required lanes, receipt-command mismatch, failed reproduction, and product changes after C. Reuse existing Git and receipt validation rather than duplicating them.
- Keep receipt/report-only checking as ordinary CI behavior; run fresh reproduction only for final closure or an explicit release-candidate audit.
- Final evidence requires positive test counts and `pass` for CPU, CUDA, WebGPU, Metal, and CUDA-AD; a skip or zero-test Metal run blocks closure.
- Keep all work in one remediation branch and one non-squash PR with coherent commits.
- Retain `sha2`; `crates/tenferro-xla/src/stablehlo.rs` uses it.
- Product/checker/test/CI/design changes after candidate C invalidate C and require affected evidence to be regenerated.

Finding coverage: Task 1 resolves T0; Task 3 resolves T1; Task 5 resolves T2; Tasks 4 and 8 resolve T3; Task 2 resolves T4 and the valid T5 items; Tasks 6–10 regenerate and close the corrected umbrella evidence.

## File Map

**Product and CI**

- Modify: `crates/tenferro-gpu/tests/integration/apple_context.rs`
- Modify: `.github/workflows/ci.yml`
- Modify: `scripts/ci/tests/test_workflow_contracts.py`
- Modify: `crates/tenferro-tensor/src/storage/group.rs`
- Modify: `crates/tenferro-tensor/src/storage/root.rs`
- Create: `crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.rs`
- Create: `crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.stderr`

**Evidence checkers and checker tests**

- Modify: `scripts/check-storage-contract-freeze.py`
- Modify: `scripts/check-storage-static-rank-codegen.py`
- Modify: `scripts/verify-storage-traversal-performance.py`
- Modify: `scripts/check-storage-hardware-matrix.py`
- Modify: `scripts/check-storage-redesign-closure.py`
- Create: `scripts/ci/tests/test_storage_evidence_contracts.py`

**Durable contracts and work log**

- Modify: `docs/design/storage-ownership-contracts.md`
- Modify: `docs/superpowers/specs/2026-08-04-p11-frozen-hardware-matrix-design.md`
- Modify: `docs/superpowers/specs/2026-08-04-p13-freeze-closure-design.md`
- Create: `docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md`

**Generated evidence descendants**

- Modify: `docs/design/storage-contract-freeze.md`
- Modify: `docs/testing/storage-traversal-performance.md`
- Modify: `docs/testing/storage-static-rank-codegen.md`
- Modify: `docs/testing/storage-hardware-matrix.md`
- Modify: `docs/worklogs/storage-documentation-source-blind-audit.md`
- Modify: `docs/worklogs/storage-redesign-closure.md`

---

### Task 1: Fix the Metal compile failure and add the cross-target CI guard

**Files:**
- Modify: `crates/tenferro-gpu/tests/integration/apple_context.rs:41-61`
- Modify: `.github/workflows/ci.yml:32-170`
- Modify: `scripts/ci/tests/test_workflow_contracts.py:14-45`

**Interfaces:**
- Consumes: existing `policy.outputs.run_rust` CI classification and `TypedTensor::with_host_write(&mut self, ...)`.
- Produces: required CI job `macos-gated-check` and a compiling mutable Apple managed tensor test.

- [ ] **Step 1: Add the failing workflow-contract test**

Add to `WorkflowContractTests`:

```python
def test_macos_gated_gpu_tests_are_cross_checked(self) -> None:
    text = read(".github/workflows/ci.yml")
    start = text.index("  macos-gated-check:")
    end = text.index("\n  coverage:", start)
    block = text[start:end]
    self.assertIn("name: macOS-gated GPU type-check", block)
    self.assertIn("targets: aarch64-apple-darwin", block)
    self.assertIn(
        "cargo check -p tenferro-gpu --features webgpu --test integration "
        "--target aarch64-apple-darwin",
        block,
    )
    self.assertIn("needs.policy.outputs.run_rust == 'true'", block)
```

- [ ] **Step 2: Run the test and confirm the guard is absent**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_workflow_contracts.WorkflowContractTests.test_macos_gated_gpu_tests_are_cross_checked -v
```

Expected: FAIL with `ValueError: substring not found` for `macos-gated-check`.

- [ ] **Step 3: Add the minimal CI job**

Add a separate Ubuntu job before `coverage` using the same policy/disposition pattern as existing required jobs:

```yaml
  macos-gated-check:
    name: macOS-gated GPU type-check
    needs: policy
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        if: needs.policy.result == 'success' && needs.policy.outputs.run_rust == 'true'
      - uses: dtolnay/rust-toolchain@stable
        if: needs.policy.result == 'success' && needs.policy.outputs.run_rust == 'true'
        with:
          targets: aarch64-apple-darwin
      - uses: Swatinem/rust-cache@v2
        if: needs.policy.result == 'success' && needs.policy.outputs.run_rust == 'true'
      - name: Type-check macOS-gated GPU tests
        if: needs.policy.result == 'success' && needs.policy.outputs.run_rust == 'true'
        run: >-
          cargo check -p tenferro-gpu --features webgpu --test integration
          --target aarch64-apple-darwin
      - name: Report macOS-gated check disposition
        if: always()
        env:
          POLICY_RESULT: ${{ needs.policy.result }}
          REQUIRED: ${{ needs.policy.outputs.run_rust }}
          REASON: ${{ needs.policy.outputs.reason }}
        run: |
          if [ "${POLICY_RESULT}" != success ]; then echo "Change classification failed"; exit 1; fi
          if [ "${REQUIRED}" != true ]; then echo "macOS-gated GPU check not required: ${REASON}"; fi
```

- [ ] **Step 4: Confirm the workflow test passes but the target check reproduces T0**

Run:

```bash
python3 -m unittest scripts.ci.tests.test_workflow_contracts.WorkflowContractTests.test_macos_gated_gpu_tests_are_cross_checked -v
rustup target add aarch64-apple-darwin
cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin
```

Expected before the Rust fix: workflow test PASS; Cargo FAIL with E0596 at `typed.with_host_write(...)`.

- [ ] **Step 5: Apply the one-line Rust fix**

Change:

```rust
let Tensor::F32(typed) = &managed else {
```

to:

```rust
let Tensor::F32(typed) = &mut managed else {
```

Do not add another helper or test-only cfg.

- [ ] **Step 6: Verify the target-gated source now compiles**

Run:

```bash
cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin
python3 -m unittest scripts.ci.tests.test_workflow_contracts -v
python3 scripts/ci/run_profile.py fmt
```

Expected: all PASS.

- [ ] **Step 7: Commit the T0 review unit**

```bash
git add .github/workflows/ci.yml \
  scripts/ci/tests/test_workflow_contracts.py \
  crates/tenferro-gpu/tests/integration/apple_context.rs
git commit -m "fix(gpu): type-check and repair Apple context tests"
```

---

### Task 2: Strengthen static rank and apply verified storage hygiene

**Files:**
- Create: `crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.rs`
- Create: `crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.stderr`
- Modify: `crates/tenferro-tensor/src/storage/group.rs:1241,1289,1359`
- Modify: `crates/tenferro-tensor/src/storage/root.rs:560-566`

**Interfaces:**
- Consumes: existing recursive UI fixture discovery in `storage_compile_contract.rs` and sealed `TensorScalar` dtype dispatch.
- Produces: a compile-time rank mismatch obligation and release-active unsafe preconditions.

- [ ] **Step 1: Add the compile-fail input without its stderr fixture**

Create `static_rank_mismatch.rs`:

```rust
use tenferro_tensor::{Rank, TypedTensor, TypedTensorView};

fn requires_rank_three(_: TypedTensorView<'_, f64, Rank<3>>) {}

fn main() {
    let tensor =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    requires_rank_three(tensor.as_view());
}
```

- [ ] **Step 2: Run trybuild and capture the expected compiler diagnostic**

Run:

```bash
TRYBUILD=overwrite cargo test -p tenferro-tensor --test storage_compile_contract storage_ui_compile_contracts
```

Expected: `static_rank_mismatch.stderr` is generated and contains E0308, `expected Rank<3>, found Rank<2>`.

Inspect the generated stderr; keep only rustc-normalized trybuild output and do not hand-weaken the diagnostic.

- [ ] **Step 3: Apply the two local hygiene changes**

In all three `GroupWriteView` constructions, change:

```rust
owner: NonNull::from(&mut *owner),
```

to:

```rust
owner: NonNull::from(owner),
```

In `cast_host_vec`, change both `debug_assert_eq!` calls to release-active assertions:

```rust
assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
```

Keep the existing `// SAFETY:` explanation and generic function; do not add a trait hierarchy.

- [ ] **Step 4: Verify compile contracts and storage tests**

Run:

```bash
cargo test -p tenferro-tensor --test storage_compile_contract
cargo test -p tenferro-tensor --test storage_static_rank
cargo test -p tenferro-tensor --lib storage::
python3 scripts/ci/run_profile.py fmt
```

Expected: all PASS; the new compile-fail case is exercised by `storage_ui_compile_contracts`.

- [ ] **Step 5: Confirm `sha2` is used and record the command output for the work log**

Run:

```bash
cargo tree -i sha2 --workspace
rg -n 'use sha2::\{Digest, Sha256\}|sha2\.workspace' \
  crates/tenferro-xla/src/stablehlo.rs crates/tenferro-xla/Cargo.toml
```

Expected: `sha2 -> tenferro-xla`; no dependency edit.

- [ ] **Step 6: Commit the T4/T5 review unit**

```bash
git add crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.rs \
  crates/tenferro-tensor/tests/ui/storage/fail/static_rank_mismatch.stderr \
  crates/tenferro-tensor/src/storage/group.rs \
  crates/tenferro-tensor/src/storage/root.rs
git commit -m "test(storage): strengthen static-rank and unsafe invariants"
```

---

### Task 3: Correct candidate refresh and evidence-only diff validation

**Files:**
- Modify: `scripts/check-storage-contract-freeze.py`
- Modify: `scripts/check-storage-static-rank-codegen.py`
- Modify: `scripts/verify-storage-traversal-performance.py`
- Create: `scripts/ci/tests/test_storage_evidence_contracts.py`

**Interfaces:**
- Produces:
  - `check-storage-contract-freeze.py --report PATH [--refresh]`
  - `check-storage-static-rank-codegen.py --report PATH [--refresh]`
  - `verify-storage-traversal-performance.py ... --report PATH [--refresh]`
  - closed `EVIDENCE_ALLOWLIST` validation for `candidate..HEAD`.

- [ ] **Step 1: Create the evidence-checker test loader and failing allowlist tests**

Create `scripts/ci/tests/test_storage_evidence_contracts.py` with this loader:

```python
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Add these tests:

```python
class FreezeEvidenceTests(unittest.TestCase):
    def test_only_closed_evidence_paths_are_accepted(self) -> None:
        module = load_script("check-storage-contract-freeze.py")
        module.validate_evidence_paths({"docs/testing/storage-hardware-matrix.md"})
        with self.assertRaisesRegex(module.CheckError, "non-evidence path"):
            module.validate_evidence_paths({"crates/tenferro-tensor/src/types.rs"})

    def test_refresh_ignores_saved_reports(self) -> None:
        saved = {"candidate_commit": "a" * 40, "status": "pass", "result": "pass"}
        for script in (
            "check-storage-contract-freeze.py",
            "check-storage-static-rank-codegen.py",
            "verify-storage-traversal-performance.py",
        ):
            with self.subTest(script=script):
                module = load_script(script)
                self.assertIsNone(
                    module.select_existing_record(saved, refresh=True)
                )
```

Use the same `select_existing_record(record, *, refresh)` helper name in all three scripts.

- [ ] **Step 2: Run the focused tests and confirm the interfaces do not exist**

```bash
python3 -m unittest \
  scripts.ci.tests.test_storage_evidence_contracts.FreezeEvidenceTests -v
```

Expected: FAIL with missing `validate_evidence_paths` or `select_previous_record`.

- [ ] **Step 3: Implement the closed evidence allowlist and freeze refresh**

Use this exact path set in `check-storage-contract-freeze.py`:

```python
EVIDENCE_ALLOWLIST = frozenset(
    {
        "docs/design/storage-contract-freeze.md",
        "docs/testing/storage-hardware-matrix.md",
        "docs/testing/storage-static-rank-codegen.md",
        "docs/testing/storage-traversal-performance.md",
        "docs/worklogs/storage-documentation-source-blind-audit.md",
        "docs/worklogs/storage-redesign-closure.md",
        "docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md",
    }
)
```

Implement `validate_evidence_paths(paths: set[str]) -> None`, rejecting the first sorted path outside the set. In ordinary mode, require the frozen candidate to be an ancestor of HEAD and validate `git diff --name-only candidate..HEAD`. In `--refresh` mode, require a clean tracked tree, use `git rev-parse HEAD` as C, ignore the stale record, and rewrite the report.

Do not hash files or inspect untracked build output.

- [ ] **Step 4: Add `--refresh` to static-rank and traversal producers**

Refactor each existing-report branch into a small helper that returns no saved report when `refresh=True`; otherwise preserve current idempotent validation. Add `parser.add_argument("--refresh", action="store_true")` and run the existing producer when refresh is selected.

Both refreshed reports must continue to use `git rev-parse HEAD` as `candidate_commit`.

- [ ] **Step 5: Run checker unit tests**

```bash
python3 -m unittest scripts.ci.tests.test_storage_evidence_contracts -v
python3 scripts/test-storage-ownership-contracts-v2.py
```

Expected: PASS; canonical ledger argv remains unchanged because `--refresh` is evidence-generation-only.

- [ ] **Step 6: Commit candidate/evidence binding behavior**

```bash
git add scripts/check-storage-contract-freeze.py \
  scripts/check-storage-static-rank-codegen.py \
  scripts/verify-storage-traversal-performance.py \
  scripts/ci/tests/test_storage_evidence_contracts.py
git commit -m "fix(storage): enforce candidate evidence boundaries"
```

---

### Task 4: Add minimal multi-host hardware capture and merge

**Files:**
- Modify: `scripts/check-storage-hardware-matrix.py`
- Modify: `scripts/ci/tests/test_storage_evidence_contracts.py`

**Interfaces:**
- Consumes: existing lane definitions and commands.
- Produces:
  - partial capture from clean HEAD with `"complete": false`;
  - repeatable `--merge-report PATH`;
  - final required report with `"complete": true` and all five lanes passing.

- [ ] **Step 1: Add failing pure merge tests**

Add `HardwareMatrixTests` with compact lane fixtures:

```python
class HardwareMatrixTests(unittest.TestCase):
    candidate = "a" * 40

    def partial(self, names: tuple[str, ...], *, candidate: str | None = None) -> dict:
        return {
            "schema": "tenferro.storage-hardware-matrix.v1",
            "candidate_commit": candidate or self.candidate,
            "complete": False,
            "lanes": [
                {
                    "lane": name,
                    "status": "pass",
                    "command": f"run-{name}",
                    "environment": "test-host",
                    "device_facts": f"test-{name}",
                    "test_count": 1,
                    "passed": 1,
                    "failed": 0,
                    "ignored": 0,
                    "evidence": f"tests/{name}.rs",
                    "skip_reason": None,
                }
                for name in names
            ],
        }

    def test_merge_accepts_one_candidate_and_all_required_lanes(self) -> None:
        module = load_script("check-storage-hardware-matrix.py")
        merged = module.merge_records(
            self.candidate,
            [self.partial(("cpu", "cuda2", "cuda-ad")), self.partial(("webgpu", "metal"))],
        )
        self.assertTrue(merged["complete"])
        self.assertEqual(merged["status"], "pass")
        self.assertEqual([lane["lane"] for lane in merged["lanes"]], list(module.REQUIRED))

    def test_merge_rejects_mismatch_duplicate_missing_and_skip(self) -> None:
        module = load_script("check-storage-hardware-matrix.py")
        mismatch = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("webgpu", "metal"), candidate="b" * 40),
        ]
        duplicate = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("cpu", "webgpu", "metal")),
        ]
        missing = [self.partial(("cpu", "cuda2", "cuda-ad", "webgpu"))]
        skipped = [
            self.partial(("cpu", "cuda2", "cuda-ad")),
            self.partial(("webgpu", "metal")),
        ]
        skipped[1]["lanes"][1]["status"] = "skip"
        skipped[1]["lanes"][1]["test_count"] = 0
        for name, records in (
            ("candidate", mismatch),
            ("duplicate", duplicate),
            ("missing", missing),
            ("skip", skipped),
        ):
            with self.subTest(name=name):
                with self.assertRaises(module.CheckError):
                    module.merge_records(self.candidate, records)
```

Each passing fixture has a positive count and per-lane environment/device facts; no broader hardware attestation is tested.

- [ ] **Step 2: Run tests and verify merge support is absent**

```bash
python3 -m unittest \
  scripts.ci.tests.test_storage_evidence_contracts.HardwareMatrixTests -v
```

Expected: FAIL with missing `merge_records`.

- [ ] **Step 3: Implement partial capture semantics**

For execution mode:

- derive candidate from clean `git rev-parse HEAD`, not the stale freeze report;
- accept an absolute `/tmp/...` output for incomplete partial reports;
- record `complete: false`, selected lanes, and per-lane host facts;
- preserve ordinary `skip` classification for unavailable development lanes;
- obtain the macOS CPU model with `sysctl -n machdep.cpu.brand_string` when `/proc/cpuinfo` is absent.

Do not add remote upload, digest, or attestation behavior.

- [ ] **Step 4: Implement final merge semantics**

Add repeatable:

```python
parser.add_argument("--merge-report", action="append", default=[], type=Path)
```

In merge mode, read C from `storage-contract-freeze.md`, load all partial fenced JSON records, call `merge_records`, and overwrite only the requested final report. Reject candidate mismatch, duplicate/missing required lanes, non-pass status, and non-positive test count. Sort lanes by `REQUIRED`.

Ordinary validation of an existing final report must apply the same completeness checks rather than accepting `structured-skip`.

- [ ] **Step 5: Run unit and existing script-contract tests**

```bash
python3 -m unittest scripts.ci.tests.test_storage_evidence_contracts -v
python3 scripts/test-storage-ownership-contracts-v2.py
```

Expected: PASS.

- [ ] **Step 6: Commit multi-host matrix support**

```bash
git add scripts/check-storage-hardware-matrix.py \
  scripts/ci/tests/test_storage_evidence_contracts.py
git commit -m "fix(storage): require merged real hardware evidence"
```

---

### Task 5: Add bounded closure reproduction

**Files:**
- Modify: `scripts/check-storage-redesign-closure.py`
- Modify: `scripts/ci/tests/test_storage_evidence_contracts.py`
- Modify: `docs/design/storage-ownership-contracts.md`

**Interfaces:**
- Produces:
  - ordinary recorded-evidence mode: `--report PATH`;
  - final mode: `--report PATH --reproduce --receipt PATH`;
  - fixed reproduction results stored in the closure JSON record.

- [ ] **Step 1: Add failing closure-mode tests**

Add `ClosureReproductionTests` covering the bounded set, delegated receipt failure, and command failure:

```python
class ClosureReproductionTests(unittest.TestCase):
    def test_reproduction_command_set_is_bounded(self) -> None:
        module = load_script("check-storage-redesign-closure.py")
        self.assertEqual(
            [item[0] for item in module.REPRODUCE_COMMANDS],
            [
                "p10-api-normalization",
                "p4-traversal-resolution-counts",
                "p3-static-rank-preservation",
                "p3-host-owner",
                None,
                None,
            ],
        )

    def test_receipt_checker_failure_stops_before_execution(self) -> None:
        module = load_script("check-storage-redesign-closure.py")
        ran = False

        def runner(argv: tuple[str, ...]) -> int:
            nonlocal ran
            ran = True
            return 0

        with self.assertRaisesRegex(module.CheckError, "receipt checker"):
            module.run_reproduction(
                Path("receipt.json"),
                receipt_validator=lambda _: 1,
                runner=runner,
            )
        self.assertFalse(ran)

    def test_nonzero_reproduction_fails(self) -> None:
        module = load_script("check-storage-redesign-closure.py")

        def runner(argv: tuple[str, ...]) -> int:
            return 1 if argv[0] == "python3" else 0

        with self.assertRaisesRegex(module.CheckError, "exit code 1"):
            module.run_reproduction(
                Path("receipt.json"),
                receipt_validator=lambda _: 0,
                runner=runner,
            )
```

Keep default validation on the existing recorded-report path; only the explicit `args.reproduce` branch calls `run_reproduction`.

- [ ] **Step 2: Run the tests and confirm reproduction support is absent**

```bash
python3 -m unittest \
  scripts.ci.tests.test_storage_evidence_contracts.ClosureReproductionTests -v
```

Expected: FAIL with missing `REPRODUCE_COMMANDS` or validation function.

- [ ] **Step 3: Define the fixed command list**

Use exact obligation IDs and argv:

```python
REPRODUCE_COMMANDS = (
    ("p10-api-normalization", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_public_api")),
    ("p4-traversal-resolution-counts", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_traversal_resolution")),
    ("p3-static-rank-preservation", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_static_rank")),
    ("p3-host-owner", ("cargo", "test", "-p", "tenferro-tensor", "--test", "storage_compile_contract")),
    (None, ("cargo", "test", "-p", "tenferro-runtime", "scoped_immediate_provider_returns_borrowed_output")),
    (None, ("python3", "scripts/ci/run_profile.py", "coverage")),
)
```

Do not expose caller-supplied command selection.

- [ ] **Step 4: Implement receipt validation and reproduction**

When `--reproduce` is selected:

1. require `--receipt`;
2. invoke the existing ownership checker on that receipt and stop on nonzero status;
3. do not reimplement receipt parsing or validation;
4. run the six fixed argv arrays with `shell=False`;
5. collect exact argv and exit status in memory;
6. write a passing report only after all six return zero.

A failing command prints its command and exit code, returns 1, and leaves the prior report unchanged.

- [ ] **Step 5: Preserve cheap default behavior and tighten final closure**

Default mode may generate or validate the recorded-evidence closure without running commands. It must require the final hardware matrix to be complete and `pass`; `structured-skip` is no longer a closure success. If a reproduced closure already exists, default validation preserves and validates its reproduction records rather than stripping them.

Record Issue #1617/Fable5 as the audit source in the generated closure record; do not attempt reviewer authentication.

- [ ] **Step 6: Document CI mode ownership**

In `docs/design/storage-ownership-contracts.md`, state:

- ordinary `ci-config` runs the canonical recorded-evidence closure command and receipt-only ownership checker;
- final P13 closure runs `--reproduce --receipt ...`;
- coverage is the existing `run_profile.py coverage` command and thresholds;
- no digest or attestation is part of either mode.

- [ ] **Step 7: Verify tests and canonical ledger compatibility**

```bash
python3 -m unittest scripts.ci.tests.test_storage_evidence_contracts -v
python3 scripts/test-storage-ownership-contracts-v2.py
python3 scripts/check-storage-ownership-contracts.py
```

Expected: PASS; the manifest's canonical default closure argv remains unchanged.

- [ ] **Step 8: Commit closure reproduction**

```bash
git add scripts/check-storage-redesign-closure.py \
  scripts/ci/tests/test_storage_evidence_contracts.py \
  docs/design/storage-ownership-contracts.md
git commit -m "fix(storage): add bounded closure reproduction"
```

---

### Task 6: Correct durable P11/P13 intent and freeze product candidate C

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-p11-frozen-hardware-matrix-design.md`
- Modify: `docs/superpowers/specs/2026-08-04-p13-freeze-closure-design.md`
- Create: `docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md`

**Interfaces:**
- Produces: the final product/checker commit C and the exact evidence allowlist contract.

- [ ] **Step 1: Update P11 and P13 contracts**

Make these points explicit:

- C contains implementation/checkers/tests/CI/public docs/durable designs;
- generated freeze/performance/codegen/hardware/doc-audit/closure reports are allowlisted descendants because a report cannot contain its own commit hash;
- all report `candidate_commit` values equal C;
- ownership receipts bind to the clean evidence HEAD on which they run;
- partial hardware captures may skip, but final required merge and closure require every lane to pass;
- product/checker/test/CI/design changes after C create a new candidate.

Remove the contradictory statement that benchmark/codegen reports naming C must already be contained in C.

- [ ] **Step 2: Create the remediation classification work log**

Record:

- T0–T5 classification and exact source evidence;
- `sha2` retained with the `cargo tree` and source-import evidence;
- selected one-PR candidate/evidence model;
- no checksums/attestation/generic runner;
- commands completed so far;
- Apple and CUDA evidence still pending at this point.

Use concrete pending-state prose such as `Hardware evidence is collected after candidate C is selected`; do not use generic deferred-work markers.

- [ ] **Step 3: Run product/checker verification**

```bash
python3 scripts/ci/run_profile.py fmt
python3 scripts/ci/run_profile.py clippy
python3 -m unittest discover -s scripts/ci/tests -v
python3 scripts/test-storage-ownership-contracts-v2.py
cargo test -p tenferro-tensor --test storage_compile_contract
cargo test -p tenferro-tensor --test storage_static_rank
cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin
git diff --check
git status --short
```

Expected: every command PASS; only the intended durable-doc/worklog edits remain before commit.

- [ ] **Step 4: Commit durable intent, then freeze C externally**

```bash
git add docs/superpowers/specs/2026-08-04-p11-frozen-hardware-matrix-design.md \
  docs/superpowers/specs/2026-08-04-p13-freeze-closure-design.md \
  docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md
git commit -m "docs(storage): correct final evidence model"
git rev-parse HEAD | tee /tmp/issue-1617-candidate.txt
test -z "$(git status --porcelain)"
```

The printed 40-hex commit is C. Do not amend or rebase commits at or before C after hardware collection begins.

- [ ] **Step 5: Push C so the Apple host can check out the exact commit**

```bash
git push -u origin codex/issue-1617-remediation-plan
```

Do not open the PR yet.

---

### Task 7: Regenerate non-hardware evidence against C

**Files:**
- Modify generated evidence paths listed in the File Map.

**Interfaces:**
- Consumes: C from `/tmp/issue-1617-candidate.txt`.
- Produces: first evidence-only commit with every non-hardware report bound to C.

- [ ] **Step 1: Assert the branch still points at clean C**

```bash
C="$(cat /tmp/issue-1617-candidate.txt)"
test "$(git rev-parse HEAD)" = "$C"
test -z "$(git status --porcelain)"
```

- [ ] **Step 2: Refresh freeze, codegen, and traversal reports**

```bash
python3 scripts/check-storage-contract-freeze.py \
  --report docs/design/storage-contract-freeze.md --refresh
python3 scripts/check-storage-static-rank-codegen.py \
  --report docs/testing/storage-static-rank-codegen.md --refresh
python3 scripts/verify-storage-traversal-performance.py \
  --baseline-obligation p1-element-access-baseline \
  --baseline-report docs/testing/storage-element-access-baseline.json \
  --report docs/testing/storage-traversal-performance.md --refresh
```

Expected: freeze/static-rank status `pass`; traversal result `pass`, not `inconclusive`.

- [ ] **Step 3: Rebuild and recheck rendered documentation**

```bash
bash scripts/build_docs_site.sh
python3 scripts/check-storage-docs.py --include-rendered
python3 scripts/check-storage-element-access-docs.py docs/guides/views-and-slicing.md
python3 scripts/check-docs-site.py
cargo test -p tenferro-tutorial-code --release \
  tutorial_binaries_run_successfully -- --exact
```

Give only rendered outputs to an independent source-blind reviewer. Update `storage-documentation-source-blind-audit.md` with C, exact commands, findings, and zero/explicitly classified gaps.

- [ ] **Step 4: Verify all regenerated candidate fields equal C**

```bash
python3 - <<'PY'
import json, re, subprocess
from pathlib import Path
c = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
paths = [
    Path("docs/design/storage-contract-freeze.md"),
    Path("docs/testing/storage-static-rank-codegen.md"),
    Path("docs/testing/storage-traversal-performance.md"),
]
for path in paths:
    match = re.search(r"```json\s*(\{.*?\})\s*```", path.read_text(), re.S)
    assert match, path
    assert json.loads(match.group(1))["candidate_commit"] == c, path
print(f"non-hardware-candidate-ok: {c}")
PY
```

- [ ] **Step 5: Commit non-hardware evidence only**

```bash
git add docs/design/storage-contract-freeze.md \
  docs/testing/storage-static-rank-codegen.md \
  docs/testing/storage-traversal-performance.md \
  docs/worklogs/storage-documentation-source-blind-audit.md
git diff --cached --name-only
git commit -m "test(storage): regenerate candidate-bound evidence"
```

Confirm the staged path list contains no Rust/Python/checker/CI/design file.

---

### Task 8: Capture Linux/CUDA and Apple evidence and merge the final matrix

**Files:**
- Modify: `docs/testing/storage-hardware-matrix.md`

**Interfaces:**
- Consumes: exact C and two temporary partial reports.
- Produces: one complete required-mode matrix bound to C.

- [ ] **Step 1: Create a detached Linux/CUDA worktree at C**

```bash
C="$(cat /tmp/issue-1617-candidate.txt)"
git worktree add --detach /tmp/tenferro-1617-linux-candidate "$C"
cd /tmp/tenferro-1617-linux-candidate
python3 scripts/check-storage-hardware-matrix.py \
  --report /tmp/storage-hardware-linux.md \
  --lanes cpu,cuda2,cuda-ad
```

Expected: all three captured lanes `pass` with positive counts. If CUDA hardware is supplied through the trusted RunPod path, run this exact command inside that checked-out C workspace and return the report artifact.

- [ ] **Step 2: Run Apple WebGPU/Metal at exact C**

On the Apple M5 Max:

```bash
git fetch origin codex/issue-1617-remediation-plan
C="$(git rev-parse origin/codex/issue-1617-remediation-plan^{commit})"
git checkout --detach "$C"
test -z "$(git status --porcelain)"
python3 scripts/check-storage-hardware-matrix.py \
  --report /tmp/storage-hardware-apple.md \
  --lanes webgpu,metal
```

Before running, compare the printed `C` to `/tmp/issue-1617-candidate.txt` from Task 6. Expected: WebGPU and Metal both `pass`; Metal executes all four `apple_context` tests, not zero tests.

Return `/tmp/storage-hardware-apple.md` to the evidence-worktree host as the hardware-run artifact. Do not proceed until both partial reports are present there.

- [ ] **Step 3: Merge on the evidence branch**

Return to the remediation worktree and run:

```bash
C="$(cat /tmp/issue-1617-candidate.txt)"
python3 scripts/check-storage-hardware-matrix.py \
  --report docs/testing/storage-hardware-matrix.md \
  --merge-report /tmp/storage-hardware-linux.md \
  --merge-report /tmp/storage-hardware-apple.md \
  --required-mode
python3 scripts/check-storage-hardware-matrix.py \
  --report docs/testing/storage-hardware-matrix.md
```

Expected: `storage-hardware-matrix-pass`; candidate C; five unique passing lanes; no skip.

- [ ] **Step 4: Commit only the merged matrix**

```bash
git add docs/testing/storage-hardware-matrix.md
git diff --cached --name-only
git commit -m "test(storage): record Linux CUDA and Apple evidence"
git worktree remove /tmp/tenferro-1617-linux-candidate
```

Expected staged path: only `docs/testing/storage-hardware-matrix.md`.

---

### Task 9: Produce recorded closure, reproduced closure, and final receipt

**Files:**
- Modify: `docs/worklogs/storage-redesign-closure.md`
- Modify: `docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md`

**Interfaces:**
- Produces: clean pre-reproduction head, pre-reproduction receipt, final evidence HEAD E, and final receipt bound to E.

- [ ] **Step 1: Generate the cheap recorded-evidence closure**

```bash
python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md
```

Expected: candidate C, hardware status `pass`, no unresolved Critical/Important finding, and recorded-evidence mode.

Update the remediation work log with non-hardware and hardware results, exact host facts, commands, and the `sha2` stale disposition.

- [ ] **Step 2: Commit a clean pre-reproduction evidence head**

```bash
git add docs/worklogs/storage-redesign-closure.md \
  docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md
git commit -m "docs(storage): assemble recorded closure evidence"
test -z "$(git status --porcelain)"
git rev-parse HEAD | tee /tmp/issue-1617-pre-reproduction-head.txt
```

- [ ] **Step 3: Generate and validate the pre-reproduction receipt**

```bash
python3 scripts/run-storage-ownership-contracts.py \
  --base-commit 402c962c61543f1477e3e3e0ade2c293b9d05ad4 \
  --receipt-out /tmp/issue-1617-pre-reproduction-receipt.json
python3 scripts/check-storage-ownership-contracts.py \
  --base-commit 402c962c61543f1477e3e3e0ade2c293b9d05ad4 \
  --receipt /tmp/issue-1617-pre-reproduction-receipt.json \
  --summary-json
```

Expected: 31 successful executions and `{"terminal": true}`.

- [ ] **Step 4: Obtain independent read-only integration review**

Give the reviewer Issue #1617, C, the pre-reproduction evidence head, the changed checker/test paths, and all generated reports. Require classifications `Critical`, `Important`, or `Minor`, candidate identity, commands inspected, and explicit hardware limitations. Save the result outside the repository at `/tmp/issue-1617-independent-audit.md`.

Do not continue if any Critical/Important finding remains. A product/checker correction requires returning to Task 6 with a new C.

- [ ] **Step 5: Run bounded reproduction**

```bash
python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md \
  --reproduce \
  --receipt /tmp/issue-1617-pre-reproduction-receipt.json
```

Expected: all six fixed commands pass, including coverage; the closure report records their exact argv and zero status.

Add the independent audit disposition to the remediation work log without copying a raw model transcript.

- [ ] **Step 6: Commit final evidence HEAD E**

```bash
git add docs/worklogs/storage-redesign-closure.md \
  docs/worklogs/2026-08-05-issue-1617-closure-hygiene-remediation.md
git commit -m "test(storage): record reproduced independent closure"
git rev-parse HEAD | tee /tmp/issue-1617-evidence-head.txt
test -z "$(git status --porcelain)"
```

- [ ] **Step 7: Generate and validate the final receipt bound to E**

```bash
python3 scripts/run-storage-ownership-contracts.py \
  --base-commit 402c962c61543f1477e3e3e0ade2c293b9d05ad4 \
  --receipt-out /tmp/issue-1617-final-receipt.json
python3 scripts/check-storage-ownership-contracts.py \
  --base-commit 402c962c61543f1477e3e3e0ade2c293b9d05ad4 \
  --receipt /tmp/issue-1617-final-receipt.json \
  --summary-json
python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md
```

Expected: receipt candidate equals E, 31/31 exit zero, terminal true, and default closure validation does not modify the report.

- [ ] **Step 8: Audit report candidate consistency and evidence-only diff**

```bash
C="$(cat /tmp/issue-1617-candidate.txt)"
E="$(cat /tmp/issue-1617-evidence-head.txt)"
python3 scripts/check-storage-contract-freeze.py \
  --report docs/design/storage-contract-freeze.md
git diff --name-only "$C..$E"
git diff --check "origin/main...$E"
test -z "$(git status --porcelain)"
```

Expected: freeze checker accepts only allowlisted evidence paths after C; worktree clean.

---

### Task 10: Run PR gates, merge, and correct Issues #1617/#1555

**Files:**
- No product edits expected. Any edit returns to the owning earlier task and creates a new C if it touches product/checker/test/CI/design.

**Interfaces:**
- Produces: merged remediation PR and corrected umbrella evidence record.

- [ ] **Step 1: Run final local gates**

```bash
python3 scripts/ci/run_profile.py fmt
python3 scripts/ci/run_profile.py clippy
python3 -m unittest discover -s scripts/ci/tests -v
python3 scripts/test-storage-ownership-contracts-v2.py
bash scripts/check-pr-fast.sh \
  --base origin/main \
  --no-fetch \
  --coverage-reviewed \
  --doc-snippets \
  --test 'cargo test -p tenferro-tensor --test storage_compile_contract'
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/issue-1617-repository-rules-review.json
git diff --check origin/main...HEAD
test -z "$(git status --porcelain)"
```

Expected: all PASS. Review warnings must be fixed or explicitly dispositioned before PR creation.

- [ ] **Step 2: Push final E and create one remediation PR**

```bash
git push origin codex/issue-1617-remediation-plan
```

Create the PR with `.github/pull_request_template.md`. Include:

- T0–T5 classification ledger;
- C and E full hashes;
- Metal/WebGPU Apple M5 Max results;
- Linux/CUDA results;
- reproduced closure and final receipt paths/results;
- `sha2` stale finding evidence;
- explicit statement that no digest/attestation/generic runner was added;
- `Refs #1617, #1555` rather than closing #1555 before merge.

- [ ] **Step 3: Wait for and inspect every required CI check**

Require rustfmt, Clippy, macOS-gated cross-target check, workspace tests, coverage, docs, CI configuration, repository-rules review, and GPU gate to pass on E. Inspect complete logs for any skipped required job; a green no-op does not count when policy says the job is required.

- [ ] **Step 4: Merge without squash**

After approvals and all required checks pass, merge with a merge commit. Record the PR number and merge commit.

- [ ] **Step 5: Correct Issue #1555 and close Issue #1617**

Comment on #1555 with:

- corrected product candidate C;
- final evidence HEAD E and merge commit;
- final report paths;
- Metal/WebGPU real Apple host result;
- final receipt 31/31 and reproduced closure result;
- explanation that the old `385a04db`/`b3e2bb6c` bindings are superseded.

Comment on #1617 with each T0–T5 disposition. State that `sha2` remained because `tenferro-xla` uses it. Close #1617 only after the remediation PR is merged and the issue comments contain the final hashes.

- [ ] **Step 6: Final post-merge verification**

Fetch `origin/main`, verify the merge commit contains E, rerun the cheap freeze/closure/ownership receipt validations on a clean checkout, and report any hardware lane as historical candidate-bound evidence rather than silently rerunning it on unavailable hardware.
