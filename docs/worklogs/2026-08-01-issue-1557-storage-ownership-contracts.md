# Work log: #1557 storage ownership contracts

Date: 2026-08-02
Scope: P1 schema-v2 ledger migration only
Design: `docs/design/storage-ownership-contracts.md`

## Decision

The Phase-1 ledger is a trusted-runner execution log. Tenferro is scientific-
computing software, not a security product or trust boundary. Repository
source, maintainers, local build tools, and the CI runner are trusted. The
implementation therefore validates reachable correctness mistakes without
pretending that a trusted runner's log is a security attestation.

The receipt is intentionally small:

```json
{
  "schema": "tenferro.storage-ownership-receipt.v1",
  "candidate_commit": "git rev-parse HEAD",
  "base_commit": null,
  "executions": [
    {
      "obligation_id": "p1-ledger",
      "argv": ["python3", "scripts/check-storage-ownership-contracts.py"],
      "cwd": ".",
      "artifact_path": "scripts/storage-ownership-contracts.toml",
      "exit_code": 0
    }
  ]
}
```

Executions are sorted by obligation ID. Command and artifact IDs are derived
from the candidate manifest and are not repeated. Git object IDs are opaque
strings produced by Git. The candidate is `HEAD`; an optional base revision is
canonicalized with `git rev-parse --verify <revision>^{commit}` before ancestor
and deferred-to-active promotion comparison. The receipt stores the resulting
opaque Git object ID, never a branch or revision alias.

For tracked manifests, verifier scripts, command targets, and artifacts, the
candidate commit and repository-relative path identify the bytes. Before
execution and receipt acceptance, the tools compare the full tracked tree with
candidate `HEAD` using one Git cleanliness check. Untracked or ignored receipt
output, build targets, logs, and unrelated user files are allowed. No global
empty-worktree requirement, detached worktree, inode identity check, or
post-receipt retarget protocol is used. Repository-relative paths and cwd are
still confined before execution, including rejecting a symlink that resolves
outside the repository.

No content checksum is part of this tracked-artifact contract. A checksum would
be justified only at a concrete untracked or cross-system boundary; this P1
implementation introduces no such boundary.

## State correction

Before the baseline measurement, only these rows were active:

- `p1-ledger`
- `p1-contract-document`
- `p1-api-parity`

`p0-control-plane` is deferred to P0, `p1-element-access-baseline` was then
deferred to P1 until the real measured report and verifier existed, and
`p2-root-claims` is deferred to P2. The remaining future rows stay deferred.
Fake active artifacts were explicitly rejected: creating a placeholder would
convert unfinished scientific evidence into a false lifecycle proof and would
violate the single tagged-obligation authority. The baseline row is now
promoted in place after the real report was measured and verified below.

The baseline command and the P10 traversal command use commit/path provenance.
The P10 command no longer accepts or names a saved baseline receipt. The
measured report uses the exact source commit and benchmark path described below.
Prepared-access `CheckedLayout`, contiguous-path, and prevalidated traversal
requirements remain unchanged.

## Implementation

- `scripts/check-storage-ownership-contracts.py` is a schema-v2-only checker.
  It validates one tagged obligation array, registry graph/cohort structure,
  direct incoming-unit prerequisite completeness, exact registry preservation
  across promotion, exact typed command allowlists, and artifact/command path
  confinement,
  deferred artifact absence, promotion identity, cohort atomicity, candidate
  commit binding, tracked-tree cleanliness, exit status, and derived terminal
  state.
- `scripts/run-storage-ownership-contracts.py` executes active argv vectors with
  `subprocess.run(..., shell=False)`, never executes deferred rows, propagates
  nonzero exit status as structured diagnostics, and writes only the minimal
  receipt fields above.
- `scripts/check-storage-design-docs.py` is a real content checker for the
  normative design artifact and its Phase-1 ledger markers.
- `scripts/test-check-storage-ownership-contracts.py` was deleted. The v1
  fixture is schema-only and is used solely to prove v1 rejection.
- The v2 suite derives active/deferred counts from the parsed manifest and
  tests reachable structural, promotion, path, runner, receipt, and exit-
  status behavior. It has no migration-event registry or historical totals.

## Verification record

The simplified test contract was first run against the preserved implementation
and was RED: 17 tests ran with six intended failures, including the old receipt
shape, stale baseline command, runner incompatibility, and retained v1 test
authority. The final suite derives its counts from the manifest and adds
reachable checks for canonical base aliases, full tracked-tree cleanliness,
relevant untracked inputs, and symlink confinement. The correctness follow-up
first reproduced both missing checks: P2 activation with a deferred P1
baseline was accepted, unit/gate/edge registry mutations were accepted, and a
cohort mutation reached only candidate-shape validation. Focused tests then
passed after deriving prerequisite state from edges and comparing the complete
base/candidate registry.

Final pre-commit evidence:

- `python3.12 -m py_compile scripts/check-storage-ownership-contracts.py scripts/run-storage-ownership-contracts.py scripts/check-storage-design-docs.py scripts/test-storage-ownership-contracts-v2.py scripts/ci/run_profile.py scripts/ci/tests/test_run_profile.py`: passed.
- Focused correctness RED: 2 test methods produced the expected 5 failures (`Ran 2 tests in 0.757s`, `FAILED (failures=5)`).
- Focused correctness GREEN: 2 test methods passed (`Ran 2 tests in 0.711s`, `OK`).
- `python3.12 scripts/test-storage-ownership-contracts-v2.py`: 22 tests, 0 failures (`Ran 22 tests in 4.619s`, `OK`).
- `python3.12 -m unittest scripts.ci.tests.test_run_profile -v`: 24 tests, 0 failures (`Ran 24 tests in 0.009s`, `OK`).
- `cargo test -p tenferro-tensor --test storage_api_parity`: 1 passed, 0 failed.
- Checker and runner `--contract-schema` probes, the design-document checker, and the production checker summary passed; the summary was `{"terminal": false}`.
- Pre-baseline manifest audit: exactly 3 active and 28 deferred obligations; no baseline-receipt command arguments.
- `git diff --check`: passed.
- `bash scripts/check-pr-fast.sh --no-fetch --base e51551755f1a42324e966ceda49e11e4933ab800 --coverage-reviewed --skip-doc-snippets --test 'cargo test -p tenferro-tensor --test storage_api_parity'`: passed, including workspace/extension fmt and clippy checks.

The implementation commit ID is reported in the handoff rather than inserted
here: adding its own ID to this worklog would require an unnecessary follow-up
amendment.

## P1 element-access baseline measurement

Date: 2026-08-02. This section records the measured artifact and the correction
that superseded the earlier source surface.

The interrupted pre-correction measurements are discarded. One run exposed
Criterion 0.5.1's sanitized group-directory names before any report was
accepted; the subsequent pre-Sol-correction run was also interrupted. Neither
run produced a retained report, and their `target/criterion` data was moved to
trash. No measurement from `014b08ca` (or an intermediate source revision) is
used.

The corrected first source-surface commit is
`da7b36e699f9f4731dec08de6a4e1ca93f20cd6f` (`da7b36e6`), rebuilt from merged
main `eab65236d6fff7ae28b5ec700b83a97b81a77740`. The benchmark, capture script,
and verifier were not modified after this commit. The report's canonical
provenance is this full measured commit plus
`crates/tenferro-tensor/benches/element_access.rs`.

The capture command was:

```text
python3 scripts/capture-storage-element-access-baseline.py \
  --root . --output docs/testing/storage-element-access-baseline.json
```

The script used these exact settings and command:

```text
cargo bench --locked -p tenferro-tensor --bench element_access -- \
  --warm-up-time 2 --measurement-time 5 --sample-size 100 --noplot
```

Cargo's optimized `bench` profile was used (`cargo bench` does not accept a
separate `--release` flag here). Criterion sample size was 100, warm-up was
2.0 s, measurement was 5.0 s, and the report unit is `ns`. Criterion 0.5.1
WallTime `estimates.json` values are already nanoseconds; the corrected capture
keeps them as-is. All recorded thread environment variables were explicitly
set to `1`: `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`RAYON_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS`. The report records actual
values for `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS`; both were the empty
string. No provider constant or redundant thread-count field is recorded.

Environment metadata:

| Field | Value |
|---|---|
| CPU | AMD EPYC 7713P 64-Core Processor; x86_64; 64 logical CPUs; affinity 0–63 |
| OS | Linux 6.8.0-101-generic |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)`; host `x86_64-unknown-linux-gnu` |
| cargo | `cargo 1.97.1 (c980f4866 2026-06-30)` |
| target | `x86_64-unknown-linux-gnu` |
| flags | `RUSTFLAGS=""`; `CARGO_ENCODED_RUSTFLAGS=""` |
| thread environment | all five recorded variables set to `"1"` |

The measured report contains these exact Criterion estimates, 95% confidence
intervals, and standard errors. Standard error is the report's uncertainty
statistic (and is finite and nonnegative); no separate machine-independent
variance threshold is inferred from it.

| Case | Estimate (ns) | 95% CI (ns) | Standard error (ns) |
|---|---:|---:|---:|
| `element_access/2d/col_major/direct_slice/4096` | 3446.968058 | 3417.173480–3480.647777 | 16.248749 |
| `element_access/2d/col_major/direct_slice_mut/4096` | 5878.583031 | 5711.830805–6038.007516 | 83.544049 |
| `element_access/2d/col_major/get/4096` | 19272.227701 | 19160.085818–19405.645514 | 63.055177 |
| `element_access/2d/col_major/get_unchecked/4096` | 15825.074363 | 15728.582453–15939.878010 | 54.339178 |
| `element_access/2d/col_major/get_mut/4096` | 21736.387091 | 21510.594557–21965.719710 | 115.800576 |
| `rank_fixed/2d/col_major/get2/4096` | 28555.716497 | 28341.129099–28800.892047 | 117.625192 |
| `rank_fixed/3d/col_major/get3/4096` | 32972.593116 | 32713.614835–33275.500962 | 143.860285 |
| `linear_iteration/col_major/as_slice_iter` | 54986.536871 | 54660.560300–55368.006011 | 180.725000 |
| `linear_iteration/col_major/dynamic_tensor_iter` | 54426.404043 | 54088.019914–54815.124458 | 186.696291 |
| `linear_iteration/col_major/tensor_iter` | 55222.942856 | 54800.867825–55709.797571 | 231.912550 |
| `linear_iteration/col_major/dynamic_tensor_iter_mut` | 78523.801851 | 76013.275215–80871.067165 | 1239.484005 |
| `strided_traversal/rectangular_transpose/logical_order_get/3840` | 14849.827750 | 14766.254679–14947.772143 | 46.367490 |

The benchmark surface preserves the contiguous/direct/checked/unchecked,
mutable, and fixed-rank cases. The strided case is a complete logical-order
traversal of a rectangular transpose, not a square random-get sample. The
recorded set includes fixed-rank 3D access, dynamic immutable iteration, and
dynamic mutable iteration. Mutable writes are observed through an aggregate of
the touched values rather than a single sentinel element.

`python3 scripts/verify-storage-element-access-baseline.py --root . --report
docs/testing/storage-element-access-baseline.json` passed. The row is promoted
by changing only its tagged state from deferred to active; artifact, command,
unit, gate, and IDs remain unchanged.

P10 may compare a candidate only when the relevant environment is compatible:
CPU architecture/model and affinity, OS/kernel class, rustc/Cargo/toolchain,
compilation target, optimized profile, thread environment, and provider/
placement where applicable. Otherwise the report is provenance but the result
is incompatible/inconclusive; no machine-independent threshold or cross-machine
threshold transfer is allowed.

## Residual risk and next action

The P0 control-plane artifact and P2 root-claims artifact remain genuine future
work. The P1 baseline is now measured, tracked, verified, and active. The runner
assumes the trusted local command environment;
it does not attest to a child process or defend against a runner that forges its
own log. Full tracked-tree cleanliness can reject a candidate with unrelated
tracked edits, which is intentional because tracked source and build metadata
are transitive inputs to cargo commands.

The post-bootstrap CI follow-up gives the `ci-config` checkout full history and
passes exactly one event-derived base to its existing production checker:
`pull_request.base.sha` for pull requests or `github.event.before` for pushes.
There is no second checker execution, v1 detection, fallback, or compatibility
exception. Local `ci-config` runs remain structural when no base is supplied.

The hosted checker requires a schema-v2 base for promotion comparisons. The
merged-main base used for this promotion is
`eab65236d6fff7ae28b5ec700b83a97b81a77740`; P0 and P2 remain deferred until
their own artifacts are real.

## Final provider-neutral prepared-access review

The Sol-medium closure review of candidate `25cba3207650fc472c76bb4e69bbb6f9ba856e6e`
found that the consolidated prepared-access sketch still placed host guards in
payloads also consumed by CUDA/WebGPU/Metal binding. The contract now keeps one
provider-neutral `PreparedRead`/`PreparedWrite` hierarchy with explicit
`Host` and `Device` variants. Only nested host payloads contain host guards and
expose contiguous slices or strided iterators. Device payloads retain the
checked capability/layout and opaque provider-prepared state, and G5 consumes
those exact payloads without constructing host-visible access.

The same review found one stale `CheckedInjectiveLayout` name. The contract now
names `CheckedInjectiveDescriptor` as the descriptor-level write proof and
`CheckedInjectiveStrided` as the mutable iterator's traversal proof. No new
runtime validation, identity protocol, registry, or compatibility mechanism
was introduced by either correction.
