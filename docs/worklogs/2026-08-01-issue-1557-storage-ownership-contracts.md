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

Only these rows are active:

- `p1-ledger`
- `p1-contract-document`
- `p1-api-parity`

`p0-control-plane` is deferred to P0, `p1-element-access-baseline` is deferred
to P1 until the real measured report and verifier exist, and `p2-root-claims`
is deferred to P2. The remaining future rows stay deferred. Fake active
artifacts were explicitly rejected: creating a placeholder would convert
unfinished scientific evidence into a false lifecycle proof and would violate
the single tagged-obligation authority.

The baseline command and the P10 traversal command use commit/path provenance.
The P10 command no longer accepts or names a saved baseline receipt. This task
does not benchmark or capture the P1 baseline. Prepared-access `CheckedLayout`,
contiguous-path, and prevalidated traversal requirements remain unchanged.

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
- Manifest audit: exactly 3 active and 28 deferred obligations; no baseline-receipt command arguments.
- `git diff --check`: passed.
- `bash scripts/check-pr-fast.sh --no-fetch --base e51551755f1a42324e966ceda49e11e4933ab800 --coverage-reviewed --skip-doc-snippets --test 'cargo test -p tenferro-tensor --test storage_api_parity'`: passed, including workspace/extension fmt and clippy checks.

The implementation commit ID is reported in the handoff rather than inserted
here: adding its own ID to this worklog would require an unnecessary follow-up
amendment.

## Residual risk and next action

The P1 baseline, P0 control-plane artifact, and P2 root-claims artifact remain
genuine future work. The runner assumes the trusted local command environment;
it does not attest to a child process or defend against a runner that forges its
own log. Full tracked-tree cleanliness can reject a candidate with unrelated
tracked edits, which is intentional because tracked source and build metadata
are transitive inputs to cargo commands.

The post-bootstrap CI follow-up gives the `ci-config` checkout full history and
passes exactly one event-derived base to its existing production checker:
`pull_request.base.sha` for pull requests or `github.event.before` for pushes.
There is no second checker execution, v1 detection, fallback, or compatibility
exception. Local `ci-config` runs remain structural when no base is supplied.

This enforcement branch cannot be published until #1593 merges because its
first hosted comparison must have a schema-v2 base. Once sequenced after that
merge, the wiring closes the P1 CI-enforcement residual. The P1 baseline, P0
control-plane artifact, and P2 root-claims artifact remain genuine future
work.
