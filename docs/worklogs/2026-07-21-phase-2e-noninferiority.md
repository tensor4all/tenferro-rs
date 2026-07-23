# Phase 2E non-inferiority evidence history

## Candidate `eba500ed`: abandoned run 0002

### Outcome

The protocol-v2 outer root for candidate
`eba500ed3297072b82e7d43869c6e54523aaee4d` is sealed as
`ABANDONED`. No allocation or timing measurement attempt started, so this
root is not performance evidence and does not classify the candidate.

The private `timing-builds` worker completed the direct-current-main release
build, then returned exit code 2 while checking the
common-lock-normalized baseline. The retained
`timing-build-failure.json` records the exact failing command:

```text
cargo metadata --locked --format-version 1
```

It ran in the isolated common-lock-normalized baseline and returned 101
because Cargo would have needed to update `Cargo.lock`. A separate diagnostic
copy resolved the lock offline and showed the sole required change: remove the
candidate-only `rayon` dev-dependency from the `tenferro-ad` package entry.
The common lock was generated from the candidate, while baseline
materialization still copied `crates/tenferro-ad/Cargo.toml` from the older
Phase 1 harness commit. The Phase 2E characterization harness had subsequently
added `rayon`, a benchmark stanza, and compiler-check configuration to that
manifest. A byte-identical common lock therefore could not be valid for both
source trees. This is a frozen harness-input defect, not a candidate
performance result.

### Immutable identities

- Candidate: `eba500ed3297072b82e7d43869c6e54523aaee4d`
- Candidate tree SHA-256:
  `decaea388115c2d4e81e594e090ac534c27c842ac1a6f590d935d500ba3f494a`
- Experiment identity:
  `f1929124e5f49d1ddb52c85be4e6a272a46ff711a1ede14a52c473c65cea929d`
- Campaign identity:
  `456caeadda20ad6177dca0c297017ad3f1e2462bd94d31ace6300187084a06ee`
- Reservation:
  `60fc0332d20768ce595dfc1cc796795b428014a85d7c7d2a4c832f8b72bacea7`
- Command contract:
  `d311e6a43f9ad8ce1a81b70b56975bcb3f7fc467aff53ef231862efc8b6a2d05`
- Context SHA-256:
  `f08cd859f9674d7889cfec0b0bbdeded9af721810d316fbeb2feddd91b772448`
- Root digest:
  `26b9daf48a12a5aa1a8ae54420cbd05935462edd95ac11685da1635e5695258c`
- Ledger digest:
  `1d7e737f0bae943d5b1f7553b376af97157c36d97939cd3605d57cb5807376a9`
- Failure artifact digest:
  `9e8d3bff0f4c1495091dc85d4696fe78395a3921b2c91701c46ebdf7eb254d0a`

The canonical root is
`docs/worklogs/artifacts/2026-07-21-phase-2e-eba500ed3297-run-0002`.
Its `abandoned-inventory.json` is the normative ownership inventory. The
index has `ACTIVE` followed by terminal `ABANDONED`; it deliberately has no
`current_evidence_root`.

### Host and command

The run began on 2026-07-23 in the Asia/Tokyo timezone with:

- 64 process-allowed CPUs;
- one-minute load 3.23 immediately before launch, normalized to 0.0505;
- no overlapping real `cargo` or `rustc` process; and
- 396 GiB free on the filesystem before launch.

The exact public command was:

```text
python3 scripts/run_phase2e.py start \
  --repository <candidate-worktree> \
  --candidate eba500ed3297072b82e7d43869c6e54523aaee4d \
  --evidence-root docs/worklogs/artifacts/2026-07-21-phase-2e-eba500ed3297-run-0002 \
  --index docs/worklogs/2026-07-21-phase-2e-index.json \
  --scratch-parent /tmp/phase2e-scratch.UicwpK
```

The ledger contains no attempt IDs; all measurement lanes remain `READY` or
`BLOCKED`. After verifying that no orchestrator, stage worker, benchmark
build, or compiler process remained, the root was sealed with
`record-index --abandoned --confirm-no-live-processes`, which returned
`PENDING_PRESERVATION`.

### Remote preservation

The sealed root and pending index were pushed in commit
`8537b9d710d453555e3963898aaab6f735f10d1a`. The permanent explanatory
report is
<https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-5056505728>.
The exact machine-validated preservation proof is
<https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-5056508586>.

`record-preserved` fetched the branch, reconstructed and validated the
committed root, index, and worklog blobs, fetched the exact proof comment, and
returned `PRESERVED`. Run 0002 remains negative evidence and does not become
`current_evidence_root`.

### Follow-up

1. Preserve this negative root without modifying its sealed inventory.
2. Add a regression test that uses real baseline/candidate manifest
   dependency differences and observes the common-lock validation failure.
3. Freeze the complete Phase 2E harness manifest snapshot instead of the
   older Phase 1 harness snapshot.
4. Because the harness and experiment identity change, freeze a new candidate
   and run the complete campaign in a fresh root only after this root reaches
   `PRESERVED`.

## Candidate `cd30b008`: abandoned run 0001

## Outcome

The first protocol-v2 outer root for candidate
`cd30b0082b86f968e21911db48d9b04bb0be820c` is preserved as
`ABANDONED`. No timing or allocation measurement attempt started, so this
root is not performance evidence and does not classify the candidate.

The private `timing-builds` worker returned exit code 2 while building the
direct-current-main `eager_dispatch_baseline` benchmark. The outer
orchestrator recorded the child and stopped before `probe-builds`. The build
stage did not persist a build manifest, captured stdout/stderr, or a typed
failure reason. Consequently the exact compiler failure cannot be recovered
from the sealed root. This observability gap must be diagnosed before a new
outer root is started; the failed build stage is not lane-retryable.

## Immutable identities

- Candidate: `cd30b0082b86f968e21911db48d9b04bb0be820c`
- Candidate tree SHA-256:
  `6f1c6dd3d7073289d1c3ccc74cc20ae4957961dec769e13e7337bfcd9b947daf`
- Current-main commit used by the protocol:
  `68855c2b65b5adc42dccca9bac04fd136a8f14c8`
- Experiment identity:
  `2e34bfcd173c5a5f181280d18b26d2c70db88c9559f030acc5b9b90fb99e0619`
- Campaign identity:
  `60cef344d5d86aee46fc6877b902445f4666fa87c09628a79d01979581043740`
- Reservation:
  `bcb7f0c44f8bf7b800ac113764939c49525c152a5ecb8fdc018c93f9dcb511e7`
- Command contract:
  `60f7942a144371e651de3bb8c53b42fae6b29dab3879156476e28a0a1addfef1`
- Root digest:
  `ef8e332588db3d662dad6882276bb68bb9768d663ccff1a1a3f5be9b4fa8cb1b`
- Ledger digest:
  `f4cd38c0d18f058c9a149bdabbf8d6dee7ee9fd78a6ece62283fae990f0e1b2f`

The canonical root is
`docs/worklogs/artifacts/2026-07-21-phase-2e-cd30b0082b86-run-0001`.
Its `abandoned-inventory.json` is the normative ownership inventory.
The index has `ACTIVE` followed by terminal `ABANDONED`; it deliberately has
no `current_evidence_root`.

## Host and command

The run began on 2026-07-23 in the Asia/Tokyo timezone with:

- 64 process-allowed CPUs, affinity `0-63`;
- one-minute load 8.53 immediately before launch, normalized to 0.133;
- no overlapping real `cargo` or `rustc` process;
- 337 GiB free on the filesystem before launch;
- Rust/Cargo 1.96.0 and Python 3.12.11.

The exact public command was:

```text
python3 scripts/run_phase2e.py start \
  --repository <candidate-worktree> \
  --candidate cd30b0082b86f968e21911db48d9b04bb0be820c \
  --evidence-root docs/worklogs/artifacts/2026-07-21-phase-2e-cd30b0082b86-run-0001 \
  --index docs/worklogs/2026-07-21-phase-2e-index.json \
  --scratch-parent /tmp/phase2e-scratch.umT38m
```

The worker ran for approximately four minutes. Its process journal records a
normal reap with exit code 2 and no termination signal. The ledger contains no
attempt IDs; all measurement lanes remain `READY` or `BLOCKED`.

After verifying that no orchestrator, stage worker, benchmark build, or rustc
process remained, the root was sealed with:

```text
python3 scripts/run_phase2e.py record-index \
  --evidence-root docs/worklogs/artifacts/2026-07-21-phase-2e-cd30b0082b86-run-0001 \
  --index docs/worklogs/2026-07-21-phase-2e-index.json \
  --abandoned \
  --confirm-no-live-processes
```

The command returned `PENDING_PRESERVATION`. A later preservation commit,
push, issue comment, and `record-preserved` transition must complete this
root before run 0002 may be allocated.

## Remote preservation

The sealed root and pending index were pushed in commit
`89adf00d297dd7e8fe07d94700989a533180e14c`. The permanent explanatory
report is
<https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-5055476266>.
The exact machine-validated preservation proof is
<https://github.com/tensor4all/tenferro-rs/issues/1436#issuecomment-5055483629>.

`record-preserved` fetched the branch, validated the commit's root, index and
worklog blobs, fetched the exact proof comment, and returned `PRESERVED`.
The follow-up index commit records that durable transition. Run 0001 remains
negative evidence and does not become `current_evidence_root`.

## Follow-up

1. Preserve this negative root without modifying its sealed inventory.
2. Reproduce only the failed build command in non-evidence diagnostic
   scratch, retaining stdout/stderr.
3. If the failure is environmental, document it and start a fresh root id
   only after this root reaches `PRESERVED`.
4. If the tooling discarded actionable failure evidence, fix and verify that
   defect, freeze and push a superseding measurement candidate if required,
   and update Issue #1436 before collecting new evidence.
