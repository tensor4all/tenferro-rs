# Phase 2E candidate `cd30b008`: abandoned run 0001

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

## Follow-up

1. Preserve this negative root without modifying its sealed inventory.
2. Reproduce only the failed build command in non-evidence diagnostic
   scratch, retaining stdout/stderr.
3. If the failure is environmental, document it and start a fresh root id
   only after this root reaches `PRESERVED`.
4. If the tooling discarded actionable failure evidence, fix and verify that
   defect, freeze and push a superseding measurement candidate if required,
   and update Issue #1436 before collecting new evidence.
