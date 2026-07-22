# Phase 2E Allocation Durability Hardening

## Summary

Task 6 cycles 4 through 6 harden the allocation non-inferiority campaign against
forged terminal JSON, result-selecting finalization failures, startup crash
windows, cleanup failure loss, and concurrent processes using one attempt with
different artifact roots.

## Context reviewed

- `docs/superpowers/specs/2026-07-21-phase-2e-atomic-noninferiority-campaign-design.md`
- `docs/superpowers/specs/2026-07-22-phase2e-allocation-durability-design.md`
- `scripts/phase2e_protocol.py`
- `scripts/phase2e_build.py`
- `scripts/run_phase1_eager_campaign.py`
- `scripts/run_phase2e_allocation_campaign.py`
- their focused protocol, build, Phase 1, and allocation tests

## Decisions

- Reuse the normative outer-root `orchestrator.lock`; no allocation-specific
  lock file was added. The child first pins and locks the outer evidence
  directory, then validates and locks `orchestrator.lock` relative to that
  descriptor. This closes lockfile replacement bypass while preserving the
  future orchestrator's index-then-root order.
- Extend every ledger attempt with one exact artifact ownership schema. Timing
  attempts use `None` identities and `NOT_APPLICABLE`; allocation attempts
  move from `RESERVED` to `BOUND` before any launch.
- Initialize in the order `RESERVED ledger`, exclusive root, `BOUND ledger`,
  canonical `RUNNING` manifest, then launches. A `RESERVED` crash is closed as
  validity `INCONCLUSIVE` without touching an unproven root; a BOUND root with
  no manifest is terminalized from a reconstructed empty canonical prefix.
- Decode persisted evidence only when its bytes are canonical JSON and contain
  neither duplicate keys nor non-finite numbers.
- Reconstruct the complete allocation sequence and PASS/FAIL result from the
  observations. Persisted state fields never substitute for that check.
- Derive all terminal state through one classifier shared by generation,
  validation, and recovery. A final failed record identifies its launch; an
  all-success prefix identifies the next launch; a full inconsistent inventory
  identifies the first `(case, role)` mismatch; 168 successful consistent
  records can only be recomputed COMPLETE PASS/FAIL.
- Suppress cleanup failures only when a primary failure is already active.
  Normal outcomes never hide lock, pinned-resource, or recovery-root close
  failures.
- Checkpoint a failed probe's canonical `record = null` RUNNING tail before
  finalization, so recovery preserves the exact launch descriptor and reason.
- Require persisted allocation probe `protocol_version` to have exact integer
  type and exact value.

## Rejected alternatives

- Locking `evidence-ledger.json` itself is invalid because atomic replacement
  changes its inode.
- Locking only `orchestrator.lock` is insufficient because replacing that
  pathname creates a new unlocked inode; the pinned directory lock is the
  stable outer serialization layer.
- A new sibling allocation lock or ownership sidecar would duplicate the
  accepted outer-root authority and ledger source of truth.
- Reusing an existing empty artifact root cannot distinguish a crash remnant
  from an unowned directory, so new execution uses exclusive creation.
- Adding another persisted failure-location field was unnecessary: the exact
  canonical prefix, final failed observation, or first deterministic mismatch
  already identifies the location independently, and the shared classifier
  prevents generator/validator reason drift.

## Verification intent

The focused suite covers strict terminal mutations, initialization atomic-write
states, sealed `pass_fds` launches, and real persisted integration. Shared
protocol and Phase 1 tests cover the ledger migration. A real forked-process
test proves lock serialization and a separate `os._exit` test proves crash
release and zero-launch recovery.

## Cycle 4 verification results

- `python3 -m unittest scripts.test_phase2e_build scripts.test_run_phase2e_allocation_campaign`: 112 passed.
- `python3 -m unittest scripts.test_phase2e_protocol scripts.test_run_phase1_eager_campaign -v`: 109 passed.
- The two-process serialization test passed independently in 0.65 seconds.
- `python3 -m py_compile` for all six changed Python modules and `git diff --check` passed.

The lock-removal mutation also exposed a test-harness cleanup defect: an early
serialization assertion could leave the deliberately blocked first child
running. The test now releases and joins both children in `finally`, so a
future regression fails promptly instead of hanging the test runner.

## Cycle 5 verification results

- The forged public-recovery reproducer failed before the fix because a full
  successful inventory could be relabeled `INCONCLUSIVE`; it now rejects that
  evidence without launching probes or mutating the ledger.
- Focused classifier tests cover launch-168 `record = null`, full successful
  inconsistency, fixed prefix interruption, and complete recovery after a
  finalization-stage interruption.
- Initialization tests observe `RESERVED -> BOUND -> RUNNING` and recover
  BOUND post-commit plus RUNNING pre/post-commit control interruptions with
  zero relaunches.
- Cleanup tests inject failures after normal outcomes at the orchestrator lock,
  pinned-resource, and recovery-root sites and verify typed errors or unchanged
  control exceptions.
- `python3 -m unittest scripts.test_phase2e_build scripts.test_run_phase2e_allocation_campaign`:
  119 passed.
- `python3 -m unittest scripts.test_phase2e_protocol scripts.test_run_phase1_eager_campaign -v`:
  109 passed.
- `python3 -m py_compile` for all six Phase 2E Python implementation/test
  modules and `git diff --check` passed.
- Repository-rules worktree review from `9c5aa7ed` passed with no findings.

## Cycle 6 verification results

- A public-stage precommit reproducer now proves that a failed first launch is
  durably checkpointed as RUNNING and recovers to the same terminal descriptor
  and reason with exactly one actual launch.
- A real forked-process race atomically replaces `orchestrator.lock` after the
  first process acquires it; the pinned outer-directory lock still serializes
  the second process. Canonical path/ledger colocation and exact-once cleanup of
  both descriptors have direct coverage.
- Failed checkpoint and lock-release tests prove ordinary and control cleanup
  failures remain secondary to the active probe/campaign failure.
- Persisted probe manifests reject float and boolean `protocol_version` values
  even when Python equality would match the supported integer.
- Process readiness waits and assertions are inside `finally` cleanup; bounded
  join escalates to terminate and kill. A missing-readiness regression proves
  the child is reaped.
- `python3 -m unittest scripts.test_phase2e_build scripts.test_run_phase2e_allocation_campaign`:
  125 passed.
- `python3 -m unittest scripts.test_phase2e_protocol scripts.test_run_phase1_eager_campaign -v`:
  109 passed.
- `python3 -m py_compile` for all six Phase 2E Python implementation/test
  modules and `git diff --check` passed.

## Residual risks

The lock contract assumes the accepted orchestrator has already created an
empty canonical regular `orchestrator.lock`. The allocation child intentionally
does not create, replace, truncate, or delete that authority.
