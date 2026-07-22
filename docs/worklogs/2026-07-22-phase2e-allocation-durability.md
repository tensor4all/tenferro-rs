# Phase 2E Allocation Durability Hardening

## Summary

Task 6 cycle 4 hardens the allocation non-inferiority campaign against forged
terminal JSON, startup crash windows, and concurrent processes using one
attempt with different artifact roots.

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
  lock file was added. The child acquires only this root lock, preserving the
  future orchestrator's index-then-root order.
- Extend every ledger attempt with one exact artifact ownership schema. Timing
  attempts use `None` identities and `NOT_APPLICABLE`; allocation attempts
  move from `RESERVED` to `BOUND` before any launch.
- Initialize in the order `RESERVED ledger`, exclusive root, canonical
  `RUNNING` manifest, `BOUND ledger`, then launches. A `RESERVED` crash is
  closed as validity `INCONCLUSIVE` without touching an unproven root.
- Decode persisted evidence only when its bytes are canonical JSON and contain
  neither duplicate keys nor non-finite numbers.
- Reconstruct the complete allocation sequence and PASS/FAIL result from the
  observations. Persisted state fields never substitute for that check.

## Rejected alternatives

- Locking `evidence-ledger.json` itself is invalid because atomic replacement
  changes its inode.
- A new sibling allocation lock or ownership sidecar would duplicate the
  accepted outer-root authority and ledger source of truth.
- Reusing an existing empty artifact root cannot distinguish a crash remnant
  from an unowned directory, so new execution uses exclusive creation.

## Verification intent

The focused suite covers strict terminal mutations, initialization atomic-write
states, sealed `pass_fds` launches, and real persisted integration. Shared
protocol and Phase 1 tests cover the ledger migration. A real forked-process
test proves lock serialization and a separate `os._exit` test proves crash
release and zero-launch recovery.

## Verification results

- `python3 -m unittest scripts.test_phase2e_build scripts.test_run_phase2e_allocation_campaign`: 112 passed.
- `python3 -m unittest scripts.test_phase2e_protocol scripts.test_run_phase1_eager_campaign -v`: 109 passed.
- The two-process serialization test passed independently in 0.65 seconds.
- `python3 -m py_compile` for all six changed Python modules and `git diff --check` passed.

The lock-removal mutation also exposed a test-harness cleanup defect: an early
serialization assertion could leave the deliberately blocked first child
running. The test now releases and joins both children in `finally`, so a
future regression fails promptly instead of hanging the test runner.

## Residual risks

The lock contract assumes the accepted orchestrator has already created an
empty canonical regular `orchestrator.lock`. The allocation child intentionally
does not create, replace, truncate, or delete that authority.
