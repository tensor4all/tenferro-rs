# Phase 2E orchestrator durability hardening

## Summary

Task 8B hardened the local Phase 2E orchestrator after the audited Task 8A
baseline `7f714d35`. The campaign index is now bound to fixed repository paths,
index and root operations use one index-then-root lock order, filesystem trust
is descriptor-bound, terminal lifecycle events carry the complete experiment
identity, and every stage subprocess is durably journaled and reaped.

Remote branch preservation and GitHub reporting were deliberately left for
Task C.

## Context reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, and workspace `CODING_RULES.md`
- shared `tensor4all-agent-rules` common repository and test rules
- `scripts/run_phase2e.py`, `scripts/phase2e_protocol.py`, and their Phase 2E
  contract tests
- the existing Task 8A gate, allocation, timing, and aggregate validators

The private `tensor4all-agent-knowledge` checkout was not configured on this
machine. The checked-out shared rules repository was available and used.

## Decisions

- The only index paths are `docs/worklogs/phase2e-index.json` and
  `docs/worklogs/.phase2e-index.lock` below the canonical repository root.
  CLI callers cannot substitute either path.
- The first remote comparison, optional index creation, index read, and ACTIVE
  reservation happen under one index lock transaction.
- Operations that need both global and root state acquire the index lock first,
  then a no-follow regular root lock relative to a retained root descriptor.
- Index reads and writes use no-follow leaf access and descriptor-relative
  atomic replacement. Lock files are regular, owner-held mode `0600` files
  whose device/inode identity is checked while held.
- Initialization distinguishes failures before ACTIVE from failures after an
  ACTIVE replacement committed. The latter self-seal as
  `INITIALIZATION_FAILURE` and append one complete TERMINAL event.
- TERMINAL events copy candidate, tree, reservation, root, experiment,
  campaign, command, and context identity from ACTIVE and add exact ledger and
  root digests. Exact `record-index` replay is idempotent; changed replay is an
  error.
- Each `Popen` is journaled with PID, PGID, Linux start ticks, stage, and argv
  before the first wait. Timeout or interruption uses TERM, a bounded grace
  wait, KILL when necessary, and an unconditional reap before recording the
  terminal journal state.
- Manual abandonment has no boolean or caller-supplied PGID bypass. It requires
  a complete durable journal and proves every recorded process group is gone.
  Initialization abandonment cannot be replayed through that path.

## TDD and race coverage

Tests were observed failing before each implementation block for:

- arbitrary index CLI paths and split first-index transactions;
- ancestor/final symlinks, FIFO lock files, index aliases, and live root
  replacement;
- foreign-root recording, incomplete terminal identity, and changed replay;
- pre-wait journaling, timeout cleanup, KeyboardInterrupt cleanup, missing or
  malformed journals, and removed CLI trust flags;
- parallel starts, parallel `record-index`, active-operation versus record
  locking, and an ACTIVE atomic write that committed before reporting a parent
  fsync failure.

The race tests assert bounded joins, one authoritative transition, no root
reuse, and identical results for exact concurrent replay.

## Verification

- `python3 -m unittest scripts.test_run_phase2e scripts.test_phase2e_protocol scripts.test_phase2e_build scripts.test_run_phase2e_gates`
  — 219 tests passed.
- `python3 -m py_compile` was run for every modified Python implementation and
  test module.
- `git diff --check` passed.

## Remaining work

Only Task C remains: remote branch preservation and GitHub preservation/report
validation. No remote mutation, push, or GitHub write was performed here.
