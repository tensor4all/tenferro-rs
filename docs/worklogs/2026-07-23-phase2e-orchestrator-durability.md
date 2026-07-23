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

- The only index paths are
  `docs/worklogs/2026-07-21-phase-2e-index.json` and
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

## Task 8B re-review hardening

The follow-up review identified four remaining pathname and lifecycle gaps. The
index transaction now retains the exact `docs/worklogs` descriptor for its lock,
remote comparison, reads, atomic replacement, rename, and directory fsync, and
rejects a renamed/replaced parent before touching the replacement. ACTIVE work
now carries the held root identity into the parent runner and subprocess journal;
parent checkpoints are descriptor-relative, and stage-worker root path use is
guarded by the retained identity. The subprocess cleanup guard begins
immediately after `Popen`, so identity, `getpgid`, and journal failures terminate
and reap the new process. Initialization creates a durable empty process journal
before any initializer can launch a child, allowing manual abandonment to
distinguish “no child launched” from missing or tampered evidence.

The focused replacement/cleanup tests and the complete affected matrix pass:
228 tests across the outer orchestrator, protocol, build, and gate suites.

## Task 8C preservation validation

Task 8C completes the local preservation implementation without performing any
remote mutation or GitHub write. Git-index and commit validation now retain
each exact `(mode, path, blob identity, content)` tuple, reject symlink/special
modes, include ignored normative files, and reconstruct the root in a fresh
temporary directory for structural and semantic validation. The staged or
committed root is bound to the exact TERMINAL status, aggregate/abandonment
digest, and ledger digest; the durable pending index and curated worklog must
also be exact mode-`100644` Git objects.

Remote validation accepts only canonical SSH/HTTPS origin URLs for
`tensor4all/tenferro-rs`, fetches only the fixed preservation branch, and
requires the preservation commit to be reachable from it. GitHub proof uses
the canonical API representation and verifies comment id, permanent URL,
repository/issue association, and body identity. PRESERVED events now retain
the complete ACTIVE/TERMINAL identity, support only exact idempotent replay,
and serialize against both parallel preservation and later starts under the
index-then-root lock order.

The final exact Step 4 suite from the implementation plan passed 433 tests.
It includes strict four-line preservation-comment parsing, durable index URL
validation, and staged/commit rejection of the force-added index lock. Focused
Git tests use temporary local repositories and fake remote/comment adapters.
No push, fetch from a real remote, or GitHub comment was performed.

## Task 10 operational wrapper

The Task 10 integration pass narrowed the public CLI to the six literal
operational command shapes in the umbrella plan while retaining the read-only
experiment-identity comparison command. Internal reservation, provenance,
contract, runtime-path, and context identities are derived by `start`, stored
as a digest-bound canonical context under the evidence root, and recovered
from the fixed index/root pair for continuation and preservation commands.
The dated index path is fixed; alternate indexes, foreign roots, reused
scratch, stale candidates, and the former internal identity flags are rejected.
The private stage worker uses a separate parser and is absent from public help.
Root-only commands derive the repository from an absolute canonical
`docs/worklogs/artifacts` root and verify its Git top level, rather than using
the caller's current directory; the fixed index and ACTIVE binding remain the
authority for mutable lifecycle operations.

`start` now creates a private empty `HOME` and a private `CARGO_HOME` outside
the measured scratch root. The latter links only canonical Cargo `git` and
`registry` source caches and contains no Cargo config or credentials. Before
creating `ACTIVE`, the wrapper runs both Task 7 `cpu-faer` feature queries with
`CARGO_NET_OFFLINE=true`. A real local preflight on 2026-07-23 passed for
`tenferro-cpu` and `tenferro-ad`; the focused outer-orchestrator suite passed
101 tests after independent specification review. At an earlier checkpoint,
the exact seven-module Phase 2E verification matrix passed 444 tests in 214.3
seconds; it awaits rerun after the current review cycle is accepted.

An independent specification review then added subprocess regressions for
private-worker help isolation, execution from `/tmp` with an absolute evidence
root, foreign/symlink root rejection, and one deterministic public lifecycle in
an isolated temporary Git repository. Separate exact
`python3 -m scripts.run_phase2e` subprocesses exercise
start/rerun-invalid-lane/continue, silent and printed validation, record-index,
commit-object reconstruction, and record-preserved. The real lifecycle owns
the fixed index and ACTIVE event, canonical context, locks, retry progress,
aggregate seal, and terminal/PRESERVED transitions. Temporary-copy adapters
replace only preflight, the expensive measurement runner and its semantic
validators, and external remote/comment verification; the production module
has no test-only switch. The lifecycle stops only at the canonical
`allocation/direct-current-main` lane, records a real ledger
`RETRYABLE` attempt, reruns that whole lane to `COMPLETE`, and then continues.
The same contract rejects build, dispatch, characterization, aggregate, and
prefix-lookalike stages before calling the runner; the allowlist contains only
the four exact allocation/timing lane stages. This regression also exposed
pathname inventory traversal through `/proc/self/fd/N`; descriptor-bound
orchestration now inventories through the held directory descriptor and has a
focused regression. The tests do not launch measurements or mutate remote
state.

A later quality pass closed four durability gaps. ACTIVE initialization now
passes the retained evidence-root identity into its initializer and writes the
context and ledger relative to that descriptor; a deterministic
rename-to-symlink race proves the original inode is sealed and the replacement
receives no bytes. Controlled HOME/CARGO_HOME creation tracks only directories
owned by the current start attempt, rolls them back after mkdir, cache-link,
validation, preflight, and pre-ACTIVE index/reservation failures, preserves the
primary exception when cleanup also fails, and leaves the exact start command
retryable. Non-`FileExistsError` setup I/O is translated to the public typed
error path. Stage-context opens are nonblocking, no-follow, close-on-exec,
regular-file checked, and descriptor leak tested for FIFO, directory, and
character-device inputs.

The final re-review rejects every relative stage-context path before resolution
or open; direct private-worker tests cover both a missing relative path and a
relative symlink to a FIFO without blocking, reading the target, or emitting a
traceback. Pre-ACTIVE execution-home ownership now retains the canonical parent
descriptor and one descriptor plus device/inode identity for each created
leaf. Rollback empties contents only through that retained leaf descriptor,
accepting only the platform's exact final-root `EINVAL` from the
symlink-attack-resistant descriptor-relative `rmtree`. It then renames the
current parent entry to a cryptographically random quarantine name and never
removes that parent pathname. Exact device/inode matches are verified empty and
retained as tombstones; mismatches are also retained with a distinct
diagnostic, so concurrently substituted data is never recursively deleted.
Cleanup records every tombstone path in the primary error notes. A failed
attempt leaves at most one tombstone per home it created, while freeing the
fixed HOME/CARGO_HOME names for an exact retry. Normal tombstones are tiny empty
directories outside the measured scratch/evidence build artifacts and remain
deliberately for identity safety. Every success and error path attempts to close
every retained descriptor even if an earlier close reports failure. If a leaf
descriptor cannot be acquired, cleanup explicitly leaves the unowned path
rather than guessing its identity and annotates the typed primary failure.
Once initialization has committed ACTIVE or ABANDONED, descriptor-close errors
are diagnostics: code 0 still launches the fixed stage runner, while code 5
still prints `ABANDONED_INITIALIZATION` and returns 5. The focused
outer-orchestrator suite passes 114 tests.

The temporary public-lifecycle adapter moved from the main test class to
`scripts/phase2e_public_lifecycle_fixture.py`. Its helper asserts one exact
entrypoint marker, records the canonical source path and SHA-256, and embeds
that digest in the temporary copy before installing adapters; production has
no fixture import or switch. The helper still reuses the comprehensive
`make_complete_root` semantic artifact producer because independently
duplicating the large Task 8A build/gate schemas would increase
producer/validator drift. It explicitly excludes lifecycle-owned context,
ledger, journal, children, progress, locks, aggregate, index, and preservation
state, all of which the subprocess test continues to create through production
paths.

## Remaining work

The final reviewer recheck remains pending. Operational preservation
(committing, pushing, and posting the real #1436 proof) remains an explicit
later external action and was not authorized in this implementation pass.
