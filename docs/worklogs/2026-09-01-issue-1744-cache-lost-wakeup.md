# Issue #1744: Prepared-plan cache lost wakeup

## Scope

Fix the prepared-plan cache waiter race that left the macOS workspace test for
PR #1741 blocked after producer publication. Preserve the existing single-flight
cache semantics and public API.

## Context read

- Shared tensor4all common repository/performance and Rust rules.
- `AGENTS.md`, `CONTRIBUTING.md`, and the cache, test-organization, and
  contribution sections of `REPOSITORY_RULES.md`.
- Issue #1744 and the stalled macOS and downstream GPU-gate Actions jobs.
- `crates/tenferro-runtime/src/runtime/cache.rs` and its module-local cache
  tests.
- Neighboring Condvar loops in the runtime queue-ticket and execution paths.

## Root cause

`handle_entry` registered a waiter while holding the cache mutex, then returned
an `EntryWaitGuard`. `EntryWaitGuard::wait_once` later reacquired the mutex and
unconditionally entered `Condvar::wait`. A producer could publish and notify in
the interval between registration and that first wait, leaving the waiter
asleep with no remaining state transition to wake it.

The existing same-key test exposed the race nondeterministically. Its eight
callers verify that one producer runs and that every waiter receives the same
cached `Arc`, but it did not control the registration-to-wait interval.

## Decisions

- Recheck the guarded predicate after acquiring the mutex and wait in a loop
  only while the same entry and attempt remain `Preparing`. This also handles
  spurious Condvar wakeups.
- Keep retained and ephemeral completion accounting in the existing
  `complete_with_locked_state` path.
- Add test-only callbacks around the real entry-wait boundary and immediately
  before `Condvar::wait` rather than a probabilistic repetition or timeout
  test. The regression pauses the waiter after registration, completes producer
  publication, and then permits the predicate check. Entering the condition
  variable after publication panics directly, so a broken implementation fails
  without sleeping.
- Do not change cache ownership, limits, statistics, public APIs, dependencies,
  backends, feature flags, or CI workflow policy.

## Same-pattern review

The other Condvar users in `runtime/cache.rs` and `runtime/execution.rs` already
check their protected predicates in loops before waiting. No related lost-wakeup
instance was found in those paths.

## Verification

- The deterministic regression failed against the old unconditional wait in
  0.03 seconds with `waiter must not sleep after producer publication` and
  completed without hanging.
- `cargo nextest run -p tenferro-runtime --cargo-profile ci 'same_key_'
  --no-fail-fast`: 2 passed.
- `cargo nextest run -p tenferro-runtime --cargo-profile ci --no-fail-fast`:
  558 passed. The compile-fail fixture was run with network access because it
  resolves the crates.io index in a temporary project.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p
  tenferro-runtime same_key_waiter_does_not_miss_publication_before_first_wait'`:
  passed repository formatting, documentation snippets, root/extension clippy,
  and the focused debug-profile regression test.

## Residual risk

The ordering of waiter registration, producer publication, predicate checking,
and any attempted condition-variable wait is controlled by barriers and a
test-only hook. The regression has no scheduling deadline, so correctness does
not depend on how quickly the waiter runs after it is released.
