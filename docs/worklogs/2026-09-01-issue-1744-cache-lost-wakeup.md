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
- Add a test-only callback around the real entry-wait boundary rather than a
  probabilistic repetition test. The regression pauses the waiter after
  registration, completes producer publication, and then permits the first
  wait attempt.
- After the bounded observation window, repeatedly issue replacement
  notifications while polling for completion up to a rescue deadline. Use an
  unscoped waiter so even deadline exhaustion cannot block failure reporting on
  an implicit scoped-thread join.
- Do not change cache ownership, limits, statistics, public APIs, dependencies,
  backends, feature flags, or CI workflow policy.

## Same-pattern review

The other Condvar users in `runtime/cache.rs` and `runtime/execution.rs` already
check their protected predicates in loops before waiting. No related lost-wakeup
instance was found in those paths.

## Verification

- The deterministic regression failed against the old unconditional wait in
  about one second with `waiter missed publication before its first
  condition-variable wait` and completed without hanging.
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

The regression uses a one-second bounded observation only to distinguish the
fixed path from an old waiter that needs rescue notifications. The ordering of
waiter registration, producer publication, and waiter release is controlled by
barriers, so correctness does not depend on probabilistically hitting the race.
The ten-second rescue deadline is failure containment for a broken
implementation, not part of the passing-path correctness criterion.
