# Issue #1542: PreparedPlanCache poison visibility

## Scope

Make the `PreparedPlanCache` poison-recovery exception explicit and cover all
three guard cleanup paths without changing the typed poison errors returned by
normal cache operations.

## Context read

- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared Rust/cache rules.
- Issue #1542 and umbrella issue #1535.
- `crates/tenferro-runtime/src/runtime/cache.rs` and its module-local cache
  tests.
- The neighboring runtime lock-poison handling in `execution.rs` and the
  existing snapshot poison regression test.

## Decisions

- Keep `recover_state` as a non-`Result` helper because it is used only from
  `Drop` implementations, which cannot return a typed error. It recovers the
  guard needed to remove guard-owned accounting and notify waiters.
- Make the exception explicit with the repository-standard `// INVARIANT:`
  marker. The mutex poison bit is deliberately not cleared, so later
  `Result`-returning cache APIs continue to report `prepared-cache.state`.
- Do not add a logging dependency or broaden poison recovery to unrelated
  runtime or CPU locks. The existing typed failure surface is the visible
  signal available after a `Drop` cleanup has completed.
- Use a test-only poison helper following the existing runtime test-helper
  pattern. It creates real `std::sync::Mutex` poison and exercises producer,
  entry-waiter, and queue-ticket guard drops through cache behavior.

## Implementation

- Replaced the implicit `unwrap_or_else(PoisonError::into_inner)` expression
  with an explicit match and invariant marker.
- Added focused tests for producer/waiter cleanup and queued-ticket cleanup
  under a poisoned state. Both assert that subsequent `stats()` calls return
  the typed `RuntimeStateError::Poisoned` value.

## Verification

- `cargo fmt --all`
- `git diff --check`
- `cargo test -p tenferro-runtime poisoned_state_stays_visible --lib`
- `cargo test -p tenferro-runtime` (unit, integration, and doctest targets)
- `bash scripts/check-pr-fast.sh --base origin/main --no-fetch
  --coverage-reviewed --test 'cargo test -p tenferro-runtime --lib'`
- Workspace and extension clippy with the repository's `-D warnings`,
  `missing_errors_doc`, and `missing_panics_doc` flags.
- Committed-head deterministic repository-rules review: pass.
- Committed-head external LLM repository-rules review: pass, no findings.
- `python3 scripts/ci/run_profile.py coverage`: `203/203` files passed.
- `python3 scripts/ci/run_profile.py docs`: rustdoc, docs consistency, site
  rendering, links, and API inventory passed.

The repository's hardware-specific GPU lanes are unaffected by this
runtime-only safety change and were left to their normal CI policy.

## Residual risk

Other `into_inner` uses in `tenferro-cpu` are outside #1542 and retain their
existing issue scope. Runtime execution lock handling was audited and already
records poison as an explicit execution error rather than fabricating healthy
state; no unrelated lock behavior is changed here.
