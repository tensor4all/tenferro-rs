# Worklog: #1556 Phase 0 A0.2 Event Provenance

## Scope

Implement the accepted Phase 0 A0.2 event-provenance contract on top of
`c2aafc72`. This work is limited to provenance-qualified event domains,
provider-neutral token admission, the scheduler-owned transfer host bridge,
and immediate/CUDA/WebGPU event-domain adapters. Typed CUDA device discovery,
storage ownership/allocation/lease changes, and fixed-ID cleanup remain out of
scope. The follow-up recorded below is limited to the private runtime-owned
event-run lifecycle in `runtime/execution.rs`, the CUDA/WebGPU provider run
lifecycle, and their existing private tests.

## Context read

- The current worktree is `codex/issue-1556-engine-identity` at accepted base
  `c2aafc72`.
- `AGENTS.md`, `REPOSITORY_RULES.md`, the Phase 0 plan, and
  `docs/worklogs/2026-08-01-issue-1556-phase0-a01-provider-device-binding.md`.
- Current `EventDomainId`, `EventDomainDriver`, `EventDomainRun`, and
  `EventToken` definitions; snapshot freeze and event-domain allocation;
  `ScheduledEventDomains`; resolved transfer routing and the shared
  `FrozenTransferRegistry`; immediate, CUDA, and WebGPU providers; and their
  owning tests.
- Shared Tensor4all common, Rust, performance, and docs/test guidance.

## RED tests and evidence

Added the runtime-owned provenance suite in
`crates/tenferro-runtime/src/runtime/tests/event_domain.rs` and provider-owned
fallback-contract tests in the CUDA and WebGPU runtime adapter test modules.
The tests cover fully qualified runtime/epoch/registration identity, direct
foreign admission, run-domain mismatch, forged completion origin, scheduler
transfer filtering and host waits, same-destination dependencies, repeatable
fan-out waits, third-domain rejection, wait failure, and launch suppression.

The focused runtime RED command was:

```text
cargo test -p tenferro-runtime --lib event_domain --no-fail-fast
```

It exited `101` because the implementation still lacks the intended A0.2
surface and behavior: `EventToken::origin`, `EventDomainRun::domain`, the
`EventDomainDriver::begin_run(EventDomainId)` parameter, the qualified
`EventDomainId` construction/accessors, `EventDomainError`/structured
`Error::EventDomain`, and the scheduler-owned test seam. The compiler did not
report a test typo after the initial fixture type correction.

The CUDA provider RED command was:

```text
cargo test -p tenferro-gpu --features cuda --lib \
  cuda_event_domain_has_no_generic_foreign_token_wait_fallback --no-fail-fast
```

It ran one test and exited `101` at the assertion because
`crates/tenferro-gpu/src/cubecl/event_domain.rs` still contains the generic
foreign-token `_ => dependency.wait()?` fallback.

The WebGPU provider RED command was:

```text
cargo test -p tenferro-gpu --features webgpu --lib \
  webgpu_event_domain_has_no_generic_foreign_token_wait_fallback --no-fail-fast
```

It ran one test and exited `101` at the assertion because
`crates/tenferro-gpu/src/webgpu/event_domain.rs` still contains the same
generic foreign-token fallback.

## Implementation decisions

- `EventDomainId` will contain the exact snapshot `RuntimeId`, `RuntimeEpoch`,
  and direct-engine `RegistrationIdentity`; freeze will derive it from those
  values rather than a slot-local counter.
- `EventDomainError` will retain expected/actual domains plus operation and
  node context. Runtime validation will wrap it structurally rather than
  encoding mismatches only in display text.
- `ScheduledEventDomains` will validate run and completion provenance, perform
  repeatable host waits only for source-domain dependencies of transfer nodes,
  and pass only same-destination-domain tokens to destination runs.
- Immediate, CUDA, and WebGPU runs/tokens will carry the exact domain and
  reject foreign origins before launch. CUDA/WebGPU will reject incompatible
  same-origin token types rather than invoking generic host waits.
- The frozen resolved routing representation and shared
  `FrozenTransferRegistry` remain unchanged except for the richer canonical
  event-domain value.

## Residual risk before implementation

The existing in-tree doctests, integration tokens, and provider adapters still
use the intentionally replaced no-argument driver API and default-less token
contract. They must be migrated in the implementation phase without adding
compatibility overloads or a default origin.

## Final contract corrections

- `EventDomainOperation` is a closed public enum with the five semantic
  operations `BeginRun`, `Enqueue`, `Drain`, `TransferBridge`, and
  `ValidateCompletion`. Runtime, Immediate, CUDA, WebGPU, tests, exports, and
  docs use the enum uniformly; provider function-name strings are not semantic
  control data.
- `ScheduledEventDomains` checks the run domain once after `begin_run` and
  again before dependency classification, so a changed run cannot host-wait or
  forward a transfer dependency. It checks again after classification and
  immediately before every `enqueue`, so a stateful provider cannot launch with
  stale provenance. The malicious stateful-run tests assert typed rejection,
  zero launch, zero destination enqueue, and zero source host-waits where
  applicable.
- `EventDomainRun::drain` documents and implements retirement of already-
  progressing work without depending on another run's drain. Explicit drain
  returns typed provider errors. The runtime-owned run wrapper now uses the
  tagged states `Pending(Box<dyn EventDomainRun>)`, `Retired`, and `Failed`.
  It consumes `Pending` before provider code, catches a provider drain panic as
  `EventDomainError::DrainPanicked`, and rejects every later operation with a
  typed lifecycle source. `ScheduledEventDomains` attempts every started run
  in first-use order and retains every failure in that order through the
  existing suppressed-error chain. Drop is a one-shot, non-panicking fallback
  for a skipped explicit drain; it retires only `Pending`, never retries a
  terminal run, and may suppress its diagnostics because Drop has no `Result`.
- Returned completion provenance is intentionally validated after the provider
  `enqueue` has returned: the launch has already happened by contract. The
  scheduler validates the token before recording it and before any downstream
  launch; dependency/run/wait provenance failures remain pre-launch checks, but
  malformed returned completions cannot undo the same-node launch.
- CUDA and WebGPU event-domain runs now carry the shared private
  `Pending`/`Retired`/`Failed` lifecycle. Explicit retirement consumes
  `Pending` before invoking native provider code; success reaches `Retired`,
  while a typed error or contained panic reaches `Failed`. Drop invokes the
  existing non-unwinding retirement helper only for `Pending`, preventing the
  runtime wrapper's provider-box drop from submitting a second barrier.
- CUDA and WebGPU retain separate provider-specific explicit drain/error paths.
  Their implicit run, submission-guard, and native-handle cleanup bodies call
  the one crate-private `tenferro-gpu` retirement helper as a one-shot,
  non-panicking best-effort fallback. Phase 0 has no bespoke structured Drop
  sink, panic-payload attestation, or untrusted-destructor threat model.

## A0.2 baseline verification

The final verification results are:

- `cargo test -p tenferro-runtime --lib --no-fail-fast`: 372 passed;
  `cargo test -p tenferro-runtime --tests --no-fail-fast`: 372 unit and 117
  integration tests passed; and `cargo test -p tenferro-runtime --doc
  --no-fail-fast`: 403 passed.
- `cargo test -p tenferro-gpu --lib --no-fail-fast`: default feature surface
  compiled and passed with zero tests; the CUDA feature suite passed 68 with
  118 hardware-dependent tests ignored; the WebGPU feature suite passed 20
  with one adapter-dependent test ignored. The shared retirement panic test,
  CUDA/WebGPU no-generic-fallback source checks, and focused adapter tests all
  passed. CUDA and WebGPU doctests passed 19 and 26 respectively.
- `cargo fmt --all -- --check`, `scripts/check-doc-snippets.py --check`, the
  repository fast gate (including root and extension CI-parity clippy), and
  `python3 scripts/check-storage-ownership-contracts.py` all passed.
- `cargo clippy -p tenferro-gpu --features webgpu --all-targets -- -D
  warnings` passed. The equivalent CUDA feature clippy invocation remains
  blocked by warnings in unchanged, pre-existing GPU files outside this slice
  (cuTENSOR argument-count/test-module layout, fusion clone/type-parameter
  lints, and CubeCL kernel macro/style lints); CUDA compilation and the full
  feature test suite passed.

No storage ownership or allocation behavior is part of this slice.

## Follow-up: event-run lifecycle cleanup

The RED tests were added to the existing runtime event-domain module:

```text
cargo test -p tenferro-runtime --lib \
  scheduler_run_drain_is_terminal_and_does_not_call_provider_again --no-fail-fast
cargo test -p tenferro-runtime --lib \
  scheduler_drain_returns_all_failures_in_run_order --no-fail-fast
```

The first failed because a successful run could be drained again. The second
failed because later provider failures were attempted but discarded after the
first error. The tests also cover the `Retired`, `Failed`, and contained-panic
terminal paths, provider-call counts, deterministic drain order, and typed
lifecycle errors. The execution test
`scheduled_execution_cleanup_error_preserves_primary_error` verifies that the
primary execution error remains the standard source while cleanup stays
attached to the typed suppression aggregate. No source-substring tests were
added.

The implementation is intentionally private and small: the runtime lifecycle
is owned by `RuntimeOwnedEventDomainRun`, the provider lifecycle is shared by
the CUDA/WebGPU runs, and ordered cleanup failures are folded with
`Error::with_suppressed`. When execution and cleanup both fail, the same typed
aggregate keeps the execution error as the standard source and attaches the
ordered cleanup aggregate as suppressed metadata. The provider lifecycle
helper's counter tests prove that explicit drain followed by Drop and implicit
Drop each invoke the retirement closure exactly once.

Fresh GREEN verification for this follow-up:

- `cargo fmt --all -- --check`, `git diff --check`, and
  `cargo check -p tenferro-runtime`: passed.
- `cargo test -p tenferro-runtime --lib --no-fail-fast`: 386 passed.
- `cargo test -p tenferro-runtime --tests --no-fail-fast`: 386 unit tests and
  118 integration tests passed.
- The focused primary/cleanup integration filter passed both tests.
- `cargo clippy -p tenferro-runtime --all-targets -- -D warnings` remains
  blocked by two pre-existing warnings in unchanged
  `runtime/engine_registration.rs` and `runtime/snapshot.rs`; this cleanup
  does not alter those files.

## Provider retirement one-shot follow-up

The provider-side RED command,
`cargo test -p tenferro-gpu --features cuda --lib event_retirement --no-fail-fast`,
failed at compilation because the new lifecycle test seam was not yet
implemented. After the shared `Pending`/`Retired`/`Failed` state and both
provider integrations were added, the focused CUDA and WebGPU commands each
passed all four retirement-helper tests. The complete CUDA library feature
run passed 87 runnable tests with 118 hardware-dependent tests ignored. The
complete WebGPU library run compiled and passed the retirement and other
non-adapter tests, but three existing adapter-probing tests exceeded 60 seconds
in this environment and the run was terminated; the bounded WebGPU retirement
command and WebGPU `cargo check` both passed.

Additional fresh checks passed:

- `cargo test -p tenferro-runtime --lib event_domain --no-fail-fast`: 20 passed.
- `cargo test -p tenferro-runtime --test integration runtime_event_domains
  --no-fail-fast`: 4 passed.
- `cargo check -p tenferro-gpu --features cuda` and `--features webgpu`.
- `cargo fmt --all -- --check` and `git diff --check`.

## Residual risk after implementation

Native CUDA/WebGPU event execution and Metal integration still require their
respective hardware/provider environments. The common non-unwinding and
one-shot retirement boundaries are in place, but Metal's event adapter will be
implemented in its own follow-up slice. Drop-only fallback diagnostics are
intentionally not surfaced; no bespoke provider diagnostic sink, panic-payload
attestation, or untrusted-destructor threat model is part of this Phase 0
cleanup. The A0.2 API intentionally has
no compatibility shim for the replaced unqualified event-token contract.
