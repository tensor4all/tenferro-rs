# Issue 967 Task 6 round 2: eager extension validation

## Summary

The eager extension bridge now validates the selected engine's physical input
signature before invoking a module factory and installs a returned extension
module through a module-scoped transactional replacement. CUDA FFT eager module
selection remains deferred to Task 7.

## Context read

- Task 6 brief/report and the Task 6 round-1 implementation/review artifacts.
- `crates/tenferro-ad/src/extension.rs` and eager runtime lock/session code.
- Runtime extension registration, snapshot, preparation, and error contracts.
- Shared Rust rules and `REPOSITORY_RULES.md` public-surface, validation, lock,
  and test guidance.

## Decisions

- `RuntimeReconfiguration::replace_extension_module_for_engine` configures a
  candidate module in a temporary registration record, verifies that that
  record's exact module ID owns `(family_id, engine_id)`, and only then inserts
  it into the candidate. The existing runtime compare-and-publish path and the
  eager extension-install mutex provide atomic publication. A failed candidate
  is never published, so an existing valid module/snapshot remains intact.
- `RuntimeConfigError::MissingExtensionEngine` carries the module ID so a
  mismatch identifies the returned module rather than only an aggregate family
  and engine pair.
- `EngineSnapshotView::accepts_input_signature` is the smallest cross-crate
  hidden query needed by `tenferro-ad`. It evaluates every advertised storage
  class through the registered ingress predicate; `InputSignature::from_reads`
  supplies placement, backend-family, allocation-domain, dtype, shape, and
  layout metadata.
- Rejected eager inputs use the existing `PrepareError::NoInputIngress`
  source with input index and placement. Factory errors remain propagated
  unchanged and the factory is not called for rejected inputs.

## Alternatives rejected

- Aggregate snapshot lookup after installation was rejected because another
  module could mask a returned module with no matching registration and because
  a failed replacement could remove a previously valid module.
- Inferring CUDA/CPU compatibility from placement in `tenferro-ad` was rejected;
  the owner-selected runtime engine remains the source of backend and ingress
  semantics.
- FFT eager backend-kind module selection was not added; it is explicitly Task
  7 scope.

## Verification

The implementation commit is recorded in the Task 6 report.

- Runtime extension tests: 16 passed.
- Runtime doctests: 413 passed.
- Full AD CPU suite: 560 passed; AD doctests: 147 passed.
- Einsum eager AD: 10 passed; FFT CPU: 20 passed; linalg CPU: 162 passed.
- CUDA bridge tests on A100: exact target, host-input rejection, and foreign
  CUDA-runtime rejection all passed with `--ignored`; rejected-input factory
  counters remained zero.
- Runtime and AD clippy (`-D warnings`), formatting, diff checks, and the
  diff-scoped repository-rules review passed. The review only noted that this
  worklog was required for the nontrivial validation/transaction change.

## Residual risks

WebGPU remains hardware-gated. CUDA FFT execution and FFT eager backend-kind
selection remain intentionally outside this task.
