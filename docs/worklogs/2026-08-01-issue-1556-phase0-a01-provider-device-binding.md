# Worklog: #1556 Phase 0 A0.1 Provider/Device Binding

## Scope

Implement the runtime control-plane provider/device binding slice from the
Phase 0 engine-identity plan. This work deliberately stops before typed CUDA
device discovery and event-token bridging.

## Context read

- Workspace and repository `AGENTS.md` instructions, `CODING_RULES.md`, and
  `REPOSITORY_RULES.md`.
- Issue #1556 body and the requested comments, including `5150596283` and
  `5150875825`.
- The complete local plan
  `docs/superpowers/plans/2026-08-01-phase0-engine-identity.md`.
- Existing runtime snapshot, transfer, preparation, scheduling, execution,
  CPU adapter, GPU adapter, and dependent AD call sites.

## Decisions

- `ProviderId` and immutable `ProviderDeviceIdentity` are public, validated,
  ordered, hashable control-plane values. `ProviderDeviceIdentity` is not
  tensor placement `DeviceId`.
- Target text is intentionally diagnostic-safe ASCII graphic text: it must be
  nonempty and rejects controls, whitespace, Unicode confusables, and other
  non-ASCII text. Providers with byte or Unicode identities must canonicalize
  them to an escaped ASCII representation before construction.
- Every direct `EngineRegistration` receives an explicit binding, and frozen
  engine views expose that same binding. CPU derives provider namespaces from
  the documented `CpuBackendKind` (`tenferro.cpu.faer` or
  `tenferro.cpu.blas`) and derives the target from the selected execution
  domain (`domain:<CpuDomainId::as_u64()>`).
- Candidate routes use one private `CandidateTransferBinding` state enum:
  `New`, `Preserved`, or validation-produced `Bound`. New registration is
  always `New`, independent of currently registered engines; only complete
  candidate validation binds both endpoints. Snapshot conversion creates
  `Preserved` bindings, and same-`Arc` idempotent registration does not reset
  them.
- Frozen preparation, scheduling, and execution use one immutable
  `FrozenTransferRegistry` keyed by `ResolvedTransferRoute`. Each resolved
  endpoint contains the logical endpoint, provider/device identity, and the
  current `EventDomainId`. `ExecutionLocation` stores that resolved endpoint
  as its only endpoint/identity/domain representation.
- Target rebind requires explicit endpoint-pair route removal, old-engine
  removal, new registration under the selected logical engine ID, route
  re-registration, and one full validation/freeze/publish. Retained preserved
  routes return a structured stale-route error and publish nothing. Direct
  `replace_engine` target-rebind diagnostics prescribe the complete workflow,
  including removing and re-registering the engine.

## TDD evidence

The first focused identity test was run before the implementation and failed
because `ProviderId`, `ProviderDeviceIdentity`, and the explicit registration
binding API did not exist. The implementation then added tests for identity
validation/traits, duplicate targets, same-target replacement, direct target
rebind rejection, stale routes and atomic publication, explicit route rebind,
registration order, pre-freeze remove/re-register ordering, target identity in
provider requests, two CPU domains, and distinct CPU provider kinds.

## Verification

- RED evidence: before implementation,
  `cargo test -p tenferro-runtime --lib provider_device_identity_validates_and_is_ordered_and_hashable`
  failed because the provider identity and explicit registration binding API
  were absent.
- `cargo fmt --all -- --check`: PASS.
- `git diff --check`: PASS.
- `cargo test -p tenferro-runtime --lib -q`: PASS (361 tests).
- `cargo test -p tenferro-runtime --test integration -q`: PASS (117 tests).
- `cargo test -p tenferro-runtime --doc -q`: PASS (397 doctests).
- `cargo doc -p tenferro-runtime --no-deps`: PASS.
- `cargo clippy -p tenferro-runtime --all-targets -- -D warnings`: PASS.
- `cargo test -p tenferro-cpu --lib -q`: PASS (502 tests), including two
  selected CPU domains and distinct Faer/BLAS provider identity coverage.
- `cargo clippy -p tenferro-cpu --all-targets -- -D warnings`: PASS.
- `cargo test -p tenferro-ad --lib -q`: PASS (57 tests).
- `cargo test -p tenferro-ad --test integration -q`: PASS (331 tests).
- `cargo clippy -p tenferro-ad --all-targets -- -D warnings`: PASS.
- `cargo test -p tenferro-gpu --lib`: PASS (0 tests; compile/link check).
- `cargo check -p tenferro-gpu --features webgpu`: PASS.
- `cargo clippy -p tenferro-gpu --lib -- -D warnings`: PASS, both default and
  `--features webgpu` builds.
- `python3 scripts/check-storage-ownership-contracts.py`: PASS, storage
  ownership contract ledger OK.

The GPU all-targets clippy command still reports three pre-existing
`drop_non_drop` lints in
`crates/tenferro-gpu/src/webgpu/tests/runtime_adapter.rs`; no A0.1 change
touches that test file. The production/library GPU checks pass.

## Remaining gates

- A0.2 remains pending: provenance-qualified event-domain identity and the
  scheduler-owned event-token bridge are not implemented here.
- B remains pending: typed CUDA device identity/discovery and CUDA selection
  migration are not implemented here.
- C remains pending: event-domain isolation and native token admission tests
  are not implemented here.
- Storage ownership, leases, allocation owners, and other later-phase changes
  remain intentionally untouched.

## Retrospective withdrawal and remediation RED

The acceptance of the candidate rooted at `5572bc9df73f5cfcf10ecd29120dbc714855e7ba`
is withdrawn. The remediation is based on parent `origin/main`
`0ee2d0dc2f8d21ff62ea682f90f34e4319108ace` after rebasing the preserved Phase 0
stack. This is a breaking redesign; no compatibility constructor, setter,
token, or fallback is retained.

The first RED test requires `PreparationOnly` versus
`Executable(ExecutableEngineContract)`, named ingress contract newtypes, and a
production reconfiguration path that assigns identity before freeze. The
current branch fails before test execution because those witness types and
constructors do not exist:

```text
cargo test -p tenferro-runtime --lib architectural_remediation --no-fail-fast
```

Observed RED evidence: unresolved imports for `EngineRegistrationState`,
`ExecutableEngineContract`, `InputIngressContract`,
`InputPlacementContract`, `InputSignatureContract`, `RuntimeInputContract`,
and `ResidentOutputContract`, plus missing
`EngineRegistration::preparation_only` and `EngineRegistration::executable`.
The test also exposed that its result type must carry snapshot and
reconfiguration errors rather than erase them into `RuntimeConfigError`.

The witness-propagation RED test was then added before refactoring frozen
storage. It requires a frozen executable slot to expose one complete witness
containing provider/device identity, execution context, executor, and event
driver. The focused command still fails at this point because
`EngineSnapshotView::executable_witness` and the remaining migrated test
constructors were not yet implemented:

```text
cargo test -p tenferro-runtime --lib frozen_executable_selection_returns_one_complete_witness --no-fail-fast
```

The observed failures included the missing `executable_witness` method, old
test-only `EngineRegistration::new` callsites, and the old
executor-bridge constructor function-pointer reference. This is the honest
RED boundary for the tagged frozen-witness redesign; no implementation claim
is made from this run.

The target state is a single immutable executable witness, a consuming
`CandidateConfig -> BoundCandidateConfig -> freeze` pipeline, and typed
preparation/execution failures before schedule admission. The existing
runtime/epoch/registration-qualified event provenance and deliberate transfer
host bridge remain valid and are not being replaced.

## Checkpoint: provider binding, candidate validation, and frozen witness

This checkpoint carries the preserved stack from parent `origin/main`
`0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`; the pre-checkpoint candidate was
`1e7df44e` (`test(runtime): add RED witness and binding contracts`). The
withdrawn accepted head remains only provenance:
`5572bc9df73f5cfcf10ecd29120dbc714855e7ba`. The redesign is intentionally
breaking: there are no compatibility constructors, setters, token APIs, or
foreign-provider fallbacks.

The executable witness is now assembled atomically as
`ProviderExecutableBinding -> ExecutableEngineContract`. The contract owns the
exact provider/device identity, execution-context identity, executor, event
driver, named `InputIngressContract`, and capabilities. Preparation-only
registrations use the separate `ProviderPreparationBinding` state. Runtime
selection and frozen storage propagate the tagged witness as one
`ExecutableEngineSnapshot`; execution and event-domain setup no longer split it
into independently looked-up executor/driver options. Candidate validation
consumes the private editor into `BoundCandidateConfig`, proving endpoint
existence, storage, provider binding, and transfer routes before freeze. Freeze
receives mandatory identities and bound routes and performs only total internal
assembly, not stale-route semantic revalidation.

### GREEN evidence

- `cargo test -p tenferro-runtime --lib --no-fail-fast`: PASS (378 tests).
- `cargo test -p tenferro-runtime --test integration --no-fail-fast`: PASS
  (118 tests), including no-launch invalid configuration and production
  snapshot-to-driver wiring behavior.
- `cargo test -p tenferro-runtime --doc --no-fail-fast`: PASS (402 doctests).
- `cargo test -p tenferro-cpu --lib --no-fail-fast`: PASS (502 tests).
- `cargo test -p tenferro-ad --lib --no-fail-fast`: PASS (57 tests).
- `cargo check -p tenferro-gpu --lib`: PASS.
- `cargo check -p tenferro-gpu --features webgpu`: PASS.
- `cargo fmt --all`: PASS.
- `git diff --check`: PASS.

The focused RED/GREEN evidence above supplements the earlier RED run in this
worklog; it does not claim that the later typed-error or retirement redesign is
complete. Hardware-only CUDA execution was not available in this environment;
CUDA compilation without hardware remains a later verification gate.

### D1 scope after proportionality refinement

This continuation completes only the remaining reachable execution-boundary
cases named by Phase 0 Task D1:

- Missing event-domain drivers, missing scheduled completions, duplicate
  transfer destinations, unsupported scheduled nodes, and missing prepared
  extension executors now retain small typed sources with the fields needed by
  callers/tests. Other explanatory `runtime_state` messages remain unchanged.
- Explicit scheduler drain remains the normal diagnostic path: it attempts
  every run in deterministic order and preserves execution as primary. `Drop`
  is only a one-shot best-effort emergency fallback when explicit drain was
  skipped; Drop-only errors may be suppressed because `Drop` has no `Result`.
  Explicitly caught provider panics retain normal Rust panic-payload drop
  semantics; Phase 0 adds no structured Drop sink.
- The two obsolete CUDA/WebGPU foreign-token source-substring tests were
  removed and their hardware-independent provider admission tests now assert
  typed rejection before launch. Unrelated historical source checks remain
  outside this task.

The two-device hardware execution test and any broader repository-wide audit
remain deferred to later phases.

## Follow-up: fallible eager-runtime construction

This Phase 0 continuation makes every public `EagerRuntime` constructor report
the existing typed `tenferro_ad::Result` instead of hiding
`RuntimeConfigError` behind a panic. It does not change eager backend privacy,
session-only extension execution, replacement/reconciliation, or fixed
provider IDs.

### RED evidence

Added `eager_runtime_constructors_return_typed_results` to the existing eager
runtime API integration contract. The test assigns each canonical constructor
to an exact function-pointer type returning
`tenferro_ad::Result<Arc<EagerRuntime>>`, including both CPU constructors and
the feature-gated CUDA/WebGPU constructors.

The required focused RED command was:

```text
cargo test -p tenferro-ad --test integration eager_runtime_constructors_return_typed_results --no-fail-fast
```

It failed before production edits with the expected `E0308` mismatches: the
three CPU constructors were `fn(...) -> Arc<EagerRuntime>` while the test
required `fn(...) -> Result<Arc<EagerRuntime>, tenferro_runtime::Error>`.

### Implementation and GREEN evidence

- `from_backend` and `from_backend_with_rules_and_cache` now return
  `Result<Self>` and map `RuntimeConfigError` through
  `Error::runtime_state_source`.
- Removed `build_eager_runtime_for_backend` and its fixed-configuration panic.
- Changed all public CPU, CUDA, and WebGPU constructors in place to return
  `Result<Arc<EagerRuntime>>`, with concrete `# Errors` rustdoc and no
  compatibility or `try_` constructors.
- Updated all workspace tests, benches, examples, current user-facing docs,
  and runnable doctests directly with the new result handling.

Fresh GREEN verification:

- `cargo test -p tenferro-ad --lib --no-fail-fast`: PASS (71 tests).
- `cargo test -p tenferro-ad --test integration --no-fail-fast`: PASS (332 tests).
- `cargo test -p tenferro-ad --doc --no-fail-fast`: PASS (137 doctests).
- `cargo test -p tenferro-ad --features cuda --test integration --no-run`: PASS.
- `cargo test -p tenferro-ad --features webgpu --test integration --no-run`: PASS.
- `cargo test -p tenferro-einsum --features autodiff --no-run`: PASS.
- `cargo test -p tenferro-fft --features autodiff --no-run`: PASS.
- `cargo test -p tenferro-linalg --features autodiff --no-run`: PASS.
- `cargo fmt --all -- --check`: PASS.
- `git diff --check`: PASS.

The first parallel dependent-crate compile attempt hit linker `SIGBUS` under
resource pressure; each requested no-run command passed when rerun
sequentially. The worktree's disposable Cargo target (57.3 GiB) was cleaned
after an initial trybuild run exhausted the filesystem; no source files were
removed.

### Residual next steps

- Keep the remaining Phase 0 provider/device identity and event-provenance
  follow-ups separate from this constructor slice.
- Preserve the typed source-chain contract when later reconciliation or fixed
  provider-ID work changes runtime configuration failure paths.
- Hardware-only CUDA/WebGPU execution remains environment-dependent and was not
  exercised by these compile-only feature gates.
