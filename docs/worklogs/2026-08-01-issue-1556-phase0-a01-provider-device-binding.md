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

The target state is a single immutable executable witness, a consuming
`CandidateConfig -> BoundCandidateConfig -> freeze` pipeline, and typed
preparation/execution failures before schedule admission. The existing
runtime/epoch/registration-qualified event provenance and deliberate transfer
host bridge remain valid and are not being replaced.
