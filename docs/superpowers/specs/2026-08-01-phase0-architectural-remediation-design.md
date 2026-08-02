# Phase 0 retrospective architectural remediation

Issue: #1556 under #1555
Parent after rebase: `0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`
Accepted-but-withdrawn candidate: `5572bc9df73f5cfcf10ecd29120dbc714855e7ba`
Compatibility: none; the withdrawn candidate API is replaced in place.

## Design decision

The runtime must store either a preparation-only provider registration or one
complete executable witness. A collection of independently mutable execution
hooks cannot represent this contract. `EngineRegistration` therefore owns an
explicit sum state:

```text
PreparationOnly { capabilities }
Executable(ExecutableEngineContract)
```

`ExecutableEngineContract` atomically owns the executor, the event-domain
driver, the named `InputIngressContract`, and the capabilities used by the
registration. Provider identity/context and the cache contract are assembled
by the same constructor. The four ingress predicates are distinct semantic
newtypes (`InputPlacementContract`, `InputSignatureContract`,
`RuntimeInputContract`, and `ResidentOutputContract`) and are passed by name
through `InputIngressContract`; runtime-input and resident-output contracts are
not interchangeable values. There are no public setters for execution state.

CPU, CUDA, WebGPU, and future providers use one runtime assembly constructor.
Adapters calculate only provider-specific identity, event driver, backend
executor, ingress predicates, and capability values. Preparation may inspect
both registration states, but a prepared executable schedule may select only
an executable witness. Missing execution state is a typed preparation error,
before schedule admission or backend launch.

Candidate editing and freezing are separate ownership states. `CandidateConfig`
is a private mutable transaction editor. Consuming validation creates
`BoundCandidateConfig`, whose direct and extension records contain mandatory
`RegistrationIdentity` values and whose transfer records contain mandatory
source/destination provider bindings. The editor uses `New` and `Preserved`
identity variants; validation consumes those variants with a checked identity
allocator. Freeze accepts only `BoundCandidateConfig`, so it does not contain
late `Option` extraction or invalid transfer variants. Route binding is likewise
bound-only after validation.

Event-domain retirement is modeled with the `Pending`, `Retired`, and `Failed`
states. Explicit drain is the normal diagnostic path: it consumes `Pending`
before provider code, attempts every run in deterministic first-use order, and
returns every failure while preserving an execution failure as the primary
error. Explicit drain followed by provider `Drop` invokes retirement exactly
once.

`Drop` is a one-shot, non-panicking, best-effort emergency fallback only when
explicit drain was skipped. It does not retry terminal runs, and its errors may
be suppressed because `Drop` has no `Result` channel. Phase 0 does not define a
runtime-wide structured `Drop` diagnostic sink, panic-payload attestation, or an
untrusted-destructor threat model.

Typed machine-readable fields are required only where callers or tests need
them: missing event-domain drivers or scheduled completions, duplicate transfer
destinations, missing executable contracts, and unsupported scheduled nodes.
Other explanatory runtime-state messages remain ordinary source-preserving
runtime errors.

## Evidence boundary

RED tests first prove the public state transitions, typed fields and source
chains, retirement boundary, aggregate cleanup, no-launch behavior, and
production snapshot-to-driver wiring. The two obsolete foreign-token source
scans are replaced by direct typed rejection assertions; unrelated historical
source checks remain outside this task. The existing provenance contracts remain
unchanged: runtime/epoch/registration-qualified `EventDomainId`, same-domain
admission, the scheduler-owned transfer host bridge, returned-completion
validation, provider panic containment, no generic CUDA/WebGPU foreign fallback,
and no compatibility token API.

The worklogs record the withdrawn acceptance, this parent/candidate provenance,
the typed-state decisions, exact RED/GREEN commands, and the absence of a
compatibility path.
