# Issue #1555 proportionate storage-contract reconciliation

## Purpose

Reconcile the complete #1555 umbrella before implementation continues. Tenferro
is scientific-computing software, not a security boundary. The redesign keeps
Rust memory and aliasing soundness, numerical correctness, validated unsafe
access, explicit provider behavior, and asynchronous resource lifetime. It
removes mechanisms whose only purpose is defending against malicious trusted
tooling or reconstructing authority at runtime.

## Authoritative model

- Physical lifetime is direct `Arc<RootResource>` ownership.
- Exclusive write access originates only from Rust exclusive borrowing or a
  freshly created owner. IDs, handles, reference counts, and events never grant
  write authority.
- Bounds, checked layout arithmetic, dtype, exact root-bound span, alignment,
  storage/provider compatibility, and write injectivity are validated once
  when a checked descriptor is constructed. Prepared access consumes those
  retained proofs and performs only access-time provider work.
- Contiguous traversal is typed slice access. Strided traversal performs only
  typed pointer access and precomputed stride/carry increments in its loop.
- Detached asynchronous work owns its provider retirement bindings, event,
  roots, and context through proven retirement. If completion cannot be proven,
  a provider-private record retains them permanently and exposes diagnostics,
  not owners or a recovery protocol.
- Group descriptor slots are structural and append-only. AD handles directly
  retain read-only records/containers. Neither uses a global liveness registry
  or generation-based physical authority.
- Git commit plus tracked repository-relative path identifies verification
  evidence. Cryptographic digests, nonce/challenge handshakes, and runner
  attestation are not required inside the trusted repository/CI boundary.
- The cutover is direct. No compatibility alias, provider bridge, AD adapter,
  submission shim, conservative-synchronization adapter, or second storage API
  is accepted.

## Cross-phase removal audit

The issue bodies predate the proportionality decision and must be edited, not
merely superseded by comments.

| Owner | Remove from the live contract | Replace with |
|---|---|---|
| #1555 | `RootResourcePin`, `UseLease` authority language, generational descriptors, cancellation, corruption revalidation, provider bridge, artifact digest | direct root ownership, prepared-once access, structural descriptor slots, detached drain semantics, exact commit/path evidence |
| #1558 / P2 | required legacy provider bridge and bridge inventory; test-only corruption as repeated map/enqueue proof | direct root/span core; constructor/prepared-boundary invalid-descriptor rejection |
| #1559 / P3 | temporary accelerator bridge exceptions | one direct public owner/view/view-mut cutover with no compatibility surface |
| #1560 / P4 | quarantine/poison, retry/recovery registry, repeated map/enqueue validation, authority-bearing leases | prepared access, Rust borrow authority, event retirement, provider-private permanent retention only for unproven completion |
| #1561 / P5 | generation IDs, tombstones, global liveness/extraction registry | append-only group-local descriptor slots and structural `Arc` uniqueness |
| #1562 / P6 | identity rechecks as authority | consuming checked descriptor reinterpretation preserving the same root |
| #1563 / P7 | CUDA quarantine, per-binding static revalidation, bridge migration, lease authority | provider-ready prepared binding and detached event retirement |
| #1564 / P8 | WebGPU/Metal quarantine, repeated domain/range validation, bridge path, lease authority | the same prepared binding/retirement contract, with provider-specific mapping behavior |
| #1565 / P9 | cancellation state machine, quarantine outcomes, generational descriptors, recovery registry | pre-admission exact rejection; post-admission drain to completed/failed or ownerless `CompletionUnproven`; direct retained records |
| #1566 / P10 | safe APIs described primarily as lease-scoped | capability-borrowed prepared sessions; unsafe escaping interop documents caller retirement duties |
| #1567 / P13 | manifest/artifact digests and bridge-removal assumptions | exact clean Git commit/path evidence and proof that no bridge/shim was introduced |
| #1568 / P11 | panic quarantine and scoped quarantine poisoning | panic drains; completion-proven typed failure or ownerless permanent retention |
| #1569 / P12 | manifest digest identity | the same exact clean Git commit and tracked evidence paths as P11/P13 |

## Review discipline

Each gate is implemented by Luna(max), then checked by a Sol(medium) spec
review and a separate Sol(medium) quality review. Reviews must distinguish
soundness requirements from threat-model expansion. A finding may add a check
only when it names a reachable memory, aliasing, numerical, provider, or
lifecycle failure that the simpler model does not cover.

Before implementation promotion, scan the design document, ledger, and every
child issue body for the removed concepts above. Explicit statements that a
mechanism is forbidden are allowed; normative requirements to implement it are
not. Phase 13 repeats the same scan over source and live documentation and
deletes `HANDOFF-2026-07-25-tenferro-unification6-wip.md` plus inbound links.

## CI contract revision

The proportional review changed evidence contracts for deferred phases, while
the ledger checker previously treated every base/candidate comparison as an
implementation promotion. The ledger now records `registry.revision = 2` and
distinguishes a design-only contract revision from promotion. A revision is a
single monotonic step, preserves the graph, membership, all states, and every
active obligation identity, and may update only deferred evidence contracts.
It cannot be combined with activation. This lets the design phase correct
future obligations without weakening immutable evidence after implementation.

## Final consistency review

The final Sol review found and the design now resolves five cross-artifact
contradictions: G1/G4/G5 use one `CheckedLayout` and one
`PreparedRead`/`PreparedWrite` hierarchy; retirement always retains bindings,
event, roots, and provider context; #1557 describes the revision-only path;
obsolete I1--I10 references were replaced by the live umbrella's named rules;
and #1558 uses the same root-bound-claim shape as G1. The Phase 13
documentation row is inside the G6 table, and dynamic-boundary testing is
described as invalid constructor/input rejection rather than a corruption-hook
or repeated-validation protocol.
