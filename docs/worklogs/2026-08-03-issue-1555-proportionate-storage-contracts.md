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
  storage/provider compatibility, and write injectivity are validated once at
  the prepared-access boundary.
- Contiguous traversal is typed slice access. Strided traversal performs only
  typed pointer access and precomputed stride/carry increments in its loop.
- Detached asynchronous work owns its resources through proven event
  retirement. If completion cannot be proven, a provider-private record retains
  the event, roots, and provider context permanently and exposes diagnostics,
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
