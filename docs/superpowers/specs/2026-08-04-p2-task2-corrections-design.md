# P2 Task 2 Corrections Design

## Scope

This change closes only the six blocking corrections recorded on Issue #1558
before P2 Task 3. It adds the private identity/span validation boundary and its
executable tests. It does not implement owners, claims, reclamation, provider
imports, access preparation, persistent splitting, groups, bridges, or any
later-phase fixture.

The implementation is based on exact `origin/main` commit
`5bbd4a0613b3c801ce924cdc4339df2c77c1ab62`.

## Design

- `AllocationKey` remains domain-qualified diagnostic identity only.
- `RootResourceIdentity` pairs one private root provenance ID with one checked
  `RootResourceExtent`.
- `RootBoundSpan` stores the exact `RootResourceIdentity` from which it was
  derived. Child spans can be derived only from an existing bound span or its
  root identity; equal extents from different roots are not interchangeable.
- `ByteRange` and root/span construction perform checked end arithmetic before
  alignment, identity, or containment decisions. Relative-range overflow is
  therefore reported before a malformed alignment in the same request.
- Requested diagnostics use one `RequestedIdentity::{Raw, Keyed, Rooted}` sum
  type. Requested identity is untrusted metadata; resolved diagnostics retain
  the checked `RootBoundSpan`.
- Operation diagnostics keep one typed context/error envelope and do not expose
  pointers, provider handles, or write authority.

The only malformed metadata constructor is `#[cfg(test)]` and exists solely to
exercise validation precedence. It is not part of the production API and does
not create recovery, quarantine, retry, or repeated-access validation state.

## Verification

The correction gate is reproducible from the committed tree. Only the Task 2
identity/span test module is wired; later RED modules are not declared until
their owning phase is selected. Tests prove range-overflow precedence,
root-bound provenance, domain identity, alignment/containment, empty-range
semantics, requested-identity variants, and typed operation diagnostics.

The worklog records the exact candidate commit and tracked paths. No digest,
nonce, attestation, compatibility bridge, or source-scan-only proof is added.
