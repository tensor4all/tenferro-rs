# Phase 3 A0 semantic-program checkpoint

Date: 2026-07-24

## Scope

This checkpoint implements only the accepted P3-A0 artifact boundary from
issue #1449 through normative correction v2.4. It intentionally leaves the
legacy graph/execution migration to P3-A1 through P3-A3.

## Implemented

- `tenferro_runtime::program` with opaque SSA values and binding keys.
- Exact, bounded, and unknown public symbolic metadata.
- Closed public core semantic operation enum and explicit extension
  effect/alias declarations. Undeclared extension semantics are rejected.
- Immutable operation/program views with bounded Debug output.
- Tensor bindings frozen separately from semantic structure.
- Failure-atomic consuming freeze and dependency import.
- Binding-preserving external semantic-transform contract.
- Cached SHA-256 semantic fingerprints and exact normalized equality.
- Exact collision checks for the transform cache bucket.
- Typed foreign-value, import, binding, structural, control-flow, query, and
  transform errors.

## Atomicity and identity evidence

- Failed finish returns neither a `SemanticProgram` nor `ProgramBindings`.
- Failed import leaves the destination builder observationally unchanged.
- Empty and duplicate roots retain their order.
- Imports preserve required inputs, bindings, metadata, guards, effects,
  aliases, placement, and provenance.
- Independently built and imported equivalent programs compare equal despite
  different builder nonces.
- Bindings and diagnostic provenance do not affect semantic identity.
- Operations, constants, metadata, guards, effects, aliases, placement, and
  ordered outputs do affect semantic identity.
- A forced compact-fingerprint collision still fails exact equality and
  occupies a separate transform-cache bucket.

## Verification

The A0 implementation was developed test-first. Focused RED/GREEN evidence
covered foreign tokens, undeclared extension semantics, binding finalization,
atomic import, structural identity, collision handling, transform locality,
binding preservation, and bounded/unknown shape guarantees.

The final checkpoint gate passed:

- runtime program, library, integration, and rustdoc tests;
- runtime all-target clippy with warnings denied;
- workspace formatting and documentation builds;
- repository documentation/public-error consistency scripts;
- workspace release tests and doctests;
- source-contract searches for raw identity/mutation escapes and forbidden
  program-module dependencies.

The release gate was run as
`cargo test --workspace --release --quiet` followed by
`cargo test --workspace --release --doc --quiet`; both commands exited
successfully with no failed tests.

## Remaining Phase 3 work

- P3-A1: the single private forward staging adapter and XLA/einsum migration.
- P3-A2: `TraceContext`, einsum tracing replacement, and pure semantic
  compilation.
- P3-A3: `GraphExecutor`/`CompiledGraph`, complete semantic AD migration for
  core, FFT, einsum, linalg, sparse, and tropical, production transform-cache
  wiring, and deletion of old public graph/execution/rule surfaces.

No Phase 3 completion or performance promotion is claimed by this checkpoint.
