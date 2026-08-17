# Issue #1702: Einsum ellipsis

## Scope

Implemented flat NumPy-style ellipsis for concrete, eager, and traced einsum
surfaces, plus the public `EinsumAxis`/`EinsumNotation` programmatic form.
`EinsumSubscripts` remains the resolved integer-label payload consumed by the
planner, runtime cache, lowering, and AD rules.

## Decisions

- Resolve one ellipsis per term after ranks are known; align axes from the
  right and validate equal-or-one dimensions, including zero with one.
- Allocate collision-free internal ellipsis labels and canonicalize explicit
  labels for ellipsis notation so string and programmatic forms share cache and
  semantic identities.
- Broadcast singleton axes before dot/general contraction in the existing eager
  and standard-op builders; no backend-specific ellipsis kernel was added.
- Keep explicit `->` mandatory. Reject output ellipsis without an input
  ellipsis, multiple ellipses per term, invalid output labels, and parenthesized
  ellipsis. Traced ellipsis requires concrete input dimensions; existing
  ellipsis-free symbolic equality behavior remains unchanged.
- Carry an internal broadcast permission bit on semantic extension payloads so
  existing symbolic equality guards are not weakened for ordinary integer-label
  einsum.

## Evidence

- Resolver, parser, diagonal, broadcast, error, and zero-rank tests are in
  `crates/tenferro-einsum/src/ellipsis.rs` and `concrete_tests.rs`.
- Eager forward/programmatic and eager AD tests are in
  `tests/integration/eager_tensor.rs`.
- Traced forward execution and string/programmatic identity tests are in
  `tests/integration/trace_context_einsum.rs`.
- Runnable documentation is in `docs/guides/einsum.md` and
  `docs/tutorial-code/src/bin/math_snippets.rs`; README, llms, ndarray mapping,
  and all three skill mirrors were updated.

## Deferred

- Parenthesized `NestedEinsum` ellipsis and implicit-output equations remain
  outside the issue's initial scope.
- Fully symbolic traced broadcast disjunctions remain deferred; unresolved
  ellipsis on symbolic dimensions returns a typed planning error.
