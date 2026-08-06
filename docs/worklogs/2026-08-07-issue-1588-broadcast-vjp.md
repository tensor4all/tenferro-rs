# Issue #1588 broadcast VJP metadata validation

## Session summary

Fixed a public semantic VJP panic caused by malformed `BroadcastInDim`
dimension metadata. The transform now returns typed `UnsupportedMetadata`
diagnostics without changing valid identity, permuted, inserted-axis, or
singleton-axis behavior.

## Context reviewed

- Issue #1588's accepted plan and acceptance criteria;
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared common, Rust, numerical,
  documentation, and test rules;
- `crates/tenferro-ad/src/semantic_transform.rs` and its semantic-transform
  integration tests;
- semantic program metadata inference in `tenferro-runtime`;
- broadcast validation and the adjacent primitive AD transpose implementation
  in `tenferro-internal-ops`.

## Root cause and decision

Semantic program metadata inference takes a `BroadcastInDim` output shape from
the operation payload, so malformed `dims` can remain in a frozen program.
`transpose_broadcast` then indexed input and output shape vectors through that
unchecked mapping and used an infallible permutation lookup. An out-of-range or
overlong mapping could therefore panic through the public semantic VJP API.

After obtaining the input and cotangent shape plans, `transpose_broadcast` now
validates mapping arity, output-axis bounds, and uniqueness before indexing.
Every rejection uses `SemanticAdTransformError::UnsupportedMetadata` with role
`Vjp` and concrete offending values. The later permutation lookup remains
fallible and returns the same typed error category if its invariant is broken.

## Rejected alternatives

- `catch_unwind` would hide rather than remove unchecked public-boundary logic.
- A global validator registry would add state and indirection for one local
  metadata contract.
- A duplicate shape engine would create competing inference semantics.
- A compatibility fallback would silently reinterpret malformed metadata.
- Broader semantic-builder validation would change construction semantics
  outside the accepted VJP repair.

## RED to GREEN evidence

The first exact public regression used output rank 1 with `dims = [2]`. Before
production edits it failed with exit 101 at `output_shape.shape[2]`: `index out
of bounds: the len is 1 but the index is 2`. After validation was added, the
same exact test passed and asserted `UnsupportedMetadata`, role `Vjp`, and a
diagnostic identifying `dims[0] = 2` and output rank 1.

Table-driven follow-up cases also pass for an overlong mapping (`dims` length 2
versus input rank 1) and duplicate output axis (`dims[1] = 0`). Positive
identity and permuted mappings remain accepted.

## Verification

- Exact out-of-bounds regression: 1 passed, 0 failed.
- Exact arity/duplicate table regression: 1 passed, 0 failed.
- `cargo test -p tenferro-ad --test integration semantic_transform`: 36 passed,
  0 failed.
- `cargo test -p tenferro-ad --test integration shape_inference`: 14 passed,
  0 failed.
- `cargo test -p tenferro-ad --lib`: 66 passed, 0 failed.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p
  tenferro-ad --test integration semantic_transform'`: passed, including
  formatting, doc snippets, workspace and extension clippy, and the focused
  integration group.
- `cargo fmt --all --check` and `git diff --check`: passed.

## Neighborhood scan

The semantic JVP path re-emits the unary operation without indexing `dims`.
The adjacent primitive AD transpose path bounds-checks axes before indexing and
uses `Option` for permutation lookup, so it does not share the panic. Eager
broadcast construction already uses the existing broadcast validation helper.
No neighboring subsystem required a change.

## Residual risks

- Raw malformed broadcast metadata remains constructible until an active VJP
  traverses it, matching the accepted repair boundary.
- An inactive cotangent path skips validation, but it also returns before shape
  indexing and therefore cannot trigger this panic.
- Rank-list membership and position scans remain quadratic in tensor rank. They
  are rank-bounded, carry the required `INVARIANT` marker, and can become axis
  maps if unusually high-rank workloads make the cost measurable.
