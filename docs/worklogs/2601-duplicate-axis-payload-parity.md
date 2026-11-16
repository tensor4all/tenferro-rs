# Duplicate-axis validation payload parity (#2601)

## Decisions

- **One role per argument family, no surface-specific literal.** A subset axis
  list (the reduction family, `reverse`) reports `DuplicateAxis { role: "axes" }`
  on every surface; a full permutation (`transpose`) reports `"permutation"`,
  owned by the tensor-core permutation validator that the eager path already
  reaches through `TensorLayout::transpose_view`. The issue's alternative —
  collapsing `transpose` into the axis-list role — was rejected because a
  permutation is a different argument kind, and adopting it would also change
  the eager/traced payloads and the tensor-core validators that linalg and the
  view APIs already rely on.
- **The concrete `transpose` gets its own signed-permutation front end**
  (`crates/tenferro-runtime/src/api/axes.rs::perm`) instead of reusing
  `tenferro_tensor::validate::validate_permutation_axes`. Reusing it would have
  changed the error *order* (it reports `RankMismatch` for a wrong-length perm
  before the eager and traced surfaces even look at the entries) and would have
  produced a different payload for a list that is both out of range and
  duplicated. `perm` and the axis-list helper `axes` share one three-step shape
  — range-check every entry, (permutations only) check the length, then scan for
  a repeat — which is the order `TensorLayout::transpose_view`,
  `validate_traced_perm` and the eager/traced `normalize_existing_axes` already
  use, verified in the source and with throwaway probes before writing them.
- **Wrong-length `transpose` payload unified as a consequence.** The concrete
  surface used to report `RankMismatch { expected: rank, actual: len }` from the
  CPU transposition validator for a wrong-length `perm`, while eager and traced
  reported `InvalidPermutationLength`. Since the new front end has to check the
  length to reproduce the other surfaces' ordering, it now reports
  `InvalidPermutationLength` too. This is a payload change on the same operation
  family and is recorded in the changelog; no test asserted the old payload.
- **The traced reduction metadata helper delegates instead of re-implementing.**
  `reduction_output_meta` had its own range/duplicate loop with the third role
  literal; it now calls the existing `validate_traced_axes`, so traced
  reductions and `reverse` share one validator.
- **A fourth spelling removed.** `tenferro-cpu`'s transposition validator used
  `role: "perm"`; it is unreachable through `CoreOps::transpose` once the front
  end validates, but it stays reachable from a direct engine call, so the
  literal now matches the family.
- **Deferred, not fixed.** The same-class role drift outside this family
  (`broadcast_in_dim`, `gather`, `dot_general` dim roles, linalg `"dim"`,
  `slice_builder` `"selection"`, tropical einsum labels) needs its own contract
  decision per family and would dominate review. Dropping the now-nearly-single
  valued `role` parameter from the public `validate_unique_axes` is a breaking
  public-API change and is likewise deferred.

## Verification conclusions and constraints

- `crates/tenferro-ad/tests/integration/duplicate_axis_payload_parity.rs`
  compares the full payload (`DuplicateAxis`/`AxisOutOfBounds`/
  `InvalidPermutationLength`, including `role`) across concrete, eager and
  traced for `reduce_sum`/`reduce_prod`/`reduce_max`/`reduce_min`/
  `reduce_sum_squares`/`reduce_mean`, `reverse` and `transpose`, with `[0, 0]`,
  a signed duplicate, an out-of-range axis, a wrong-length perm and the two
  precedence cases (range beats duplicate, length beats duplicate), plus an
  `i32` `reduce_mean` case where the axis payload must beat the unsupported
  dtype. It fails on `main` (`DuplicateAxis { role: "axes" }` vs
  `"axis"`/`"permutation"`, and the precedence cases) and passes after the
  change. A companion test runs one valid reduction, `reverse` and
  `transpose` on all three surfaces and compares the values, so the fix is
  shown not to move valid-input numerics.
- The workspace test suite (`cargo nextest run --workspace`, 3609 passed, 267
  skipped), the doctests of the touched crates, `cargo fmt --all --check` and
  the CI clippy group (`-D warnings -D clippy::missing_errors_doc -D
  clippy::missing_panics_doc`) pass.
- Not covered here: GPU/XLA backends were not exercised for these ops (they are
  not touched), and the C API mapping is unchanged because it ignores `role`
  (`docs/design/capi.md`).
- Review: `review: gpt-6.1-sol changes-requested-then-fixed` on the finished
  diff (pre-implementation plan review: `ollama-cloud/glm-5.3-flash`, three
  delta rounds). The post-review found two real gaps beyond the first pass,
  both fixed and covered by the parity test: the concrete layer reported a
  duplicate that preceded a later out-of-range entry (now every entry is range
  checked before any repeat, as on the eager and traced layers), and
  `reduce_mean` raised `UnsupportedDType` before an invalid axis list on the
  eager and traced composites (now the axis payload wins on all three).
- Deferred from that review as separate defects, both on the graph-construction
  interface rather than the three operation surfaces: `shape_infer` treats
  `Reverse` as shape-preserving and never validates its axes through
  `SemanticProgramBuilder::add_op`, and the internal transpose/reduction shape
  inference validators still differ from the public payloads for wrong-length
  and mixed out-of-range/duplicate lists.
