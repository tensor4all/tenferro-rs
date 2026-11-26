# Checked Semantic AD Residual Access

## Status

Implemented (issue #1711). Verified against `origin/main` at `e608e7ec` on 2026-11-27; the
"Implementation record" section below cites the code for each claim. One part of the original design was
wrong in the code, not in this document: `ResidualSpec::declares_input` answered `false` for every index
`>= 64`, so an `all_inputs()` rule (einsum with 65 operands) hit `UndeclaredResidualValue`. That is fixed by
#2585 (see "`ResidualSpec` arity").

## Goal

Semantic linear-transpose and direct primal-VJP rules can use a primal tensor value only when their `ResidualSpec` declares that input/output. Metadata remains available for every primal without retaining or exposing its `ProgramValue`.

## Request contract

For both `SemanticLinearTransposeRequest` and `SemanticPrimalVjpRequest`:

```rust
pub fn primal_input_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError>;
pub fn primal_output_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError>;
pub fn primal_input_meta(&self, index: usize) -> Result<&ProgramValueMetadata, SemanticAdError>;
pub fn primal_output_meta(&self, index: usize) -> Result<&ProgramValueMetadata, SemanticAdError>;
pub fn primal_input_count(&self) -> usize;
pub fn primal_output_count(&self) -> usize;
```

Value access first checks bounds, then requires `ResidualSpec::declares_input` or `declares_output`. An undeclared access returns a new typed `SemanticAdError::UndeclaredResidualValue` carrying the family, primal input/output kind, and index. An out-of-range access returns `SemanticAdError::PrimalIndexOutOfBounds` with the kind, index, and length. Define a public non-exhaustive `PrimalValueKind::{Input, Output}` diagnostic enum.

Metadata access checks only bounds. It returns an owned-at-request-construction metadata snapshot by reference; it never returns a `ProgramValue`. Requests own boxed input/output `ProgramValueMetadata` snapshots collected from the destination builder before rule dispatch. Metadata snapshots may allocate but do not retain tensor values.

Remove the unrestricted `primal_inputs()` and `primal_outputs()` slice accessors from transpose and direct primal-VJP requests after all in-tree and shipped extension rules migrate. Do not leave deprecated raw-value shims. `SemanticLinearizeRequest` keeps its raw primal access: linearization is the phase that computes values/residuals and is not constrained by a transpose residual declaration.

The two checked request structs cease to be `Copy`; their accessors borrow `&self`, allowing repeated access while preserving request ownership. Existing non-primal accessors (`op`, cotangents, activity, residuals, mask, provenance) also borrow `&self`.

## Implementation record

Every row was read against the code at `e608e7ec`, not carried over from the design.

| Claim | Where |
| --- | --- |
| `PrimalValueKind::{Input, Output}` is public and `#[non_exhaustive]` | `crates/tenferro-ad/src/semantic_extension.rs:77-84` |
| `PrimalIndexOutOfBounds { family_id, kind, index, len }` and `UndeclaredResidualValue { family_id, kind, index }` | `crates/tenferro-ad/src/semantic_extension.rs:120-140` |
| Transpose request: `primal_{input,output}_value`, `_meta`, `_count` on `&self` | `crates/tenferro-ad/src/semantic_extension.rs:307,334,360,388,412,428` |
| Direct primal-VJP request: the same six accessors | `crates/tenferro-ad/src/semantic_extension.rs:556,583,609,637,661,677` |
| Bounds are checked before the mask; the mask is `declares_input`/`declares_output` | `checked_primal_value`, `crates/tenferro-ad/src/semantic_extension.rs:1188-1215` |
| Metadata access checks bounds only | `checked_primal_metadata`, `crates/tenferro-ad/src/semantic_extension.rs:1217-1231` |
| Both requests own boxed metadata snapshots taken before rule dispatch | `crates/tenferro-ad/src/semantic_extension.rs:265-266,515-516,1071-1072,1126-1127` |
| The two checked request structs are not `Copy` (no derive) | `crates/tenferro-ad/src/semantic_extension.rs:261,511` |
| `SemanticLinearizeRequest` keeps raw `primal_inputs()`/`primal_outputs()` and stays `Copy` | `crates/tenferro-ad/src/semantic_extension.rs:187-210` |
| No raw primal slice accessor on the transpose/direct request types | source-contract test, `crates/tenferro-ad/tests/integration/semantic_extension.rs:254-276` |

### `ResidualSpec` arity

`ResidualSpec` keeps one bit per index for `0..64` plus one flag per side that declares every index `>= 64`
(`crates/tenferro-internal-ops/src/primitive_rules/mod.rs:101-111`). `all_inputs()`/`all_outputs()` (`:198-205`) delegate to
`with_all_inputs`/`with_all_outputs`, which set the flag (`:180-191`), and `declares_input`/`declares_output`
consult it (`:208-223`). An explicit single index
`>= 64` cannot be stored, so it is refused with an `assert!` in every build (`index_bit`, `:118-124`).
Rules that declare "all" therefore have no operand limit; rules that declare individual indices are limited
to `0..64`. Tests:
`crates/tenferro-internal-ops/src/primitive_rules/tests.rs:50-58` (refusal),
`crates/tenferro-einsum/tests/integration/many_operand_vjp.rs:26` (65-operand einsum VJP reads every residual).

## Dispatch and retention

Before dispatch, `SemanticExtensionRuleSet` snapshots metadata for all primal inputs/outputs using the validated destination builder. The rule's `ResidualSpec` remains the only tensor-retention authority. Checked access does not widen the mask, retain all primal values, or synthesize missing residuals.

Both linear-transpose and direct primal-VJP dispatch use the same private checked-access helpers so bounds/mask/error semantics cannot drift.

## Migration

Migrate einsum, FFT, linalg, sparse, tropical, tests, and the #1710 Wilson-like fixture:

- use checked value accessors only for indices declared in each rule's `ResidualSpec`;
- use metadata accessors/counts when only dtype/shape/arity is needed;
- collect checked values explicitly when a legacy adapter needs an ordered slice.

No rule may call a checked value accessor for an undeclared index, including inactive inputs.

## Tests

Found in the tree (checked 2026-11-27):

- Typed errors for an undeclared input and output access, through both transpose and direct primal-VJP dispatch:
  `crates/tenferro-ad/tests/integration/semantic_extension.rs:518` (identity assertions) and `:559` (undeclared access).
- Declared access, metadata access, counts, and the bounds error: `crates/tenferro-ad/src/semantic_extension/tests.rs:104`
  (bounds error asserted at `:147`, undeclared at `:160`).
- No raw primal value slice on transpose/direct requests: `crates/tenferro-ad/tests/integration/semantic_extension.rs:254`.
- Wilson-like multi-input fixtures: `crates/tenferro-ad/tests/integration/multi_input_traced.rs:20-21`.
- `ResidualSpec` arity: see "`ResidualSpec` arity" above.

- Bounds errors precede mask errors: the rule in `crates/tenferro-ad/tests/integration/semantic_extension.rs:194-237`
  declares `ResidualSpec::none()` and reads input 1 of a length-1 request, which is both out of bounds and
  undeclared, and requires `PrimalIndexOutOfBounds`; the dispatch test at `:559` runs it. Swapping the two checks in
  `checked_primal_value` (`semantic_extension.rs:1188-1215`) would fail it.
- Retention follows the mask, partially: `crates/tenferro-ad/src/eager/tests/residuals.rs:154`
  (`intermediate_inputs_are_retained_without_retaining_all_outputs`) asserts retained counts. It is not an
  "equals the declared mask positions" test, because `collect()` dedupes by semantic identity
  (`crates/tenferro-ad/src/eager/residuals.rs:45-56`).

Originally listed but **not** found: a release-profile-specific test run (no `--release` or `--all-features`
lane exists in `scripts/ci/run_profile.py`) and a test asserting saved residual bindings are exactly the
declared mask positions. Treat these as unverified until a test is cited here.

## Non-goals

- No derivative-math or activity-analysis change.
- No automatic mask widening.
- No retain-all compatibility path or deprecated raw accessor.
- No panic/debug-assert enforcement.
- No restriction on `SemanticLinearizeRequest` primal access.

## Verification

Run focused semantic extension/transform tests in debug and release-equivalent profiles, all standard and nested extension tests, `tenferro-ad` doctests/clippy, retention/coverage checks, and combined PR gates.
