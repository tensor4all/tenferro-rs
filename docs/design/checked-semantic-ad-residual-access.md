# Checked Semantic AD Residual Access

## Status

Accepted design for issue #1711. Implementation requires an independent design-review verdict before code changes.

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

- Fake transpose and direct primal-VJP rules attempt an undeclared input and output value access; both return exact typed errors in ordinary and release-equivalent builds.
- Metadata access to the same undeclared positions succeeds and exposes dtype/shape without a `ProgramValue`.
- Bounds errors are typed and precede mask errors.
- Declared all-input Wilson-like rules, inactive inputs, and non-unit cotangent seeds retain numerical behavior.
- Source-contract checks ensure transpose/direct request types expose no raw primal value slice accessor.
- Retention tests confirm saved residual bindings include only declared mask positions.

## Non-goals

- No derivative-math or activity-analysis change.
- No automatic mask widening.
- No retain-all compatibility path or deprecated raw accessor.
- No panic/debug-assert enforcement.
- No restriction on `SemanticLinearizeRequest` primal access.

## Verification

Run focused semantic extension/transform tests in debug and release-equivalent profiles, all standard and nested extension tests, `tenferro-ad` doctests/clippy, retention/coverage checks, and combined PR gates.
