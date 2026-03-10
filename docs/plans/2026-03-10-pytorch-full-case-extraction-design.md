# PyTorch Full AD Case Extraction Design

## Goal

Expand `tensor-ad-oracles` from the current hand-picked v1 subset to the full set
of PyTorch linalg AD-relevant case families, and tighten family tolerances using
measured agreement between the PyTorch oracle and the finite-difference oracle.

## Problem

The current database covers only a small subset of PyTorch's linalg AD surface:

- `svd`
- `eigh`
- `solve`
- `cholesky`
- `qr`
- `pinv_singular`

PyTorch upstream defines a wider AD-relevant surface in
`torch/testing/_internal/opinfo/definitions/linalg.py`, including multiple
variants, wrappers, and observable-processing rules that affect the derivative
contract. Some important error families also live outside OpInfo, notably the
spectral gauge-ill-defined tests in `test/test_linalg.py`.

The current tolerance table is also mostly hand-set. It is checked in CI, but it
is not yet derived systematically from the actual residuals between the two
reference sources.

## Design Principles

1. Treat PyTorch OpInfo as the primary source of AD case families.
2. Do not scrape all of `test_linalg.py` literally; include only tests that
   define additional derivative semantics not already represented in OpInfo.
3. Keep the published DB machine-readable and stable; put extraction logic in the
   generator, not in handwritten JSON.
4. Use measured oracle agreement to tighten tolerances, but do not overfit to a
   single host by collapsing tolerances down to machine epsilon.

## Recommended Scope

### Included

All linalg `OpInfo` entries in upstream `linalg.py` that satisfy at least one of:

- `supports_forward_ad=True`
- `supports_fwgrad_bwgrad=True`

For each included OpInfo, extract:

- `name`
- `variant_test_name`
- `sample_inputs_func`
- `gradcheck_wrapper`
- `output_process_fn_grad`
- `gradcheck_fast_mode`
- explicit `toleranceOverride(...)` decorators

Also include explicit derivative/error families from `test/test_linalg.py` that
are not modeled by OpInfo, starting with the existing spectral
`gauge_ill_defined` cases.

### Excluded

These remain out of scope for the oracle DB:

- device/backend-specific skips and expected failures
- `out=` coverage
- inplace coverage
- JIT/schema/fake tensor/composite compliance infrastructure checks
- forward-value-only tests that do not change the derivative contract

## Extraction Model

### Upstream Inventory

Add an inventory step that walks PyTorch's `op_db` and emits a structured table
of AD-relevant linalg entries. The inventory record should include:

- upstream op name
- variant name
- sample-input function
- wrapper name
- observable-processing function identity
- dtype support relevant to v1 (`float64`, `complex128`)
- any explicit tolerance decorators attached to generic tests

This inventory becomes the source of truth for which families the generator must
cover.

### Family Mapping

Each upstream OpInfo entry maps to one or more DB families:

- `identity` for outputs with no extra gauge handling
- processed observable families for spectral ops
- explicit error families for ill-defined losses

For operations with `output_process_fn_grad`, the DB family boundary must follow
the processed observable, not the raw op output.

## Tolerance Design

### Sources

There are two tolerance-related sources:

1. Upstream PyTorch metadata
   - `toleranceOverride(...)`
   - `precisionOverride(...)`
2. Measured residuals in `tensor-ad-oracles`
   - `pytorch_jvp` vs `fd_jvp`
   - adjoint consistency residual

Upstream metadata should be stored as extracted provenance/hints, but not used as
the canonical DB tolerance for `float64`/`complex128` CPU cases. Many upstream
overrides exist for `float32`, `mps`, `cuda`, or noncontiguous-specific behavior,
which does not match the canonical oracle environment.

### Canonical Tolerance Rule

For each `(op, family, dtype)`:

1. Replay all cases and collect:
   - maximum absolute JVP disagreement
   - maximum relative JVP disagreement
   - maximum absolute adjoint residual
   - maximum relative adjoint residual
2. Compute proposed tolerances from the measured maxima:
   - multiply by a fixed safety factor
   - round up to the next power of ten
3. Tighten only if the current tolerance is more than ten orders of magnitude
   looser than the observed residuals

Recommended defaults:

- `safety_factor = 1e3`
- keep separate relative and absolute proposals
- apply a floor to avoid host-specific overfitting

This makes the rule stable, auditable, and easy to explain in CI:

- if current tolerance is reasonable, keep it
- if current tolerance is clearly too loose, tighten it

## CI Contract

The CI contract should have three layers:

1. Replay integrity
   - live `PyTorch JVP ~= FD JVP`
   - live adjoint consistency
2. Regeneration integrity
   - regenerated DB is semantically equal to published DB within case tolerance
3. Tolerance audit
   - current family tolerance is not more than `1e10` looser than observed
     residuals

The new tolerance audit should fail when a family remains obviously under-tight.

## Testing Strategy

### Unit Tests

- inventory parser for upstream OpInfo entries
- mapping from upstream observable-processing functions to DB family names
- tolerance proposal logic
- tolerance-audit threshold logic

### Integration Tests

- extract inventory from the pinned PyTorch checkout
- regenerate the expanded family set
- replay live oracle checks
- run tolerance audit over the full published DB

## Risks

### Risk: "All tests" is interpreted too literally

Mitigation:

- define "all tests" as "all AD-relevant linalg case families" in the design and
  README
- keep non-AD infrastructure tests out of the DB

### Risk: over-tightening tolerances

Mitigation:

- use measured maxima across the full family, not a single case
- apply safety factor and round-up-to-power-of-ten
- enforce floors

### Risk: upstream drift

Mitigation:

- keep torch version pinned
- store upstream extraction metadata in provenance
- run regeneration in CI on every PR and push to `main`

## Deliverables

1. Upstream inventory script for all AD-relevant PyTorch linalg OpInfo entries
2. Expanded case registry derived from that inventory
3. Generator support for all mapped families
4. Tolerance audit script and tests
5. Regenerated JSON DB
6. CI gate that fails on obviously over-loose tolerances
