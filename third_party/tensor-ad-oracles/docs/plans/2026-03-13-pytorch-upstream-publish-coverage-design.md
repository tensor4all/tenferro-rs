# PyTorch Upstream Publish Coverage Design

## Goal

Make PyTorch's AD-relevant upstream cases the source of truth for the published
`tensor-ad-oracles` surface, and expand the database so every case family that
fits the current DB contract is materialized and tracked explicitly.

## Problem

The current database already uses PyTorch as its numerical oracle, but the
published family set is still partly hand-shaped. That creates blind spots:

- PyTorch upstream has complex `linalg.svd` success coverage, but the published
  DB currently has only real-valued SVD success cases plus one complex
  gauge-error case.
- It is not obvious which upstream AD-relevant families are already published
  and which are still missing.
- The existing provenance block records where a case came from, but not why a
  case exists or what bug surface it is intended to cover.

This makes it too easy to miss important publishable families, especially
complex and gauge-sensitive spectral cases.

## Design Principles

1. Treat PyTorch upstream as the source of truth for AD-relevant case families.
2. Publish every upstream family that can be represented by the current DB as a
   `success` or `error` case.
3. Use an inventory-first workflow so missing publishable families are visible
   before generator changes are made.
4. Keep the JSON case format stable and machine-readable; add only the minimum
   metadata needed to explain case intent.
5. Preserve the current DB distinction between raw operator rules and
   gauge-reduced observables.

## Scope

### Included

- AD-relevant PyTorch upstream families from:
  - `torch/testing/_internal/opinfo/definitions/linalg.py`
  - `torch/testing/_internal/common_methods_invocations.py`
- Explicit derivative/error families outside OpInfo that define additional AD
  contract and fit the current DB model, starting with
  `test/test_linalg.py::test_invariance_error_spectral_decompositions`
- All publishable dtypes already supported by the DB contract:
  - `float32`
  - `float64`
  - `complex64`
  - `complex128`

### Excluded

- Upstream infrastructure tests that do not define a new derivative contract
- Device/backend-specific xfails and skip logic as first-class DB families
- Placeholder inventory rows for upstream cases that cannot yet be published by
  the current DB contract

## Data Model Changes

### `provenance.comment`

Add an optional `comment` field to the case-level `provenance` block.

This remains free-form text in the first phase. Intended uses include:

- `from PyTorch OpInfo`
- `complex SVD success coverage`
- `complex SVD gauge-term audit target`

This field is not meant to replace structured provenance such as
`source_file/source_function`. It exists to capture case intent.

## Architecture

### 1. Upstream Inventory

Build or extend an inventory layer that normalizes every AD-relevant upstream
family into one machine-readable record. Each record should include:

- upstream op name
- upstream variant name
- inventory kind (`linalg` or scalar-style upstream source)
- source file and source function
- `output_process_fn_grad`
- `gradcheck_wrapper`
- AD capability flags
- dtype support relevant to the DB contract

This inventory becomes the canonical list of candidate publishable families.

### 2. Publish Mapping

Map each upstream inventory row to one or more DB case families only if the
result fits the existing `success`/`error` case model.

Examples:

- `linalg.svd` maps to:
  - `svd/u_abs`
  - `svd/s`
  - `svd/vh_abs`
  - `svd/uvh_product`
- explicit spectral error tests map to:
  - `svd/gauge_ill_defined`
  - `eigh/gauge_ill_defined`

The mapping layer remains responsible for translating upstream processed outputs
to DB observables.

### 3. Coverage Diff

Generate a human-readable report at:

- `docs/generated/pytorch-upstream-publish-coverage.md`

The report should compare:

- upstream inventory rows
- mapped publishable families
- currently materialized case families and dtypes

Its purpose is to answer:

- what exists upstream
- what is publishable
- what is already published
- what is still missing

### 4. Generator Expansion

Expand the generator so all mapped publishable families are actually
materialized. The key near-term requirement is to stop silently dropping complex
success coverage when upstream provides it.

In particular, the design must make missing complex spectral success families
visible and fixable. Complex SVD success coverage is the motivating example.

## Detectability Implications

This work does not by itself guarantee that downstream consumers such as
`tenferro-rs` replay every published family correctly. It does, however, remove
one major ambiguity: whether the DB itself omitted an upstream case family.

The coverage report and `provenance.comment` together should make it easy to
see when a case was added specifically to improve bug detectability, for example
for complex SVD gauge-sensitive AD behavior.

## Testing Strategy

### Unit Tests

- schema contract test for optional `provenance.comment`
- materialization tests that preserve the comment field
- upstream inventory and family mapping coverage tests
- report-generation tests for the publish coverage report

### Integration Tests

- materialize representative missing publishable families into a temporary case
  root
- verify generated cases include the expected dtype coverage
- validate schema and case integrity over regenerated output

## Risks

### Risk: inventory and published surface drift again

Mitigation:

- generate the coverage diff report from code, not by hand
- add tests requiring every upstream row to be either published or intentionally
  outside current publish scope

### Risk: complex spectral coverage still looks complete when it is not

Mitigation:

- include dtype-level coverage in the report
- add targeted tests asserting complex success coverage for key spectral families

### Risk: free-form comments become inconsistent

Mitigation:

- keep `comment` optional and descriptive in phase 1
- defer any controlled vocabulary until there is real usage pressure

## Deliverables

1. Optional `provenance.comment` in the case schema and materialization path
2. A generated publish-coverage report under `docs/generated/`
3. Inventory-to-publish mapping tests that operate at the upstream-family level
4. Generator changes that materialize all currently publishable upstream
   families
5. Regenerated cases that close known gaps, including complex SVD success
   coverage when upstream supports it
