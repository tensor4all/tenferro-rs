# Tensor AD Oracles Surface Gap Design

## Goal

Separate `tensor-ad-oracles` unsupported coverage into:

- oracle families already expressible with existing tenferro public APIs
- oracle families that require new public product surface before replay support

This design exists to prevent replay backlog work from hiding real product API
gaps.

## Context

The current vendored oracle coverage report in
`docs/generated/tensor-ad-oracles-support.md` mixes together several distinct
states:

- replay not implemented for an existing tenferro API
- oracle naming differs from tenferro naming, but the operation is already
  expressible
- tenferro does not yet expose a public API that matches the oracle family

Treating all three as one unsupported bucket makes roadmap decisions noisy. The
next replay-expansion phase should only target families already expressible
through the current tenferro public surface. Everything else should first be
tracked as product-surface work.

## Decision

Adopt a three-bucket classification for the published oracle database:

1. `Replay only`
2. `Needs public API issue`
3. `Needs explicit product decision`

The buckets are defined by the current tenferro public API surface, not by
internal backend helpers or by whether an operation could theoretically be
assembled from lower-level building blocks.

## Classification Rule

### Replay Only

Use this bucket when the published oracle family is already representable via
an existing public API with materially matching semantics.

Examples:

- `eigh` maps to `tenferro_linalg::eigen`
- `eigvals` maps to `tenferro_linalg::eig(...).values`
- `eigvalsh` maps to `tenferro_linalg::eigen(...).values`
- `svdvals` maps to `tenferro_linalg::svd(...).s`
- `matrix_norm` and `vector_norm` map to `tenferro_linalg::norm`
- `pinv_singular` maps to `tenferro_linalg::pinv`

These families should not spawn product implementation issues. They belong in
the replay backlog.

### Needs Public API Issue

Use this bucket when tenferro does not expose a public API with the same user
level contract as the oracle family.

Examples:

- `cond`
- `cross`
- `householder_product`
- `lu_solve`
- `matrix_power`
- `tensorinv`
- `tensorsolve`
- `vander`

These families should be tracked as product/API work before replay support is
attempted.

### Needs Explicit Product Decision

Use this bucket when the oracle family is close to existing tenferro APIs but
not obviously equivalent enough to suppress product discussion.

This bucket is intentionally small. Its purpose is to force a deliberate
decision instead of letting edge cases drift.

Initial decision bucket:

- `lu_factor`
- `multi_dot`
- `pinv_hermitian`
- `vecdot`

## Preliminary Family Mapping

### Replay Only

- `cholesky`
- `det`
- `diagonal`
- `eig`
- `eigh`
- `eigvals`
- `eigvalsh`
- `inv`
- `lstsq_grad_oriented`
- `lu`
- `matrix_norm`
- `norm`
- `pinv`
- `pinv_singular`
- `qr`
- `slogdet`
- `solve`
- `solve_triangular`
- `svd`
- `svdvals`
- `vector_norm`

### Needs Public API Issue

- `cholesky_ex`
- `cond`
- `cross`
- `householder_product`
- `inv_ex`
- `lu_factor_ex`
- `lu_solve`
- `matrix_power`
- `solve_ex`
- `tensorinv`
- `tensorsolve`
- `vander`

### Needs Explicit Product Decision

- `lu_factor`
- `multi_dot`
- `pinv_hermitian`
- `vecdot`

## Naming Policy

Oracle names do not require one-to-one tenferro API names.

A family counts as covered by existing public surface when:

- the operation is already public
- the returned observable can be materialized without adding new product API
- the semantic contract is close enough that a tenferro user would reasonably
  view it as the same operation

This is why `eigh` maps to `eigen`, and why `svdvals` does not require a new
`svdvals` symbol.

## Issue Strategy

Create one issue per missing product-surface family unless multiple oracle
families are clearly one feature slice.

Recommended grouping:

- `structured inverse/solve variants`: `cholesky_ex`, `inv_ex`, `solve_ex`
- `tensor construction ops`: `cross`, `householder_product`, `vander`
- `higher-level linear algebra ops`: `cond`, `lu_solve`, `matrix_power`,
  `tensorinv`, `tensorsolve`

Do not open implementation issues yet for the decision bucket. Open one short
triage issue or design note if necessary, then resolve each family into either
`Replay only` or `Needs public API issue`.

## Non-Goals

- implementing replay support in this phase
- changing the generated support report yet
- adding oracle-parity wrapper APIs just to match upstream naming
- classifying internal backend-only helpers as public coverage

## Success Criteria

This phase is complete when:

1. every currently unsupported oracle family is placed into one of the three
   buckets
2. missing public-surface families have implementation issues
3. the replay backlog only contains families already expressible via current
   tenferro public APIs
