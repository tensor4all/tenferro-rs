# Tenferro Complex Linalg AD Rollout Design

**Date:** 2026-03-31

**Status:** Proposed and approved

## Goal

Roll out complex AD support for the remaining linalg operations in upstream
`tenferro::Tensor`, using an oracle-first process: math notes, oracle cases,
and replay support become the gating layer before downstream `frule` / `rrule`,
`LinearizedOp`, and public `Tensor` seams are enabled.

## Scope

This design covers complex AD rollout for:

- `inv`
- `cholesky`
- `pinv`
- `matrix_exp`
- `det`
- `slogdet`
- `norm`
- `lstsq`
- `eigen`
- `eig`

This design explicitly excludes:

- higher-order AD / HVP
- public API shape changes for these operators
- historical doc rewrites in old `docs/plans/` records

## Current State

The current public `Tensor + jvp(...)` surface already exposes these operators,
but complex AD support is incomplete.

There are two distinct situations:

1. Most target ops already have complex-capable primal implementations in
   `tenferro-linalg`, but their stateless `_frule` / `_rrule` and dynamic
   wrappers remain real-only.
2. `eig` is different: it is blocked first by the primal/output contract
   itself, not only by the AD rules.

The upstream oracle repository now contains the missing oracle-side work under
issue `tensor-ad-oracles#17`, including newly-added `matrix_exp` cases and an
explicit complex strategy note for `eig`.

## Design Decision

Adopt an **oracle-first, batch-by-output-contract** rollout.

The downstream implementation order is not by source file and not by public API
family. It is by the complexity of the output/cotangent contract.

### Batch A: complex in / complex out / same-domain

- `inv`
- `cholesky`
- `pinv`
- `matrix_exp`

These are the safest first wave because the public contract does not need to
change. Complex inputs produce complex outputs in the same tensor domain, so
the rollout is mostly:

- oracle replay
- `frule` / `rrule`
- `LinearizedOp::jvp/vjp`
- `Tensor` dtype gate removal

### Batch B: complex in / scalar or mixed structured out

- `det`
- `slogdet`
- `norm`
- `lstsq`
- `eigen`

These require more care because output and cotangent domains differ:

- `det`: complex scalar output
- `slogdet`: mixed output contract
- `norm`: real scalar output from complex input
- `lstsq`: structured result packaging
- `eigen`: real eigenvalues plus complex eigenvectors

Batch B should start only after Batch A is green.

### Batch C: special-case `eig`

`eig` remains its own wave. It is gauge-sensitive, has a distinct output
contract, and future complex-input enablement is not the same problem as
making the other complex-capable primals differentiable.

`eig` therefore gets a dedicated design/review step after Batch A and Batch B
are closed.

## Layering

Each batch must pass through the same four layers, in order.

### 1. `tenferro-linalg` stateless rules

`_frule` / `_rrule` are the first downstream gate because oracle replay calls
them directly. This layer owns the mathematical implementation.

### 2. Oracle replay

`tenferro-linalg/tests/oracle_db/replay.rs` is the second gate. A batch is not
ready until replay validates the new complex oracle cases.

### 3. Internal linearized seam

`tenferro-internal-ad-linalg` then lifts those stateless rules into
`LinearizedOp::jvp/vjp`.

If a `LinearizedOp` implementation is not a thin delegation to the underlying
`frule` / `rrule`, focused seam tests are required by `REPOSITORY_RULES.md`.

### 4. Public `Tensor` seam

Only after the first three layers are green do we remove the public real-only
gates in the `Tensor` facade and update README / rustdoc support tables.

## Testing Policy

Each batch must satisfy three independent test families.

### Oracle replay

- `tenferro-linalg/tests/oracle_db/replay.rs`

This validates math against the vendored oracle DB.

### Internal seam tests

- `internal/tenferro-internal-ad-linalg/tests/dyn_linalg_ops.rs`

This validates that the runtime packaging into `LinearizedOp::jvp/vjp` is
correct for complex tensors.

### Public seam tests

- `tenferro/tests/integration/linalg_surface_tests.rs`
- `tenferro/tests/integration/autograd_surface_tests.rs`

This validates that users can actually exercise the public `Tensor` API with
complex inputs.

## Docs Policy

The repo rules for this rollout are intentionally narrow:

1. README / rustdoc / examples must not claim beyond the current public
   surface.
2. If `LinearizedOp::jvp/vjp` is not a thin `frule` / `rrule` delegation,
   focused seam tests are required.
3. No ad hoc fixes that violate DRY / KISS / Layering.

This means:

- support tables must be updated batch-by-batch
- unsupported items must stay explicitly unsupported until their batch closes
- no temporary compatibility shims are allowed

## Summary

The rollout order is:

1. Batch A: `inv`, `cholesky`, `pinv`, `matrix_exp`
2. Batch B: `det`, `slogdet`, `norm`, `lstsq`, `eigen`
3. Batch C: `eig`

Each batch follows the same path:

1. oracle replay green
2. stateless complex `_frule` / `_rrule` green
3. internal `LinearizedOp` seam green
4. public `Tensor` seam green

This keeps the rollout incremental without introducing compatibility layers or
allowing the public documentation to outrun the real implementation.
