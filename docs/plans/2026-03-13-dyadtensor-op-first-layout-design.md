# Dyadtensor Op-First Layout Design

## Goal

Reorganize `extension/tenferro-dyadtensor` so that operation families are the
primary navigation path. A developer looking for `svd`, `einsum`, or `sum`
should be able to find the relevant primal surface, AD wiring, builders, and
tests under one local subtree instead of jumping across `api/`, `dyn_types/`,
`ad_value/`, and `reverse_tape/`.

## Problem

The current crate is reasonably split by file size, but the top-level layout is
still infra-first rather than operation-first.

- `api/` is too broad to communicate responsibility.
- `dyn_types/` looks like an implementation root even though it is support
  machinery.
- an operation such as SVD spans:
  `lib.rs` re-exports, `api/ad/eager_linalg.rs`, `api/ad_builders/...`,
  `ad_results.rs`, `reverse_tape`, and `tenferro-linalg`.
- the tree makes infrastructure details more visible than the feature being
  implemented.

This makes the crate harder to read, harder to extend with new ops, and easier
to accidentally grow in an ad hoc direction.

## Design Principles

- `op-first`: the main reading path should be "find the operation family".
- `shared core stays small`: runtime, tape, dynamic wrappers, and AD value
  definitions remain shared infrastructure, not top-level feature buckets.
- `generic execution stays intact`: high-level code must continue to route
  through `tenferro-prims` and `tenferro-linalg-prims` contracts rather than
  concrete CPU helpers.
- `no compatibility scaffolding`: this is a direct cutover, not a staged
  parallel tree.
- `tests stay out of production files`: only `#[cfg(test)] mod tests;` remains
  in source modules.

## Target Top-Level Layout

```text
src/
  lib.rs

  core/
    mod.rs
    value/
    node/
    dyn/
    convert/
    error.rs

  runtime/
    mod.rs
    context.rs
    dispatch.rs

  tape/
    mod.rs
    registry.rs
    scalar.rs
    tensor.rs

  structured/
    mod.rs
    layout.rs
    einsum.rs
    meta/

  ops/
    mod.rs
    primal.rs
    ad.rs
    scalar/
    reduction/
    einsum/
    linalg/
```

## Operation Layout

Each operation family owns its local story. For example:

```text
ops/linalg/svd/
  mod.rs
  primal.rs
  ad.rs
  builder.rs
  result.rs
  tests/
```

This keeps:

- public entrypoints,
- builder wiring,
- AD-specific wrapping,
- family-local tests

in one place.

Shared cross-family machinery remains in `core/`, `runtime/`, `tape/`, and
`structured/`.

## Module Mapping From Current Tree

- `ad_value/` -> `core/value/` and `core/node/`
- `dyn_types/` -> `core/dyn/` and `core/convert/`
- `api/runtime.rs`, `context.rs`, `runtime.rs` -> `runtime/`
- `reverse_tape/` -> `tape/`
- `structured/` stays `structured/`
- `api/primal_builders.rs`, `api/linalg_builders/*`, `api/ad/*`,
  `api/ad_builders/*`, `api/scalar_ad_builders/*` -> `ops/**`
- `api/ad_results.rs` -> family-local result modules where possible

## Public API Shape

`lib.rs` should stop acting as the only way to discover the crate. It should
still provide ergonomic re-exports, but the primary source of those re-exports
should become `ops::*`.

The intended public shape is:

- `tenferro_dyadtensor::ops::primal::*`
- `tenferro_dyadtensor::ops::ad::*`
- selected flat re-exports in `lib.rs` for convenience

The flat re-exports remain, but they should mirror an intelligible internal
structure instead of hiding it.

## Runtime And Tape Boundaries

`ops/*` should depend only on narrow shared helpers such as:

- `runtime::with_standard_runtime(...)`
- `runtime::with_linalg_runtime(...)`
- `tape::register_tensor_pullback(...)`
- `tape::register_scalar_pullback(...)`

Operation modules should not need to know tape storage layout or runtime holder
details.

## Generic Execution Requirement

This refactor must preserve the current execution layering:

- semiring/scalar/analytic paths go through `tenferro-prims`
- linalg paths go through `tenferro-linalg-prims`
- no new `backend::cpu::*`, `with_cpu_runtime(...)`, or
  `ensure_cpu_backend(...)` dependencies are allowed to appear in high-level
  dyadtensor code

## Migration Strategy

This is a direct cutover in one branch and one PR.

1. create the new module skeleton
2. move shared core modules
3. move scalar/reduction/einsum operations
4. move linalg operation families one by one
5. simplify re-exports and remove obsolete directories
6. update docs and structure tests

No compatibility mirror tree should remain at the end.

## Testing

The refactor is structural, so tests must protect both behavior and the new
layout.

- existing functional tests must stay green
- organization tests must be updated to assert the new module layout
- new structure tests should protect the op-first directories from collapsing
  back into monolithic `api/*` files
- no inline unit test suites in production files

## Documentation

Update these docs with the new mental model:

- `docs/api_index.md`
- `docs/design/architecture.md`
- `docs/design/autodiff.md`
- `docs/design/supported-ops.md`

The crate-level docs for `tenferro-dyadtensor` should describe:

- `core` as AD value infrastructure
- `runtime` as execution-runtime selection
- `tape` as reverse-mode bookkeeping
- `ops` as the main entry and maintenance surface

## Success Criteria

- a reader can find SVD AD implementation by staying within `ops/linalg/svd/`
  plus `tenferro-linalg`
- `dyn_types/` no longer exists as a top-level module
- `api/` is removed or reduced to a thin compatibility-free shell that only
  re-exports `ops`
- runtime/tape/shared helpers stay small and generic
- workspace verification passes unchanged
