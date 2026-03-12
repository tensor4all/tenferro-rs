# Dyadtensor Registry Redesign Design

## Goal

Remove the ad hoc generic registry machinery from `tenferro-dyadtensor` and
replace it with two explicit subsystems:

- a runtime-scoped holder dedicated to `RuntimeContext`
- a reverse-tape rule store dedicated to AD pullback registration and lookup

The end state should keep the current high-level AD surface intact while
eliminating the misleading "generic global context" abstraction and reducing
`thread_local + TypeId + Any` usage to the smallest boundary needed for typed
reverse-mode state.

## Why This Change

`extension/tenferro-dyadtensor/src/context.rs` currently exposes a generic
`set_global_context::<T>` / `with_global_context::<T>` API backed by
`thread_local!` storage keyed by `TypeId`. In practice, production code uses it
to hold only `RuntimeContext`.

`extension/tenferro-dyadtensor/src/reverse_tape/registry.rs` uses a similar
`thread_local + TypeId + Any` pattern for five separate registries:

- tensor pullback rules
- tensor bridge rules
- tensor-to-scalar bridge rules
- scalar mixed rules
- scalar pullback rules

These two mechanisms solve different problems:

- runtime selection
- reverse-mode bookkeeping

Keeping them under the same generic design pattern makes the API look more
general than it really is, weakens type safety, and duplicates registry logic.

## Non-Goals

- Do not change the mathematical behavior of AD rules.
- Do not redesign the `RuntimeContext` enum itself.
- Do not change `tenferro-prims` or `tenferro-linalg-prims` contracts.
- Do not add GPU execution support beyond current capability checks.

## Design Summary

### 1. Replace Generic Global Context With A Runtime-Only Holder

`context.rs` will stop being a generic `TypeId -> Any` registry. Instead, it
will become a dedicated runtime holder for `RuntimeContext`.

The public API should become conceptually:

- `set_default_runtime(ctx: RuntimeContext) -> RuntimeGuard`
- `with_default_runtime(f)`
- `try_with_default_runtime(f)` if needed by internal helpers/tests

The generic surface:

- `set_global_context`
- `with_global_context`
- `try_with_global_context`
- `GlobalContextGuard<C>`

will be removed from the public API.

This makes the public contract match the real use case and removes
`MissingGlobalContext` / `ContextTypeMismatch` as active design concepts.

### 2. Introduce A Dedicated Tape Rule Store

The reverse-tape registry should become a single `TapeRuleStore` per tape.

Conceptually:

- outer map: `TapeId -> TapeRuleStore`
- inner store: typed rule tables for tensor/scalar/bridge variants

`TapeRuleStore` should own:

- tensor pullback rules
- bridge pullback rules
- scalar bridge pullback rules
- scalar mixed pullback rules
- scalar pullback rules

The store may still use `TypeId` internally for typed buckets, but that
mechanism must be private to the store. Call sites should stop thinking in
terms of "global registry of erased states" and instead interact with
purpose-built registration helpers.

### 3. Keep The Public Registration API Small

The existing registration entrypoints can stay if they remain thin wrappers
over the new store:

- `register_rule`
- `register_bridge_rule`
- `register_scalar_bridge_rule`
- `register_scalar_mixed_rule`
- `register_scalar_rule`

Likewise, lookup/pullback helpers can stay if they are backed by the new store.

The point is not to rename everything; it is to remove the ad hoc storage model
underneath them.

### 4. Error Surface Cleanup

`Error::MissingGlobalContext` and `Error::ContextTypeMismatch` should be
removed. Runtime access should surface only:

- `Error::RuntimeNotConfigured`

Reverse-tape lookup failures should continue to surface as:

- `Error::InvalidAdTensor`
- `Error::InvalidAdScalar`

with messages about missing rules or tape-state mismatch where appropriate.

### 5. Test Strategy

The redesign needs three classes of tests.

- runtime holder tests
  verify scoped set/restore behavior and missing-runtime behavior
- reverse tape registry tests
  verify registration, lookup, bridging, and missing-rule behavior
- structural guard tests
  prevent reintroduction of generic `set_global_context::<T>`-style APIs and
  prevent a return to multiple ad hoc registries

As with the rest of the repository, test bodies should live in dedicated test
modules rather than inline in production files.

### 6. Documentation

Update:

- `extension/tenferro-dyadtensor` rustdoc exports
- `docs/design/supported-ops.md` if wording changes
- relevant AD/runtime docs

The docs should say that dyadtensor has:

- a scoped default runtime holder for builder execution
- a reverse-tape rule store for AD bookkeeping

They should not imply arbitrary global context storage.

## Implementation Order

1. Introduce runtime-only holder and migrate `runtime.rs` plus exports.
2. Replace reverse-tape registries with `TapeRuleStore`.
3. Migrate all registration and pullback call sites.
4. Update errors and tests.
5. Reread for similar ad hoc registry patterns.
6. Run full workspace verification.

## Success Criteria

- Production code no longer exports or uses `set_global_context::<T>`-style APIs.
- `context.rs` no longer stores arbitrary `TypeId -> Any` values.
- `reverse_tape/registry.rs` no longer keeps five separate thread-local maps.
- Runtime handling and reverse-tape bookkeeping are clearly separate concepts.
- Docs describe the new design directly.
