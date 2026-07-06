# AD Transform Cache And DCE Design

## Goal

Finish the AD transform cache/memoization work and add a generic multi-output
DCE pass without hiding long-lived state in globals or thread locals.

## Requirements

- Eager and traced AD transforms must be able to share the same cache when they
  are built from the same explicit AD owner.
- Whole-graph cached transform results can be large, so the cache must have
  bounded defaults, user-configurable limits, retained-size introspection, and
  clear APIs.
- Cache stats must report logical retained bytes, not process RSS.
- `EagerRuntime` aggregate cache APIs must continue to include AD transform
  cache stats.
- `TracedTensor` extension methods may remain stateless; hidden process-global
  caches are out of scope.
- Multi-output DCE must prune unused output slots through an explicit legality
  hook. The optimizer must not assume that arbitrary multi-output operations can
  be partially evaluated.

## Cache Ownership

`AdContext` becomes the explicit owner of a shared AD transform cache handle.
It already owns traced AD extension rules and is passed to eager runtimes via
`EagerRuntime::with_*_and_ad_context`, so it is the natural owner for shared
eager/traced transform state.

The implementation will introduce a crate-local cache handle such as
`AdTransformCache`:

- `AdContext` stores one handle.
- `EagerRuntime::with_*_and_ad_context` clones the handle from `AdContext`.
- `EagerRuntime::with_*_backend` creates a private handle with the same default
  limits, preserving the current eager-only behavior.
- `TracedTensor::vjp`, `TracedTensor::jvp`, and `TracedTensor::grad` direct
  extension methods stay stateless. Cached traced transforms are used by
  `AdContext::{vjp,jvp,grad}`.

This keeps cache state explicit, avoids global caches, and lets users clear or
configure one AD context that backs both traced and eager calls.

## Cache API

Public cache API should stay owner-scoped:

- `AdContext::ad_transform_cache_limits() -> Result<AdTransformCacheLimits>`
- `AdContext::set_ad_transform_cache_limits(AdTransformCacheLimits) -> Result<()>`
- `AdContext::clear_ad_transform_caches() -> Result<()>`
- `AdContext::ad_transform_cache_stats() -> Result<CacheStats>`
- `AdContext::clear_caches() -> Result<()>`
- `AdContext::cache_stats() -> Result<AdContextCacheStats>`
- `EagerRuntime::ad_transform_cache_limits() -> Result<AdTransformCacheLimits>`
- `EagerRuntime::set_ad_transform_cache_limits(AdTransformCacheLimits) -> Result<()>`
- `EagerRuntime::clear_ad_transform_caches() -> Result<()>`

`EagerRuntime::cache_stats()` and `EagerRuntime::clear_caches()` remain the
aggregate eager APIs. When an eager runtime shares an `AdContext` cache, these
methods operate on the attached cache handle.

`AdTransformCacheLimits` should include:

- `max_entries: NonZeroUsize`
- `max_retained_bytes: Option<NonZeroUsize>`

The default should be bounded by both entry count and retained bytes. The exact
numbers can be adjusted with tests, but the design target is conservative:
large graphs should not grow unbounded just because transform outputs are
memoized.

## Cache Entries And Keys

The cache stores graph transform artifacts only. It must not store tensor data
buffers, backend allocations, or execution outputs.

Initial entry kinds:

- eager recorded graph linearization:
  `(recorded_graph_fingerprint, output_slots) -> Arc<LinearizedGraph<StdTensorOp>>`
- traced JVP linearization:
  `(resolved_graph_fingerprint, output_key, wrt_input_key, shape_metadata_fingerprint)
  -> Arc<LinearizedGraph<StdTensorOp>>`
- traced VJP transposed graph:
  `(resolved_graph_fingerprint, output_key, wrt_input_key, shape_metadata_fingerprint)
  -> cached linearized graph plus optimized transposed graph`

Keys use compact structural fingerprints plus exact identity fields that are
already available from computegraph value keys and AD metadata. They must not
format whole graphs into strings. Equality remains exact over key fields after
the fingerprint prefilter.

Retained bytes are tracked incrementally per entry. Eviction happens after
insert and removes least-recently-used entries until both `max_entries` and
`max_retained_bytes` are satisfied.

## Traced Metadata Safety

Traced AD rules can depend on shape metadata. A traced transform cache key must
include a metadata fingerprint for the values that can affect emitted AD graph
structure. If the metadata fingerprint cannot be computed safely for a transform
path, that path should bypass the cache rather than store an unsafe entry.

This is deliberately stricter than caching only the graph fingerprint. Reusing a
transform across incompatible symbolic/concrete shape metadata would be a
semantic bug.

## Generic Multi-Output DCE

The current optimizer already prunes unreachable operations from active tangent
outputs. The missing part is partial pruning of live multi-output operations
where only some output slots are required.

The new pass computes a live-output mask for every retained operation. For an
operation with more than one output, it asks the operation whether a reduced
operation is legal for that mask.

For extension ops, `ExtensionOp` gains a default method returning `None`, so
existing extensions remain source-compatible:

```rust
fn prune_outputs(&self, _live_outputs: &[bool]) -> Option<Arc<dyn ExtensionOp>> {
    None
}
```

The optimizer only rewrites when the hook returns a reduced operation whose
output count matches the number of `true` entries in `live_outputs`. The
optimizer computes the kept slot list from the same mask and remaps only those
old slots to the reduced operation's outputs. Unsupported operations stay
unchanged and remain correct.

## Documentation Updates

Update the AD pipeline/spec docs to replace the old "future cache" wording with
the actual owner/lifetime/limits/stats contract:

- `docs/architecture/ad-pipeline.md`
- `docs/spec/ad-contract.md`

Add a work log because this is a nontrivial AD/cache refactor with explicit
tradeoffs:

- `docs/worklogs/2026-07-06-ad-transform-cache-and-dce.md`

## Rejected Alternatives

### Global Or Thread-Local Cache

Rejected because repository cache rules require explicit top-level owners and
user-facing clear/configure/stats APIs. Hidden global state would also make
extension-rule ownership ambiguous.

### Put AD Cache On GraphCompiler

Rejected because eager AD does not require a graph compiler owner, and AD
transform cache semantics belong to AD rule/transform ownership rather than
runtime lowering.

### Prune Arbitrary Multi-Output Ops Without A Hook

Rejected because output slots can be semantically coupled. A generic pass can
identify unused slots, but only the operation family can say whether a smaller
operation is equivalent.
