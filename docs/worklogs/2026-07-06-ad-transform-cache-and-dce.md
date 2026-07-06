# AD Transform Cache And DCE

## Summary

This session finished the AD transform cache/memoization work and added a
generic, legality-hooked multi-output DCE path for AD transform graphs.

- `AdContext` now owns a bounded AD transform cache with public limits, stats,
  and clear APIs.
- Eager runtimes created from an `AdContext` share the same AD transform cache;
  direct eager runtimes keep a private cache.
- Context-driven traced JVP/VJP/grad use the shared cache, while direct traced
  extension methods remain stateless.
- The traced AD optimizer can prune unused output slots for extension ops that
  explicitly implement `ExtensionOp::prune_outputs`.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- shared tensor4all rules for Rust, docs/tests, performance, and numerical work
- GitHub issue context around #1256, #1303, and follow-up #1311
- `docs/architecture/ad-pipeline.md`
- `docs/spec/ad-contract.md`
- `crates/tenferro-ad/src/context.rs`
- `crates/tenferro-ad/src/eager.rs`
- `crates/tenferro-ad/src/traced.rs`
- `crates/tenferro-ad/src/traced/optimizer.rs`
- `crates/tenferro-ad/src/transform_cache.rs`
- `crates/tenferro-internal-ops/src/ext_op.rs`

## Decisions

- Put the shared cache on `AdContext`, not on a compiler or a global. The AD
  context already owns extension AD rules and is the one explicit owner shared
  by traced AD and eager runtimes built from that context.
- Keep direct `TracedTensorAdExt` methods stateless. Users who want shared
  traced cache behavior go through `AdContext`.
- Store graph transform artifacts only. Cache entries contain linearized and
  optimized graph structures, not tensor buffers, backend allocations, or
  execution outputs.
- Bound the cache by both entry count and logical retained bytes. The default is
  128 entries and 64 MiB of logical retained bytes.
- Make multi-output DCE opt-in through an extension hook. The optimizer can
  identify unused output slots generically, but only the operation family can
  state that a reduced operation is semantically equivalent.

## Implementation Notes

- `AdTransformCacheLimits` exposes entry and retained-byte limits.
- `AdContext` exposes `ad_transform_cache_limits`,
  `set_ad_transform_cache_limits`, `ad_transform_cache_stats`,
  `clear_ad_transform_caches`, `cache_stats`, and `clear_caches`.
- `EagerRuntime::with_*_and_ad_context` clones the `AdContext` cache handle.
  Eager aggregate cache stats and clear APIs include AD transform entries.
- Traced cache keys cover root graph structure, output key, `wrt` input key, and
  traced input aliases. Future rules that depend on metadata not represented in
  those fields must extend the key or bypass caching.
- `ExtensionOp::prune_outputs` defaults to `None`. When it returns a reduced
  op whose output count matches the live output mask, the optimizer remaps the
  retained slots to the new outputs.

## Verification Performed

- `cargo fmt --all`
- `cargo test -p tenferro-ad optimizer_prunes_unused_multi_output_slots_with_extension_hook`
- `cargo test -p tenferro-ad --test ad_optimizer`
- `cargo test -p tenferro-ad --test cache_management`
- `cargo fmt --all --check`
- `cargo test -p tenferro-internal-ops ext_op`
- `git diff --check`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --output-json /tmp/repository-rules-review.json`
- `cargo test -p tenferro-ad eager_runtime_ad_transform_cache_reuses_recorded_graph_linearization`
- `cargo test -p tenferro-ad --release`
- `cargo test -p tenferro-internal-ops --release`
- `cargo doc -p tenferro-ad -p tenferro-internal-ops --no-deps`

## Residual Risks

- Retained-byte accounting is an implementation estimate for cache payloads. It
  is intended for cache policy and introspection, not exact allocator usage.
- The traced transform cache key is structural over graph roots, requested
  output/input identity, and aliases. Shape-dependent future AD rules must add
  the relevant metadata to the key or skip cache insertion for that path.
