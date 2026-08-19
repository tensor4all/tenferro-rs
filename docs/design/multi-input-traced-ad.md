# Multi-Input Traced JVP and VJP

## Status

Accepted design for issue #1710. Implementation requires an independent design-review verdict before code changes.

## Goal

`AdContext` requests derivatives for multiple active traced leaf inputs through one compiled semantic source and one mask-based AD transform.

## Public API

Add methods to `AdContext`:

```rust
pub fn vjp_many(
    &self,
    output: &TracedTensor,
    wrts: &[&TracedTensor],
    cotangent: &TracedTensor,
) -> Result<Vec<Option<TracedTensor>>>;

pub fn jvp_many(
    &self,
    output: &TracedTensor,
    wrt_tangents: &[(&TracedTensor, &TracedTensor)],
) -> Result<Option<TracedTensor>>;
```

`vjp_many` preserves the requested order. A requested leaf absent from the source graph has `None` in its slot. Duplicate `wrt` leaves are allowed and repeat the same derivative result in each aligned slot.

`jvp_many` returns the one directional derivative produced by all supplied distinct active inputs. An absent/unreachable leaf is ignored; if none are active, the result is `None`. Duplicate `wrt` leaves are rejected from the raw pair list with typed graph-build `InvalidArgument` before compilation or reachability filtering because two tangent values for one semantic seed slot are ambiguous; callers explicitly add tangents before the call when that is intended.

An empty request is valid: `vjp_many` validates that the cotangent carries concrete data, then returns an empty vector without compiling or transforming; `jvp_many` returns `None` without running a transform.

## Implementation

Refactor the existing single-input helpers in `tenferro-ad/src/traced.rs` into many-input implementations:

1. Resolve every `wrt` to its leaf input key and validate tangent/cotangent attached data and metadata.
2. Compile `output` once with `compile_ad_source`.
3. Map requested keys to source input indices and build one union `active_inputs` mask.
4. Call `semantic_jvp_with_cache` or `semantic_vjp_with_cache` exactly once.
5. Bind all derivative seed inputs in one `derivative_tensor_from_program` construction.
6. For VJP, map derivative output indices back to the original requested order, returning `None` for unreachable/inactive slots and cloning the traced result for duplicate slots.

All graph/context and dtype/shape validation remains in the existing leaf-key, attached-value, semantic transform, and derivative tensor builders. The new helpers must not introduce separate compilation or execution caches.

Existing `jvp`, `jvp_optional`, `vjp`, and `vjp_optional` stay source-compatible and delegate to the many-input implementation with one entry. Their current inactive/error semantics remain unchanged.

## Tests

- Two and four distinct active inputs produce one JVP/VJP semantic transform; transform-cache/instrumentation observes one transformed derivative graph.
- A synthetic Wilson-like extension direct VJP emits one multi-output force node; compiled runtime instrumentation observes one force execution, not one per requested input.
- Each many-input VJP slot agrees numerically with the current single-input result.
- Unreachable inputs align to `None`; empty requests avoid transformation.
- Duplicate VJP requests repeat aligned results; duplicate JVP requests fail typed before transformation.
- Context mismatch and tangent dtype/shape mismatch remain typed.
- Existing single-input tests remain unchanged and pass through delegation.

## Non-goals

- No Jacobian materialization.
- No batching of independent cotangent seeds.
- No global AD context/cache.
- No synthesis of unsupported higher-order extension rules.
- No eager multi-input API in this issue; the accepted request targets traced convenience APIs.

## Verification

Run focused traced/semantic AD tests, extension multi-output instrumentation tests, numerical comparisons, `tenferro-ad` doctests and clippy, modified-file coverage review, and the combined PR gates.
