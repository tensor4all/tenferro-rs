# NaryEinsum contraction-order cache for ExecOp dispatch

**Issue:** [#722](https://github.com/tensor4all/tenferro-rs/issues/722)
**Date:** 2026-04-18
**Status:** Approved design, pending implementation plan

## Problem

`execute_nary_einsum()` in `tenferro/src/exec.rs:623-717` calls `ContractionTree::optimize_with_options()` on every invocation. The optimization is an O(n!) search over contraction orders and dominates runtime for N-ary einsum with many tensors (84+ in the representative benchmark referenced by #664). The ExecOp dispatch path (reached e.g. during AD backward, iterative solvers, repeated eager evaluation) re-runs the full pipeline from scratch every call.

The `Engine::einsum_cache` at `tenferro/src/engine.rs:55` exists and is keyed by `(subscripts, shapes)`, but is only consulted by the TracedTensor build path in `tenferro/src/einsum.rs:275-286`. The ExecOp dispatch path cannot access it because `execute_nary_einsum` takes `&mut B: TensorBackend` and is reachable from the public free function `eval_exec_ir(&mut B, &ExecProgram, Vec<Tensor>)`, which has no Engine handle.

A secondary problem: the existing `einsum_cache` is **unbounded**. Workloads that process many distinct `(subscripts, shapes)` combinations (dynamic batch sizes, varying problem sizes) grow the cache without limit.

## Scope

**In scope:**
- Cache the optimized `ContractionTree` for N-ary einsum on the ExecOp dispatch path
- Make `Engine::einsum_cache` bounded (LRU) and share a single cache across both TracedTensor and ExecOp paths
- Preserve the existing public free-function API (`eval_exec_ir`, `eval_exec_ir_unsegmented`) without breaking changes

**Out of scope (follow-up):**
- Caching the fully lowered `ExecProgram` instead of just the `ContractionTree` — the tree optimization is the dominant cost; fragment/compile is small constant work. Revisit only if profiling justifies
- Making `Engine::compile_cache` bounded — same unbounded problem, but separate concern; track separately
- Migrating any other caches in the codebase

## Design

### What is cached

The optimized `Arc<ContractionTree>`, keyed by `(subscripts: String, shapes: Vec<Vec<usize>>)`. This matches the existing TracedTensor path exactly — cache format is unchanged, only the storage container and access points change.

### Storage

Replace the field on `Engine`:

```rust
// Before
pub(crate) einsum_cache: HashMap<(String, Vec<Vec<usize>>), Arc<ContractionTree>>,

// After
pub(crate) einsum_cache: LruCache<(String, Vec<Vec<usize>>), Arc<ContractionTree>>,
```

Using the `lru` crate (~500 LOC, canonical Rust LRU implementation). Default capacity: **256 entries** (typical `ContractionTree` is a few KB; 256 × few-KB ≈ under 1 MB).

### Public API additions

```rust
impl<B: TensorBackend> Engine<B> {
    // Existing
    pub fn new(backend: B) -> Self;  // uses DEFAULT_EINSUM_CACHE_CAPACITY = 256

    // New
    pub fn with_einsum_cache_capacity(backend: B, capacity: NonZeroUsize) -> Self;
    pub fn set_einsum_cache_capacity(&mut self, capacity: NonZeroUsize);

    // New — NaryEinsum-aware evaluator that threads the cache through
    pub fn eval_exec_ir(
        &mut self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>>;
}
```

`einsum_cache_len()` (existing) stays; its meaning (# retained entries) is preserved under the LRU.

### Threading the cache

`execute_nary_einsum()` gains a cache parameter:

```rust
fn execute_nary_einsum<B: TensorBackend>(
    backend: &mut B,
    inputs: &[&Tensor],
    subscripts: &str,
    mode: DispatchMode,
    einsum_cache: &mut LruCache<(String, Vec<Vec<usize>>), Arc<ContractionTree>>,
) -> Result<Tensor>
```

Inside, the `ContractionTree::optimize_with_options()` call is replaced with a cache lookup:

```rust
let key = (subscripts.to_string(), shapes.clone());
let tree = if let Some(cached) = einsum_cache.get(&key) {
    cached.clone()
} else {
    let tree = Arc::new(ContractionTree::optimize_with_options(
        &subs, &shape_refs, &ContractionOptimizerOptions::default(),
    )?);
    einsum_cache.put(key, tree.clone());
    tree
};
```

This mirrors the existing TracedTensor-path pattern at `einsum.rs:275-286`, modulo `get`/`put` naming.

The cache is threaded via a pair of functions at each layer: a public free function that preserves its current signature and an internal `_with_cache` variant that takes the cache explicitly.

- **Engine-aware path:** `Engine::eval_exec_ir(&mut self, program, inputs)` calls the internal `eval_exec_segmented_with_cache(backend, program, inputs, &mut self.einsum_cache)`, which calls `execute_nary_einsum(..., &mut cache)`. `TracedTensor::eval_with_inputs` (already takes `&mut Engine<B>`) and `eval_all` migrate to this path.
- **Public free-function path:** `eval_exec_ir`, `eval_exec_ir_unsegmented`, and `eval_exec_segmented` keep their existing public signatures. Each constructs a **local, ephemeral** `LruCache` (capacity 256) scoped to the single call and delegates to the internal `_with_cache` variant. This preserves API compatibility and still avoids duplicated work **within** a single call (e.g., multiple `NaryEinsum` instructions in the same program with identical subscripts+shapes). Repeated calls from code that holds no Engine pay the optimization cost each time — acceptable, since callers who want persistent caching should route through `Engine`.

### TracedTensor path migration

`einsum.rs:275-286` currently does:
```rust
let cache_key = (subscripts.to_string(), shapes.clone());
let tree = if let Some(cached) = engine.einsum_cache.get(&cache_key) {
    cached.clone()
} else {
    let tree = Arc::new(resolve_strategy(...)?);
    engine.einsum_cache.insert(cache_key, tree.clone());
    tree
};
```

Update to use `LruCache::get` / `LruCache::put`. `LruCache::get(&K)` takes `&mut self` (it touches recency), so the surrounding function already has `&mut engine` and no signature change is needed.

### Error handling

Pure-function cache lookup — no new error paths. Over-capacity insertion evicts the LRU tail silently. `NonZeroUsize` in the setter APIs prevents zero-capacity misconfiguration at the type level.

### Testing strategy

- **Correctness regression:** existing einsum tests must pass unchanged (hit/miss paths produce identical results).
- **New tests** in `tenferro/tests/`:
  - Cache hit detected: call the same einsum twice; assert `engine.einsum_cache_len() == 1` after both calls (a miss would produce 2 distinct entries on repeated distinct keys, but for a repeated identical key len stays at 1 — correctness regression tests cover the "produces same output" side).
  - LRU eviction: build an engine with `with_einsum_cache_capacity(2)`, call einsum with 3 distinct `(subscripts, shapes)` in order A, B, A, C. After the sequence, assert A and C are retained (A was recently used so survives eviction; C just inserted), B evicted. Check via `einsum_cache_len() == 2` plus probing `cached_contains(key)` through a test-only helper on `Engine`.
  - `set_einsum_cache_capacity` shrink: populate to 10 entries, set capacity=3, assert `einsum_cache_len() == 3`.
  - Free-function ephemeral cache: within a single `eval_exec_ir` call containing two `NaryEinsum` instructions with identical subscripts+shapes, both produce correct results. Hit-detection here requires a test-only probe on the ephemeral cache or a module-private counter; defer to follow-up if the correctness test is sufficient evidence.
- **Benchmark (smoke, not regression gate):** measure repeated-call latency on a ~10-tensor contraction. Expected: second call ≈ execution time only, first-call delta shows the skipped optimization cost.

### Observability

`Engine::einsum_cache_len()` stays as the primary introspection. Add:
- `Engine::einsum_cache_capacity() -> NonZeroUsize`

No hit/miss counters in this PR (YAGNI; can add if #728 benchmark work surfaces a need).

## Components touched

| File | Change |
|------|--------|
| `tenferro/Cargo.toml` | Add `lru` dependency |
| `Cargo.toml` (workspace) | Add `lru` to `[workspace.dependencies]` if used >1 crate (currently only `tenferro`, so direct dep is fine) |
| `tenferro/src/engine.rs` | Replace `HashMap` with `LruCache`; add `with_einsum_cache_capacity`, `set_einsum_cache_capacity`, `einsum_cache_capacity`, `eval_exec_ir` method |
| `tenferro/src/exec.rs` | Add cache parameter to `execute_nary_einsum`; add `eval_exec_ir_unsegmented_with_cache` internal variant; existing public `eval_exec_ir`/`eval_exec_ir_unsegmented` keep their signatures and delegate via an ephemeral `LruCache` |
| `tenferro/src/einsum.rs` | Update TracedTensor path to `LruCache::get`/`put` (trivial rename) |
| `tenferro/src/segment.rs` | Add `eval_exec_segmented_with_cache` internal variant; existing public `eval_exec_segmented` keeps its signature and delegates via an ephemeral `LruCache` |
| `tenferro/tests/` | New cache-behavior tests |

## Dependencies and risks

**New dependency:** `lru` crate (actively maintained, MIT/Apache-2.0, widely used). Transitive deps: `hashbrown` (already pulled in transitively via std).

**Risks:**
- `LruCache::get` takes `&mut self` (updates recency). Any code paths that had `&engine` must become `&mut engine`. The TracedTensor path already has `&mut engine`, so no impact there.
- Free-function `eval_exec_ir` paying repeated optimization cost is a known trade-off for preserving API compatibility. Users hitting this can migrate to `Engine::eval_exec_ir`.
- LRU semantics with `Arc<ContractionTree>` — evicting from the cache doesn't drop the tree if another `Arc` holds it (e.g. an in-flight evaluation). This is correct behavior, not a bug, but worth noting.

## Non-goals

- Do not cache the lowered `ExecProgram`
- Do not refactor `Engine::compile_cache` — separate concern
- Do not add async / concurrent access — `Engine` is single-threaded `&mut self` by design
