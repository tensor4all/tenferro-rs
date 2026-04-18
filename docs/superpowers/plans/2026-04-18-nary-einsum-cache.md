# NaryEinsum contraction-order cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound `Engine::einsum_cache` via LRU and thread it through the ExecOp dispatch path so `execute_nary_einsum()` reuses optimized `ContractionTree`s on repeated `(subscripts, shapes)`, eliminating the per-call O(n!) optimization on the ExecOp path.

**Architecture:** Replace `HashMap` with `LruCache` on `Engine` (default capacity 256). Add internal `*_with_cache` variants of the three public execution entry points (`eval_exec_ir`, `eval_exec_ir_unsegmented`, `eval_exec_segmented`); keep the public signatures and have them delegate via an ephemeral `LruCache`. Add `Engine::eval_exec_ir` that threads the persistent cache. Migrate `TracedTensor::eval_with_inputs` / `eval_all` to the new Engine method.

**Tech Stack:** Rust, `lru` crate for LRU cache, existing `tenferro_einsum::ContractionTree`, `Arc`, `NonZeroUsize`.

**Spec:** `docs/superpowers/specs/2026-04-18-nary-einsum-cache-design.md`

**Branch:** `feat/722-nary-einsum-cache` (already checked out; spec commit already made)

---

## File Structure

| File | Role | Action |
|------|------|--------|
| `Cargo.toml` (workspace) | workspace deps | add `lru` |
| `tenferro/Cargo.toml` | crate deps | pull `lru` via `workspace = true` |
| `tenferro/src/engine.rs` | cache owner & Engine API | swap container, add capacity APIs, add `eval_exec_ir` method |
| `tenferro/src/exec.rs` | execution entry points + `execute_nary_einsum` | add `NaryEinsumCache` alias, `*_with_cache` variants, thread cache through `execute_nary_einsum` |
| `tenferro/src/segment.rs` | segmented evaluator | add `eval_exec_segmented_with_cache`, keep public signature |
| `tenferro/src/einsum.rs` | TracedTensor einsum build path | swap `HashMap` methods for `LruCache` methods |
| `tenferro/src/traced.rs` | TracedTensor `eval_with_inputs`, `eval_all` | route through `engine.eval_exec_ir` |
| `tenferro/tests/nary_einsum_cache.rs` | **new** — LRU behavior tests | create |

---

### Task 1: Add `lru` dependency

**Files:**
- Modify: `Cargo.toml` (workspace)
- Modify: `tenferro/Cargo.toml`

- [ ] **Step 1: Add `lru` to workspace dependencies**

Edit `/home/shinaoka/tensor4all/tenferro-rs/Cargo.toml`, inside `[workspace.dependencies]`, add (after `serde_json = "1"` around line 30):

```toml
lru = "0.12"
```

- [ ] **Step 2: Pull `lru` into `tenferro` crate**

Edit `/home/shinaoka/tensor4all/tenferro-rs/tenferro/Cargo.toml`, inside `[dependencies]`, add (after `thiserror.workspace = true` around line 34):

```toml
lru = { workspace = true }
```

- [ ] **Step 3: Verify it builds**

Run: `cargo build -p tenferro`
Expected: compiles successfully (no warnings about unused `lru`; it's only declared, not yet used).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml tenferro/Cargo.toml
git commit -m "feat: add lru crate dep for bounded einsum cache (#722)"
```

---

### Task 2: Swap `einsum_cache` container to `LruCache`

This is a mechanical refactor. No behavior change at existing call sites beyond bounded capacity. Default capacity 256 is large enough that existing tests don't hit eviction.

**Files:**
- Modify: `tenferro/src/engine.rs`
- Modify: `tenferro/src/einsum.rs:275-286`

- [ ] **Step 1: Replace the `einsum_cache` field and add a type alias**

In `tenferro/src/engine.rs`, replace the imports and `Engine` struct. Find the current code:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use super::exec::ExecProgram;
use tenferro_einsum::ContractionTree;
use tenferro_tensor::{cpu::CpuBackend, TensorBackend};
```

Replace the `HashMap` import (keep the rest) and add `LruCache` / `NonZeroUsize`:

```rust
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;

use super::exec::ExecProgram;
use tenferro_einsum::ContractionTree;
use tenferro_tensor::{cpu::CpuBackend, TensorBackend};

/// Key used for the N-ary einsum cache: `(subscripts, shapes)`.
pub(crate) type EinsumCacheKey = (String, Vec<Vec<usize>>);

/// LRU cache of optimized contraction trees keyed by einsum subscripts + input shapes.
pub(crate) type NaryEinsumCache = LruCache<EinsumCacheKey, Arc<ContractionTree>>;

/// Default capacity for `Engine::einsum_cache`.
///
/// Each `ContractionTree` is typically a few KB; 256 entries ≈ under 1 MB.
pub const DEFAULT_EINSUM_CACHE_CAPACITY: usize = 256;
```

Then find the field in `Engine`:

```rust
pub(crate) einsum_cache: HashMap<(String, Vec<Vec<usize>>), Arc<ContractionTree>>,
```

Replace with:

```rust
pub(crate) einsum_cache: NaryEinsumCache,
```

And update `Engine::new` — find:

```rust
pub fn new(backend: B) -> Self {
    Self {
        backend,
        compile_cache: HashMap::new(),
        einsum_cache: HashMap::new(),
    }
}
```

Replace with:

```rust
pub fn new(backend: B) -> Self {
    Self {
        backend,
        compile_cache: HashMap::new(),
        einsum_cache: LruCache::new(
            NonZeroUsize::new(DEFAULT_EINSUM_CACHE_CAPACITY)
                .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
        ),
    }
}
```

- [ ] **Step 2: Update the single call site in `einsum.rs`**

In `/home/shinaoka/tensor4all/tenferro-rs/tenferro/src/einsum.rs`, find the current cache read/write block (around line 274-286):

```rust
EinsumOptimize::Auto(opts) => {
    let cache_key = (subscripts.to_string(), shapes.clone());
    let tree = if let Some(cached) = engine.einsum_cache.get(&cache_key) {
        cached.clone()
    } else {
        let tree = Arc::new(resolve_strategy(
            EinsumOptimize::Auto(opts),
            &subs,
            &shape_refs,
        )?);
        engine.einsum_cache.insert(cache_key, tree.clone());
        tree
    };
    Ok(build_traced_from_tree(
        inputs,
        &subs,
        tree.as_ref(),
        &shapes,
    ))
}
```

Replace with:

```rust
EinsumOptimize::Auto(opts) => {
    let cache_key = (subscripts.to_string(), shapes.clone());
    let tree = if let Some(cached) = engine.einsum_cache.get(&cache_key) {
        cached.clone()
    } else {
        let tree = Arc::new(resolve_strategy(
            EinsumOptimize::Auto(opts),
            &subs,
            &shape_refs,
        )?);
        engine.einsum_cache.put(cache_key, tree.clone());
        tree
    };
    Ok(build_traced_from_tree(
        inputs,
        &subs,
        tree.as_ref(),
        &shapes,
    ))
}
```

Only `insert` → `put` changes. `LruCache::get` still returns `Option<&V>` and takes `&mut self`; the existing enclosing function already has `&mut engine`, so no further change.

- [ ] **Step 3: Verify the workspace builds and existing tests still pass**

Run: `cargo build --workspace`
Expected: compiles.

Run: `cargo test --workspace --release`
Expected: all existing tests pass. If any fail, the failure is not a test-expectation mismatch — investigate and fix.

- [ ] **Step 4: Commit**

```bash
git add tenferro/src/engine.rs tenferro/src/einsum.rs
git commit -m "refactor: use lru::LruCache for Engine::einsum_cache (#722)

Bounded capacity (default 256), LRU eviction.
Preserves behavior for existing call site in einsum.rs."
```

---

### Task 3: Add capacity-configuration API on `Engine`

**Files:**
- Modify: `tenferro/src/engine.rs`
- Test: `tenferro/tests/nary_einsum_cache.rs` (new)

- [ ] **Step 1: Write the failing tests**

Create `/home/shinaoka/tensor4all/tenferro-rs/tenferro/tests/nary_einsum_cache.rs`:

```rust
//! Tests for the LRU-backed Engine::einsum_cache.

use std::num::NonZeroUsize;

use tenferro::einsum::einsum;
use tenferro::{CpuBackend, Engine, TracedTensor};

fn run_matmul(engine: &mut Engine<CpuBackend>, rows: usize, cols: usize, mid: usize) {
    let a = TracedTensor::from_vec(
        vec![rows, mid],
        (0..rows * mid).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let b = TracedTensor::from_vec(
        vec![mid, cols],
        (0..mid * cols).map(|i| i as f64).collect::<Vec<_>>(),
    );
    let mut c = einsum(engine, &[&a, &b], "ij,jk->ik").expect("einsum");
    c.eval(engine).expect("eval");
}

#[test]
fn default_capacity_is_nonzero() {
    let engine = Engine::new(CpuBackend::new());
    assert_eq!(
        engine.einsum_cache_capacity(),
        NonZeroUsize::new(tenferro::engine::DEFAULT_EINSUM_CACHE_CAPACITY).unwrap(),
    );
}

#[test]
fn with_einsum_cache_capacity_sets_capacity() {
    let cap = NonZeroUsize::new(4).unwrap();
    let engine = Engine::with_einsum_cache_capacity(CpuBackend::new(), cap);
    assert_eq!(engine.einsum_cache_capacity(), cap);
}

#[test]
fn set_einsum_cache_capacity_shrinks_len() {
    let mut engine = Engine::with_einsum_cache_capacity(
        CpuBackend::new(),
        NonZeroUsize::new(10).unwrap(),
    );
    // Populate with 5 distinct einsum shapes (same subscripts, different shapes).
    // Same subscripts + different shapes => different cache keys.
    // Using the TracedTensor build path which writes to einsum_cache.
    for k in 1..=5 {
        run_matmul(&mut engine, 2, 2, k);
    }
    assert_eq!(engine.einsum_cache_len(), 5);
    engine.set_einsum_cache_capacity(NonZeroUsize::new(3).unwrap());
    assert_eq!(engine.einsum_cache_len(), 3);
    assert_eq!(engine.einsum_cache_capacity(), NonZeroUsize::new(3).unwrap());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p tenferro --test nary_einsum_cache --release`
Expected: FAIL (compile error — `Engine::with_einsum_cache_capacity`, `einsum_cache_capacity`, `DEFAULT_EINSUM_CACHE_CAPACITY` not public / not defined).

- [ ] **Step 3: Add the capacity APIs and re-export the constant**

In `tenferro/src/engine.rs`, find the `impl<B: TensorBackend> Engine<B>` block and add after `einsum_cache_len()`:

```rust
/// Construct a new engine with an explicit `einsum_cache` capacity.
///
/// # Examples
///
/// ```ignore
/// use std::num::NonZeroUsize;
/// use tenferro::{CpuBackend, Engine};
///
/// let engine = Engine::with_einsum_cache_capacity(
///     CpuBackend::new(),
///     NonZeroUsize::new(64).unwrap(),
/// );
/// ```
pub fn with_einsum_cache_capacity(backend: B, capacity: NonZeroUsize) -> Self {
    Self {
        backend,
        compile_cache: HashMap::new(),
        einsum_cache: LruCache::new(capacity),
    }
}

/// Current capacity of the einsum contraction-tree cache.
///
/// # Examples
///
/// ```ignore
/// use tenferro::{CpuBackend, Engine};
///
/// let engine = Engine::new(CpuBackend::new());
/// assert_eq!(engine.einsum_cache_capacity().get(), tenferro::engine::DEFAULT_EINSUM_CACHE_CAPACITY);
/// ```
pub fn einsum_cache_capacity(&self) -> NonZeroUsize {
    self.einsum_cache.cap()
}

/// Resize the einsum contraction-tree cache.
///
/// Shrinking below the current length evicts least-recently-used entries.
///
/// # Examples
///
/// ```ignore
/// use std::num::NonZeroUsize;
/// use tenferro::{CpuBackend, Engine};
///
/// let mut engine = Engine::new(CpuBackend::new());
/// engine.set_einsum_cache_capacity(NonZeroUsize::new(32).unwrap());
/// ```
pub fn set_einsum_cache_capacity(&mut self, capacity: NonZeroUsize) {
    self.einsum_cache.resize(capacity);
}
```

Also ensure the module `engine` is publicly accessible with the `DEFAULT_EINSUM_CACHE_CAPACITY` constant. Confirm `tenferro/src/lib.rs` has `pub mod engine;` or equivalent.

If `engine` is not already `pub mod`, make it so. Run:

```bash
grep -n "mod engine" tenferro/src/lib.rs
```

If it shows `pub mod engine;`, you're good. If it shows `mod engine;` (private), change to `pub mod engine;`. The constant and type aliases should already be reachable via `tenferro::engine::*`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p tenferro --test nary_einsum_cache --release`
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/engine.rs tenferro/src/lib.rs tenferro/tests/nary_einsum_cache.rs
git commit -m "feat: add einsum cache capacity API on Engine (#722)

with_einsum_cache_capacity / einsum_cache_capacity / set_einsum_cache_capacity.
Default capacity exposed as DEFAULT_EINSUM_CACHE_CAPACITY."
```

---

### Task 4: Add `contains` probe for tests, and an LRU eviction behavior test

**Files:**
- Modify: `tenferro/src/engine.rs`
- Modify: `tenferro/tests/nary_einsum_cache.rs`

- [ ] **Step 1: Write the failing LRU eviction test**

Append to `tenferro/tests/nary_einsum_cache.rs`:

```rust
#[test]
fn lru_eviction_preserves_recently_used() {
    let mut engine = Engine::with_einsum_cache_capacity(
        CpuBackend::new(),
        NonZeroUsize::new(2).unwrap(),
    );

    // Three distinct cache keys via shapes A, B, C.
    // Sequence: A (miss), B (miss), A (hit — now MRU), C (miss — evicts B).
    // Expected final state: A and C present, B evicted.

    let key_a = ("ij,jk->ik".to_string(), vec![vec![2, 3], vec![3, 2]]);
    let key_b = ("ij,jk->ik".to_string(), vec![vec![2, 4], vec![4, 2]]);
    let key_c = ("ij,jk->ik".to_string(), vec![vec![2, 5], vec![5, 2]]);

    run_matmul(&mut engine, 2, 2, 3); // A
    run_matmul(&mut engine, 2, 2, 4); // B
    run_matmul(&mut engine, 2, 2, 3); // A again — should be a hit, moves A to MRU
    run_matmul(&mut engine, 2, 2, 5); // C — cache full, evicts LRU (which is B)

    assert_eq!(engine.einsum_cache_len(), 2);
    assert!(engine.einsum_cache_contains(&key_a), "A should be retained (MRU)");
    assert!(!engine.einsum_cache_contains(&key_b), "B should be evicted");
    assert!(engine.einsum_cache_contains(&key_c), "C should be present (just inserted)");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tenferro --test nary_einsum_cache --release lru_eviction_preserves_recently_used`
Expected: FAIL — `einsum_cache_contains` method not defined.

- [ ] **Step 3: Add `einsum_cache_contains`**

In `tenferro/src/engine.rs`, inside the `impl<B: TensorBackend> Engine<B>` block, add after `einsum_cache_len`:

```rust
/// Returns `true` if the einsum cache contains a tree for `key`.
///
/// Does not modify LRU recency.
///
/// # Examples
///
/// ```ignore
/// use tenferro::{CpuBackend, Engine};
///
/// let engine = Engine::new(CpuBackend::new());
/// let key = ("ij,jk->ik".to_string(), vec![vec![2, 3], vec![3, 4]]);
/// assert!(!engine.einsum_cache_contains(&key));
/// ```
pub fn einsum_cache_contains(&self, key: &(String, Vec<Vec<usize>>)) -> bool {
    self.einsum_cache.contains(key)
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p tenferro --test nary_einsum_cache --release`
Expected: all four tests pass.

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/engine.rs tenferro/tests/nary_einsum_cache.rs
git commit -m "test: verify LRU eviction semantics for einsum_cache (#722)

Adds Engine::einsum_cache_contains probe (non-recency-touching) and
a test that exercises the sequence A,B,A,C with capacity=2, asserting
B is evicted while A (MRU) and C (newest) are retained."
```

---

### Task 5: Add `execute_nary_einsum` cache parameter and `_with_cache` internal variants

**Files:**
- Modify: `tenferro/src/exec.rs`

- [ ] **Step 1: Change `execute_nary_einsum` signature and use the cache**

In `tenferro/src/exec.rs`, find the current definition at line 623:

```rust
fn execute_nary_einsum<B: TensorBackend>(
    backend: &mut B,
    inputs: &[&Tensor],
    subscripts: &str,
    mode: DispatchMode,
) -> Result<Tensor> {
    use tenferro_einsum::{
        build_einsum_fragment, ContractionOptimizerOptions, ContractionTree, Subscripts,
    };
    // ...
    let tree = ContractionTree::optimize_with_options(
        &subs,
        &shape_refs,
        &ContractionOptimizerOptions::default(),
    )
    .map_err(|e| Error::ContractionError(format!("{e}")))?;
```

Replace the signature and the optimization call. Add these imports at the top of `exec.rs` (check if already present — `LruCache`, `Arc`, `ContractionTree`):

```rust
use crate::engine::NaryEinsumCache;
```

And the existing `use std::sync::Arc;` at the top of exec.rs — it's already there (line 1). Good.

Change the function signature to:

```rust
fn execute_nary_einsum<B: TensorBackend>(
    backend: &mut B,
    inputs: &[&Tensor],
    subscripts: &str,
    mode: DispatchMode,
    cache: &mut NaryEinsumCache,
) -> Result<Tensor> {
    use tenferro_einsum::{
        build_einsum_fragment, ContractionOptimizerOptions, ContractionTree, Subscripts,
    };
```

Replace the tree construction block:

```rust
let tree = ContractionTree::optimize_with_options(
    &subs,
    &shape_refs,
    &ContractionOptimizerOptions::default(),
)
.map_err(|e| Error::ContractionError(format!("{e}")))?;
```

With:

```rust
let cache_key = (subscripts.to_string(), shapes.clone());
let tree_arc = if let Some(cached) = cache.get(&cache_key) {
    cached.clone()
} else {
    let tree = Arc::new(
        ContractionTree::optimize_with_options(
            &subs,
            &shape_refs,
            &ContractionOptimizerOptions::default(),
        )
        .map_err(|e| Error::ContractionError(format!("{e}")))?,
    );
    cache.put(cache_key, tree.clone());
    tree
};
let tree = tree_arc.as_ref();
```

Note: `ContractionTree` is unused from the `use` statement if removed — keep the import since `Arc<ContractionTree>` uses the type.

Also find the nested calls in the same function around line 705-707:

```rust
DispatchMode::Unsegmented => eval_exec_ir_unsegmented(backend, &program, program_inputs)?,
DispatchMode::Segmented => {
    crate::segment::eval_exec_segmented(backend, &program, program_inputs)?
}
```

Replace with the `_with_cache` variants (added in Step 2 below) so any nested `NaryEinsum` reuses the same cache:

```rust
DispatchMode::Unsegmented => {
    eval_exec_ir_unsegmented_with_cache(backend, &program, program_inputs, cache)?
}
DispatchMode::Segmented => {
    crate::segment::eval_exec_segmented_with_cache(backend, &program, program_inputs, cache)?
}
```

- [ ] **Step 2: Add `eval_exec_ir_unsegmented_with_cache` and update public wrappers**

In `tenferro/src/exec.rs`, replace the current public `eval_exec_ir` and `eval_exec_ir_unsegmented` with:

```rust
pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut cache = NaryEinsumCache::new(
        std::num::NonZeroUsize::new(crate::engine::DEFAULT_EINSUM_CACHE_CAPACITY)
            .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
    );
    crate::segment::eval_exec_segmented_with_cache(backend, program, inputs, &mut cache)
}
```

and

```rust
pub fn eval_exec_ir_unsegmented<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut cache = NaryEinsumCache::new(
        std::num::NonZeroUsize::new(crate::engine::DEFAULT_EINSUM_CACHE_CAPACITY)
            .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
    );
    eval_exec_ir_unsegmented_with_cache(backend, program, inputs, &mut cache)
}
```

Then add the internal variant, which is the current body of `eval_exec_ir_unsegmented` plus a cache parameter. Find the existing body (lines 291-onwards) and extract it verbatim into:

```rust
pub(crate) fn eval_exec_ir_unsegmented_with_cache<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    cache: &mut NaryEinsumCache,
) -> Result<Vec<Tensor>> {
    // ... exact body that currently lives in eval_exec_ir_unsegmented ...
}
```

Inside that body, find the single call site that invokes `execute_nary_einsum` (around line 537):

```rust
let result = execute_nary_einsum(backend, &inputs, subscripts, mode)?;
```

Change to:

```rust
let result = execute_nary_einsum(backend, &inputs, subscripts, mode, cache)?;
```

Also find `execute_ffi_instruction` — it's called in the unsegmented path with `DispatchMode::Unsegmented`. Check if it calls `execute_nary_einsum` internally. Use:

```bash
grep -n "execute_nary_einsum\|execute_ffi_instruction" tenferro/src/exec.rs
```

Any caller of `execute_nary_einsum` in `exec.rs` must take the cache param. Update signatures along the chain so the cache flows from `eval_exec_ir_unsegmented_with_cache` down.

- [ ] **Step 3: Build & check**

Run: `cargo build -p tenferro`
Expected: compiles. If not, follow the error chain to propagate the `cache` parameter through any remaining internal callers.

- [ ] **Step 4: Commit**

```bash
git add tenferro/src/exec.rs
git commit -m "feat: thread NaryEinsumCache through exec dispatch (#722)

execute_nary_einsum now consults a caller-provided LruCache before
re-running ContractionTree::optimize_with_options. Public free
functions eval_exec_ir / eval_exec_ir_unsegmented keep their
signatures and construct an ephemeral LruCache of default capacity."
```

---

### Task 6: Add `eval_exec_segmented_with_cache` and keep public signature

**Files:**
- Modify: `tenferro/src/segment.rs`

- [ ] **Step 1: Replace public `eval_exec_segmented` with a thin wrapper and add the internal variant**

In `tenferro/src/segment.rs`, find the current public function at line 141:

```rust
pub fn eval_exec_segmented<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let segments = segment_exec_program(program);
    let mut slots = initialize_slots(program, inputs);
    // ... body ...
}
```

Replace with a wrapper:

```rust
pub fn eval_exec_segmented<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut cache = crate::engine::NaryEinsumCache::new(
        std::num::NonZeroUsize::new(crate::engine::DEFAULT_EINSUM_CACHE_CAPACITY)
            .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
    );
    eval_exec_segmented_with_cache(backend, program, inputs, &mut cache)
}

pub(crate) fn eval_exec_segmented_with_cache<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    cache: &mut crate::engine::NaryEinsumCache,
) -> Result<Vec<Tensor>> {
    let segments = segment_exec_program(program);
    let mut slots = initialize_slots(program, inputs);
    // ... existing body ...
}
```

The body of `eval_exec_segmented_with_cache` is the exact body of the old `eval_exec_segmented`, but the single call to `execute_ffi_instruction(backend, &mut slots, inst, DispatchMode::Segmented)?;` (line 189) must thread the cache through to any `execute_nary_einsum` call it transitively makes.

- [ ] **Step 2: Thread `cache` through `execute_ffi_instruction` (and any other internal dispatch fn)**

Locate `execute_ffi_instruction` — it's in `tenferro/src/exec.rs`. Use:

```bash
grep -n "fn execute_ffi_instruction" tenferro/src/exec.rs
```

Update its signature to take `cache: &mut NaryEinsumCache` and propagate to any inner call to `execute_nary_einsum`. Every caller of `execute_ffi_instruction` (both in `exec.rs` and `segment.rs`) must then pass the cache.

- [ ] **Step 3: Build & run the workspace tests**

Run: `cargo build --workspace`
Expected: compiles.

Run: `cargo test --workspace --release`
Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add tenferro/src/exec.rs tenferro/src/segment.rs
git commit -m "feat: add eval_exec_segmented_with_cache, thread cache via FFI dispatch (#722)

Public eval_exec_segmented keeps its signature, constructs an
ephemeral cache. Internal _with_cache variant takes the cache by
ref so Engine-driven eval can reuse a persistent cache."
```

---

### Task 7: Add `Engine::eval_exec_ir` method and route `TracedTensor::eval`-family calls through it

**Files:**
- Modify: `tenferro/src/engine.rs`
- Modify: `tenferro/src/traced.rs:22, 602, 1734`

- [ ] **Step 1: Add `Engine::eval_exec_ir`**

In `tenferro/src/engine.rs`, add to `impl<B: TensorBackend> Engine<B>`:

```rust
/// Evaluate an `ExecProgram` through this engine, reusing the persistent
/// `einsum_cache` for any `NaryEinsum` ops encountered in the program.
///
/// # Examples
///
/// ```ignore
/// use tenferro::{CpuBackend, Engine};
/// use tenferro::exec::ExecProgram;
///
/// let mut engine = Engine::new(CpuBackend::new());
/// // let outputs = engine.eval_exec_ir(&program, inputs)?;
/// ```
pub fn eval_exec_ir(
    &mut self,
    program: &ExecProgram,
    inputs: Vec<tenferro_tensor::Tensor>,
) -> crate::error::Result<Vec<tenferro_tensor::Tensor>> {
    crate::segment::eval_exec_segmented_with_cache(
        &mut self.backend,
        program,
        inputs,
        &mut self.einsum_cache,
    )
}
```

- [ ] **Step 2: Migrate `traced.rs` call sites**

In `tenferro/src/traced.rs`:

Find line 22:
```rust
use super::exec::eval_exec_ir;
```

Since we're migrating, either remove the import (if no remaining use) or keep it if another caller remains. After migration, if unused, remove. For now, keep it — we'll remove at the end if `cargo check` warns.

Find the call at line 602:
```rust
let mut results = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors)?;
```

Replace with:
```rust
let mut results = engine.eval_exec_ir(&cached_exec, input_tensors)?;
```

Find the call at line 1734:
```rust
let results: Vec<Tensor> = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors)?;
```

Replace with:
```rust
let results: Vec<Tensor> = engine.eval_exec_ir(&cached_exec, input_tensors)?;
```

- [ ] **Step 3: Clean up unused import if the above removed all uses**

Run: `cargo build -p tenferro 2>&1 | grep -E "unused import.*eval_exec_ir"`

If it warns about unused `eval_exec_ir`, remove the import at line 22. Otherwise leave it.

- [ ] **Step 4: Build & test**

Run: `cargo build --workspace`
Expected: compiles cleanly (no warnings).

Run: `cargo test --workspace --release`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/engine.rs tenferro/src/traced.rs
git commit -m "feat: add Engine::eval_exec_ir method (#722)

Routes TracedTensor::eval_with_inputs and eval_all through the
engine so Engine::einsum_cache is reused across calls for
NaryEinsum ops in the program."
```

---

### Task 8: Verify cache hit on the ExecOp dispatch path

**Files:**
- Modify: `tenferro/tests/nary_einsum_cache.rs`

This test verifies the end-to-end cache-hit behavior on the ExecOp path (not just TracedTensor path).

- [ ] **Step 1: Write the failing test**

Append to `tenferro/tests/nary_einsum_cache.rs`:

```rust
/// When an ExecProgram containing a NaryEinsum instruction is evaluated twice
/// through Engine::eval_exec_ir with identical inputs, the second call must hit
/// the cache — `einsum_cache_len()` stays at 1 after both runs.
#[test]
fn nary_einsum_on_exec_path_hits_cache() {
    use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
    use tenferro_tensor::{DType, TypedTensor};

    let mut engine = Engine::new(CpuBackend::new());

    // Build an einsum with at least one symbolic-shape input so the graph keeps
    // a NaryEinsum op (not decomposed at build time).
    let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let b = TracedTensor::from_vec(vec![3, 4], vec![1.0_f64; 12]);
    let mut c =
        tenferro::einsum::einsum(&mut engine, &[&a, &b], "ij,jk->ik").expect("einsum");

    // Concrete input for the symbolic leg.
    let a_concrete = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    // First eval: miss -> inserts one entry.
    c.eval_with_inputs(&mut engine, &[(&a, a_concrete.clone())])
        .expect("eval 1");
    let len_after_first = engine.einsum_cache_len();
    assert_eq!(len_after_first, 1, "expected one cache entry after first eval");

    // Second eval with the same concrete input: must hit the cache.
    c.eval_with_inputs(&mut engine, &[(&a, a_concrete)])
        .expect("eval 2");
    let len_after_second = engine.einsum_cache_len();
    assert_eq!(
        len_after_second, 1,
        "cache len must stay at 1 on repeated identical (subscripts, shapes)"
    );
}
```

- [ ] **Step 2: Run tests to verify pass**

Run: `cargo test -p tenferro --test nary_einsum_cache --release`
Expected: all five tests pass, including the new `nary_einsum_on_exec_path_hits_cache`.

If the test fails because `einsum_cache_len()` increments to 2 on the second call, the cache-threading chain is broken somewhere between `Engine::eval_exec_ir` and `execute_nary_einsum`. Investigate by tracing: does the second call reach `execute_nary_einsum` with the engine's cache, or an ephemeral one?

- [ ] **Step 3: Commit**

```bash
git add tenferro/tests/nary_einsum_cache.rs
git commit -m "test: verify NaryEinsum cache hit on ExecOp dispatch path (#722)

End-to-end: two Engine::eval_exec_ir calls on a program containing a
NaryEinsum op with identical subscripts+shapes result in
einsum_cache_len() == 1, confirming cache-hit reuse."
```

---

### Task 9: Full workspace verification and doc build

**Files:** none (validation only)

- [ ] **Step 1: fmt / tests / doc**

Run these in sequence:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo doc --workspace --no-deps
```

Expected:
- `cargo fmt --all --check` — clean
- `cargo test --workspace --release` — all pass
- `cargo doc --workspace --no-deps` — no warnings

If `cargo fmt` fails: run `cargo fmt --all` and commit.

- [ ] **Step 2: Coverage check (new code)**

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: passes. If the new files (`engine.rs` changes, the test file itself) drop below 90%, add a targeted test covering the untested branch.

- [ ] **Step 3: Docs site build check**

```bash
python3 scripts/check-docs-site.py
```

Expected: passes.

- [ ] **Step 4: If all green, amend-or-commit and push**

Only create one final "chore" commit if something like formatting changes slipped in. Otherwise nothing to do here.

```bash
git status
# If there are untracked/uncommitted changes beyond expected, investigate.
```

---

### Task 10: Open the PR

**Files:** none (GitHub)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/722-nary-einsum-cache
```

- [ ] **Step 2: Create the PR**

```bash
gh pr create --title "perf: bounded LRU cache for NaryEinsum contraction order (#722)" --body "$(cat <<'EOF'
## Summary

- Replaces `Engine::einsum_cache` (previously unbounded `HashMap`) with a bounded `lru::LruCache` (default capacity 256).
- Threads the cache through the ExecOp dispatch path so repeated `(subscripts, shapes)` skip `ContractionTree::optimize_with_options` (O(n!)).
- Adds `Engine::eval_exec_ir`, `with_einsum_cache_capacity`, `einsum_cache_capacity`, `set_einsum_cache_capacity`, `einsum_cache_contains`.
- Public free functions `eval_exec_ir` / `eval_exec_ir_unsegmented` / `eval_exec_segmented` keep their signatures and internally construct an ephemeral LruCache (default capacity) — API compatible.

Closes #722.

## Test plan
- [x] `cargo fmt --all --check`
- [x] `cargo test --workspace --release`
- [x] `cargo doc --workspace --no-deps`
- [x] New test file `tenferro/tests/nary_einsum_cache.rs` covering: default capacity, `with_einsum_cache_capacity`, `set_einsum_cache_capacity` shrink, LRU eviction order (A,B,A,C with cap=2 → B evicted), end-to-end cache-hit on ExecOp dispatch path via `Engine::eval_exec_ir`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Enable auto-merge**

```bash
gh pr merge --auto --squash --delete-branch
```

Expected: PR opens; auto-merge queued; merges after required checks pass.

---

## Plan Self-Review

**Spec coverage:**

| Spec item | Covered by |
|---|---|
| Replace HashMap with LruCache | Task 2 |
| Default capacity 256 + `DEFAULT_EINSUM_CACHE_CAPACITY` | Task 2 |
| `with_einsum_cache_capacity`, `set_einsum_cache_capacity`, `einsum_cache_capacity` | Task 3 |
| `einsum_cache_contains` (testability) | Task 4 |
| Cache param on `execute_nary_einsum` | Task 5 |
| `eval_exec_ir_unsegmented_with_cache` internal + public wrapper | Task 5 |
| `eval_exec_segmented_with_cache` internal + public wrapper | Task 6 |
| `Engine::eval_exec_ir` method | Task 7 |
| TracedTensor routes through Engine method | Task 7 |
| LRU eviction test | Task 4 |
| Shrink test | Task 3 |
| Cache-hit test on ExecOp path | Task 8 |
| `lru` workspace + crate dep | Task 1 |
| NonZeroUsize type safety | Task 3 |

No spec items uncovered.

**Placeholder check:** No TBD / TODO / "handle edge cases" / vague phrasing in the task steps. Code blocks present for every code change.

**Type consistency:**
- `NaryEinsumCache` type alias used consistently in `exec.rs`, `segment.rs`, `engine.rs`.
- `einsum_cache_contains` takes `&(String, Vec<Vec<usize>>)` in both definition and test call.
- `NonZeroUsize` used in `with_einsum_cache_capacity`, `set_einsum_cache_capacity`, `einsum_cache_capacity` return value.
- `DEFAULT_EINSUM_CACHE_CAPACITY` defined as `usize`, converted via `NonZeroUsize::new(...).expect(...)` consistently at all three call sites (engine.rs, exec.rs wrappers, segment.rs wrapper).
