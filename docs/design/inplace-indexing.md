# Partial In-place Update Design

## Goal

Add safe and backend-portable support for partial in-place tensor updates
(`x[idx] = value`) in tenferro-rs, aligned with the existing layered design:

- `tenferro-tensor`: views/ownership metadata
- `tenferro-tensor::TensorBackend`: backend execution protocol
- graph-level AD integration: mutation and aliasing safety rules

## Reference Behavior (PyTorch C++)

PyTorch uses two execution paths:

1. Basic indexing (`int`, `slice`, `ellipsis`, `none`, simple bool)
   - Build a view (`select`/`slice`) and write via `copy_`/`fill_`.
2. Advanced indexing (tensor indices / masks)
   - Dispatch to `index_put_` kernels (CPU/CUDA).
   - For CUDA `accumulate=true`, use sort-based path for correctness.

Autograd safety is enforced independently by view+in-place checks and
version counters.

## Fit with Current tenferro Design

### Already available

- Zero-copy view ops (`select`, `narrow`, `permute`, `diagonal`, `broadcast`).
- Shared-buffer ownership model (`Arc<BufferInner<T>>`).
- Backend protocol surface (`TensorBackend` / `BackendSession`).
- Existing out-parameter and buffer-reuse patterns in backend execution.

### Missing pieces

- No dedicated indexing primitive family yet.
- No unified version-counter based mutation tracking.
- No AD mutation guard equivalent to PyTorch `check_inplace`.

Conclusion: integration is feasible, but should be staged.

## Proposed API Surface

### Tensor-level API

```rust
pub enum TensorIndex {
    Integer(isize),
    Slice { start: Option<isize>, stop: Option<isize>, step: isize },
    Ellipsis,
    NoneAxis,
    Bool(bool),
    // Phase 2+
    IndexTensor(Tensor<i64>),
    BoolMask(Tensor<bool>),
}

impl<T: Scalar> Tensor<T> {
    pub fn set_item_(&mut self, indices: &[TensorIndex], value: &Tensor<T>) -> Result<()>;
}
```

### Backend extension (Phase 2+)

```rust
pub enum IndexingDescriptor {
    IndexPut {
        accumulate: bool,
    },
}

// Future shape: add an index_put-style method to TensorBackend/BackendSession
// once mutation/version-counter semantics are specified.
```

## Execution Model

### Path A: basic indexing (Phase 1)

- Parse indices into a strided view of destination.
- Use copy semantics compatible with current `alpha=1, beta=0` style behavior.
- CPU first; no advanced index tensor support required.

This path can be implemented without adding a new backend kernel by composing
existing view + contiguous/copy logic.

### Path B: advanced indexing (Phase 2+)

- Normalize tensor/mask indices.
- Dispatch to a backend `TensorBackend`/`BackendSession` indexing method if
  available.
- Fallback behavior:
  - CPU backend: reference implementation.
  - Backends without extension: return explicit `Error::DeviceError` or use
    documented slow fallback (if implemented).

## Aliasing Semantics and COW

### Alias-preserving semantics (adopted)

`set_item_` should preserve alias behavior of zero-copy views:

- If `v` is a view of `a` (for example via `narrow`/`select`), then writing
  through `v` must update the shared underlying storage.
- Equivalent aliases must observe the same updated values.

Example expectation:

```rust
let mut a = ...;
let mut v = a.narrow(1, 2, 3)?;
v.set_item_(..., &value)?;
// update is visible through `a` and other aliases of the same storage
```

### Why not full COW as the default

A write-time implicit clone ("copy on write") makes mutation behavior less
predictable for view-heavy tensor code:

- An in-place call may unexpectedly stop affecting aliases.
- The meaning of "in-place" diverges from PyTorch-style storage semantics.
- AD safety still requires mutation tracking and checks; COW does not replace
  version counter / view+in-place validation.

### Practical policy

- Do not perform implicit deep-copy split on `set_item_` for normal execution.
- Keep mutation semantics storage-based (alias-preserving).
- Handle AD safety explicitly via guards and version tracking (see below).

## AD Safety Policy

### Phase 1 policy (recommended)

- Disallow `set_item_` when mutation would interact with AD-tracked values.
- Return a clear error instead of allowing silent gradient corruption.

### Phase 2 policy

Introduce explicit mutation tracking:

- Shared version counter across aliased views.
- Bump on any in-place write.
- Validate saved values during backward/HVP and error on stale versions.
- Add view+in-place conflict checks for tracked tensors.

This mirrors the PyTorch safety model conceptually, while keeping tensor
storage independent from the graph-level AD rules owned by
`tenferro-internal-ops` and the semantic AD surfaces in `tenferro-ad`.

## GPU Notes

- API should be backend-neutral from day one.
- CubeCL/CUDA kernels for advanced indexing can be added incrementally behind
  the existing backend traits. ROCm remains a stub until explicitly implemented.
- Deterministic behavior with duplicate indices should be explicitly specified
  (especially for `accumulate=true`).

## Rollout Plan

1. Phase 1: `set_item_` basic indexing only (CPU), AD-unsafe cases rejected.
2. Phase 2: add a backend indexing method and `IndexingDescriptor::IndexPut`.
3. Phase 3: version counter + view/in-place safety checks for AD.
4. Phase 4: GPU advanced indexing kernels and deterministic policy completion.

## Non-goals (initial phases)

- Full NumPy/PyTorch advanced indexing parity in Phase 1.
- Implicit cross-device data movement during assignment.
- Enabling in-place mutation inside AD paths before safety tracking lands.
