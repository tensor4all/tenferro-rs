# Thread-Safe Chainrules and DynAdTensor Design

This redesign makes thread-safe reverse-mode the canonical foundation in `chainrules` and removes `DynTape` from the public `tenferro-dyadtensor` surface.

## Goals

- Replace the split reverse-mode implementations (`Tape/TrackedValue` and `Variable/AutogradContext`) with one thread-safe engine.
- Make `DynAdTensor` and its reverse graph handles naturally `Send + Sync` by construction.
- Remove `DynTape` from the public `dyadtensor` API and make `DynAdTensor` the only user-facing AD tensor object.
- Keep `Diag` support in `DynTensor`, but treat linalg as dense-only and return runtime errors for structured non-dense inputs.
- Leave PyTorch-like dtype conversion/promotion to issue `#500`; do not expand that scope here.

## Current Problems

- `chainrules::Tape<V>` uses `Rc<RefCell<_>>`, so `TrackedValue<V>` and any wrapper over it are `!Send + !Sync`.
- `chainrules::Variable<V>` already uses `Arc<Mutex<AutogradContext<V>>>`, so the crate has two reverse engines with duplicated logic.
- `tenferro-einsum` reverse rules capture `Rc<RefCell<BackendContext<...>>>`, so even a thread-safe tape would not be enough by itself.
- `tenferro-dyadtensor` still exposes `DynTape`, leaking graph mechanics into the public frontend.

## Recommended Architecture

### 1. One reverse engine in `chainrules`

Introduce an internal shared graph core named `AutogradGraph<V>`.

- `Tape<V>` becomes a thin graph-first wrapper over `Arc<Mutex<AutogradGraph<V>>>`.
- `Variable<V>` becomes a thin value-first wrapper over the same `Arc<Mutex<AutogradGraph<V>>>`.
- `TrackedValue<V>` remains as the graph-first value handle, but now points at the same graph type as `Variable<V>`.
- `AutogradContext<V>` is removed or renamed into `AutogradGraph<V>` to avoid two names for the same engine.

The graph core owns:
- node storage
- reverse rules
- leaf tangent seeds
- cotangent accumulation helpers
- HVP helpers
- graph-liveness state

### 2. Stronger thread-safe rule boundary

`chainrules-core` should require thread-safe reverse/forward rules.

- `ReverseRule<V>: Send + Sync`
- `ForwardRule<V>: Send + Sync`

This forces downstream captured state to become thread-safe too.

### 3. Shared graph implementation, not shared naming only

The duplicated reverse traversal logic in:
- `extern/chainrules/src/engine/tape.rs`
- `extern/chainrules/src/engine/context.rs`
- `extern/chainrules/src/engine/variable.rs`

must be collapsed into one implementation inside `AutogradGraph<V>`.

`Tape::pullback`, `Tape::hvp`, `Variable::backward`, `Variable::backward_hvp`, and related helpers should delegate to the same graph-core functions.

### 4. Lock strategy

Canonical foundation is thread-safe, but the first step does not need a sophisticated lock-free traversal.

Phase 1 lock policy:
- `AutogradGraph<V>` is stored under `Arc<Mutex<_>>`.
- reverse traversal may execute under the graph lock initially if needed for simplicity.
- correctness and thread-safety take priority over minimizing contention.

A later optimization can snapshot rules/nodes and release the graph lock before executing pullbacks, but that is not required for the first clean redesign.

### 5. `tenferro-einsum` captured state

`tracked_einsum` and its reverse rule must stop using `Rc<RefCell<BackendContext<...>>>`.

Use `Arc<Mutex<BackendContext<...>>>` consistently for reverse-mode tracked paths.
The rule objects stored in `AutogradGraph<V>` must satisfy the new `Send + Sync` bounds.

### 6. `tenferro-dyadtensor` public surface

`DynAdTensor` becomes the only public AD tensor object.

- Remove public `DynTape`.
- Remove public typed `AdTensor<T>` from the main dyadtensor entry surface.
- `DynAdTensor` internally wraps one of:
  - primal `DynTensor`
  - forward `DualValue<DynTensor>`
  - reverse `Variable<DynTensor>`

Users create reverse graphs through `DynAdTensor` methods such as:
- `requires_grad_(true)`
- explicit constructors that do not expose the tape type
- `backward(...)`, `grad(...)`, `pullback_wrt(...)`

### 7. `DynTensor` remains internal payload

`DynTensor` remains the internal dynamic primal payload type.
It still supports dense and `Diag` structured values.

`DynAdTensor` is the public wrapper.
`DynTensor` does not need to remain a prominent public type.

### 8. Structured linalg boundary

Keep the earlier policy:
- structured-safe: einsum, reduction, layout-preserving linear ops
- dense-only: linalg ops (`svd`, `qr`, `lu`, `eig`, `solve`, etc.)

If `axis_classes` contains equivalence classes (non-dense structured layout), linalg should return a runtime error from the dyadtensor frontend.

## Naming

Because `chainrules` is a generic engine, names should reflect generic graph semantics rather than PyTorch-specific per-tensor metadata.

Recommended names:
- public wrapper: `Tape<V>`
- public value handle: `TrackedValue<V>`
- public value-first handle: `Variable<V>`
- internal graph core: `AutogradGraph<V>`
- optional private storage type: `AutogradGraphInner<V>` or `AutogradGraphState<V>`

Avoid:
- `AutogradMeta` (PyTorch uses this for per-value metadata, not shared graph storage)
- `Context` as the canonical core name (too ambiguous with backend/runtime contexts)

## Testing Policy

- `chainrules` tests should confirm `Tape<V>` and `Variable<V>` share the same graph engine semantics.
- Add compile-time `Send + Sync` assertions for:
  - `Tape<V>`
  - `TrackedValue<V>`
  - `Variable<V>`
  - `DynAdTensor`
- `tenferro-einsum` reverse AD tests should continue to pass with `Arc<Mutex<BackendContext<...>>>`.
- `tenferro-dyadtensor` integration tests should move to `DynAdTensor`-only reverse-mode examples.
- Add regression tests that `DynTape` is gone from the public surface.
- Keep dense-only structured-linalg rejection tests.

## Documentation Updates

Update:
- `extern/chainrules/src/lib.rs`
- `docs/design/autodiff.md`
- `docs/api_index.md`
- `extension/tenferro-dyadtensor/README.upstream.md`
- `tenferro-dyadtensor` crate docs

The docs should say:
- `chainrules` is thread-safe by default
- `DynAdTensor` is the public dynamic AD tensor object
- scalar semantics are rank-0 tensors
- structured linalg is dense-only

## Non-Goals

- Do not redesign dtype conversion/promotion in this PR. That is tracked separately in issue `#500`.
- Do not introduce a second single-thread fast path.
- Do not broaden structured linalg support beyond dense-only.
