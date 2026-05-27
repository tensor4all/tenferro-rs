# NaryEinsum Ergonomic API Design

- **Issue:** tensor4all/tenferro-rs#658
- **Date:** 2026-04-18
- **Status:** Approved (ready for implementation plan)

## Background

Issue #658 proposed two things: (1) a `NaryEinsum` op variant that preserves N-ary
einsum as a single graph node until execution time, and (2) ergonomic constructors
for `TracedTensor` / `Tensor` that make the "static vs symbolic shape" choice
explicit. Since the issue was filed, (1) has been implemented as part of the
1-layer IR refactor (#732): `StdTensorOp::NaryEinsum` exists
(`tenferro-ops/src/std_tensor_op.rs:94-99`), it lowers directly to `ExecOp::NaryEinsum`
via `compile_std_to_exec` (`tenferro/src/compiler.rs:13`), and dispatch between
binary decomposition (all shapes concrete) vs. single `NaryEinsum` node (any
symbolic input) runs on `try_concrete_shape` at `tenferro/src/einsum.rs:263-267`.

This design covers the remaining work: (2) the ergonomic constructor surface and the
execution API required to drive graphs that contain shape-symbolic placeholders.

## Goals

- Add public API to construct `TracedTensor` with either a concrete shape or a
  symbolic (rank-only) shape, with or without attached data.
- Add `Engine::eval_with_inputs` so graphs containing data-less placeholders can
  be evaluated by binding concrete tensors at call time.
- Preserve the existing AD construction path — `grad` / `vjp` / `jvp` are unchanged.
  Graphs produced by AD on symbolic inputs are evaluated via `eval_with_inputs`.
- Add `From<TypedTensor<T>> for Tensor` to eliminate repeated `Tensor::F64(...)`
  boilerplate in tests.
- Rename `TracedTensor::new` / `Tensor::new` / `TracedTensor::from_tensor` to
  more descriptive names. Remove the old names in the same PR.

## Non-goals

- Defining new AD semantics for N-ary einsum (already handled by decomposition
  through existing binary AD rules).
- Execution-time contraction-path caching (tracked separately at #722).
- `dot_decomposer` completion (tracked separately at #729).
- `EagerTensor` changes — this design only touches the traced surface.
- Any change to `StdTensorOp::NaryEinsum` or `ExecOp` — the op vocabulary is
  already complete (as of #732).

## Terminology

The existing codebase uses "concrete" to mean "every dim is `SymDim::Const`"
(`try_concrete_shape`, `concrete_shape` helpers at `tenferro/src/traced.rs:90-111`).
"Symbolic" means the shape contains at least one non-constant `SymDim`. This
design adopts the same vocabulary in the new public API for consistency.

## § 1 — Constructor API

### `TracedTensor` constructors

```rust
impl TracedTensor {
    /// Build from a concrete `Tensor`, keeping its shape as a concrete
    /// `shape_hint` (current behavior of `from_tensor`).
    pub fn from_tensor_concrete_shape(tensor: Tensor) -> Self;

    /// Build from a concrete `Tensor` but treat its shape as symbolic during
    /// graph construction (the tensor data is still attached for eval).
    pub fn from_tensor_symbolic_shape(tensor: Tensor) -> Self;

    /// Build a placeholder (no data) with a fixed concrete shape.
    /// Must be bound via `eval_with_inputs` before evaluation.
    pub fn input_concrete_shape(dtype: DType, shape: &[usize]) -> Self;

    /// Build a placeholder (no data) with rank only (all dims symbolic).
    /// Must be bound via `eval_with_inputs` before evaluation.
    pub fn input_symbolic_shape(dtype: DType, rank: usize) -> Self;

    /// Build from typed `Vec<T>` + shape (renamed from `new`).
    pub fn from_vec<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self;

    /// Returns `true` iff every dim in `shape_hint` is a constant `SymDim`.
    pub fn is_concrete_shape(&self) -> bool;
}
```

Classification:

| Constructor | Data | `shape_hint` | Treated as symbolic by `try_concrete_shape` | Needs `eval_with_inputs` |
|---|---|---|---|---|
| `from_vec(shape, data)` | Yes | `Some([Const(d0), Const(d1), …])` | No | No |
| `from_tensor_concrete_shape(tensor)` | Yes | `Some([Const(d0), …])` | No | No |
| `from_tensor_symbolic_shape(tensor)` | Yes | `None` | Yes | No (data used at eval) |
| `input_concrete_shape(dtype, shape)` | No | `Some([Const(d0), …])` | No | **Yes** |
| `input_symbolic_shape(dtype, rank)` | No | `None` | Yes | **Yes** |

Note: `from_tensor_symbolic_shape` keeps the underlying tensor's data (so
`eval()` still works by reading the data's shape at runtime), but advertises
a symbolic shape to graph passes — this lets callers build a single graph
against one concrete tensor while disabling shape-specific build-time
optimizations.

### `Tensor` constructor

```rust
impl Tensor {
    /// Dispatch on `T` via `TensorScalar::into_tensor`. Renamed from `new`.
    pub fn from_vec<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self;
}
```

### `From<TypedTensor<T>>` impls

```rust
impl From<TypedTensor<f64>>     for Tensor { /* -> Tensor::F64 */ }
impl From<TypedTensor<f32>>     for Tensor { /* -> Tensor::F32 */ }
impl From<TypedTensor<Complex64>> for Tensor { /* -> Tensor::C64 */ }
impl From<TypedTensor<Complex32>> for Tensor { /* -> Tensor::C32 */ }
```

### Removed (no deprecated alias)

- `TracedTensor::new`
- `TracedTensor::from_tensor`
- `Tensor::new`

All call sites are migrated in the same PR. `TypedTensor::from_vec` remains.

## § 2 — Eval API

```rust
impl TracedTensor {
    pub fn eval_with_inputs<B: TensorBackend>(
        &mut self,
        engine: &mut Engine<B>,
        bindings: &[(&TracedTensor, &Tensor)],
    ) -> Result<&Tensor>;
}
```

### Binding

Each `(&TracedTensor, &Tensor)` pair says "this placeholder gets this data".

A placeholder's identity is its `TensorInputKey`, derivable from the leaf
TracedTensor via `self.fragment.vals()[self.val].key.clone()`. An accessor
`TracedTensor::input_key(&self) -> Option<TensorInputKey>` returning `Some`
iff `self` is a leaf (single-node fragment wrapping an input) is added for
this purpose.

Resolution walks `graph.inputs` (the compiled graph's ordered input keys):

1. If the key is present in `inputs_map` (data-carrying leaf: `from_vec` /
   `from_tensor_*`) → use that stored `Arc<Tensor>`.
2. Otherwise → look up the key in a binding map built from `bindings` (keyed
   on placeholder input keys) and use the bound `&Tensor`. Missing keys
   surface as `UnboundPlaceholder`.

### Validation (eager, before execution)

| Check | Error variant |
|---|---|
| Left side of a binding is not a placeholder (has data) | `UnexpectedBinding` |
| A placeholder in the graph has no binding | `UnboundPlaceholder` |
| Same placeholder bound twice | `DuplicateBinding` |
| Bound tensor's dtype ≠ placeholder dtype | `PlaceholderDtypeMismatch` |
| Placeholder from `input_concrete_shape` and bound tensor shape differs | `PlaceholderShapeMismatch` |
| Placeholder from `input_symbolic_shape` and bound tensor rank differs | `PlaceholderRankMismatch` |

All new variants live in `tenferro::error::Error`.

### Caching

`ExecProgram` shape fields are `DimExpr`, resolved at execution time from input
shapes (`DimExpr::eval_all`, `tenferro/src/exec.rs:224,242`). One compiled
`ExecProgram` therefore handles all shape instantiations of the same graph.
`Engine::compile_cache` key stays unchanged — no shape signature in the key.

### Relationship to existing `eval()`

- `eval()` remains for graphs where every input has attached data. Behavior
  unchanged.
- `eval_with_inputs(bindings)` handles the superset: accepts data-carrying and
  placeholder leaves. If the graph has no placeholders and `bindings` is empty,
  it degenerates to `eval()`. Mismatched binding count → error.

## § 3 — AD behavior

Graph construction (`vjp`, `jvp`, `grad`, `grad_optional`) is unchanged. Placeholders
are valid `wrt` targets because they are registered as normal input leaves via
`next_input_key()` (same mechanism as `from_tensor_*`) — the AD engine sees
them as standard leaves regardless of whether they carry data.

The only runtime consequence is that gradient graphs built over symbolic inputs
must also be evaluated via `eval_with_inputs`:

```rust
let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
let y = f(&x);
let mut dy_dx = y.grad(&x)?;
let out = dy_dx.eval_with_inputs(&mut engine, &[(&x, &concrete_x)])?;
```

HVP, iterative_ad, and oracle_replay use cases require no code changes — they
continue to call `eval()` when inputs are concrete.

## § 4 — Test plan

New integration tests under `tenferro/tests/`:

| Test file | Coverage |
|---|---|
| `symbolic_input.rs` | E2E: construct via `input_concrete_shape` / `input_symbolic_shape`, evaluate via `eval_with_inputs`, verify output |
| `nary_einsum_symbolic.rs` | Verify that mixing at least one `input_symbolic_shape` into `einsum` causes the compiled IR to contain `ExecOp::NaryEinsum` (not binary decomposition) |
| `symbolic_grad.rs` | Build a symbolic graph, call `grad`, evaluate the gradient graph via `eval_with_inputs` with the same bindings |
| `binding_validation.rs` | Each new `Error` variant is triggered by the expected mismatch |

Existing tests are updated mechanically for the rename (see §5). No existing
test logic changes beyond the rename.

Coverage target: maintain 90%+ per file (repository policy). All four new
modules must hit that threshold.

## § 5 — Implementation ordering

One PR, split into groups for subagent parallelization via
`superpowers:dispatching-parallel-agents`:

| Group | Agent count | Files | Work |
|---|---|---|---|
| **Core** | 1 (sequential) | `tenferro-tensor/src/types.rs`, `tenferro/src/traced.rs`, `tenferro/src/engine.rs`, `tenferro/src/error.rs` | Add new constructors, `From` impls, `eval_with_inputs`, new `Error` variants. Remove old constructors. |
| **A** | 1 agent | `tenferro/tests/**.rs` (~25 files) | Mechanical rename: `TracedTensor::new` → `from_vec`; `TracedTensor::from_tensor` → `from_tensor_concrete_shape`; `Tensor::new` → `from_vec`. Drop local `f64_tensor` helpers made redundant by `Tensor::from_vec` + `From` impls. |
| **B** | 1 agent | `tenferro/src/**.rs` excluding Core | Same mechanical rename. |
| **C** | 1 agent | `tenferro-tensor/src/**` (non-Core), `tenferro-einsum/src/**` | Same. |
| **D** | 1 agent | `docs/**.md`, `README.md`, `tenferro/README.md` | Update code samples to new names. |
| **E** | 1 (sequential, last) | `tenferro/tests/{symbolic_input,nary_einsum_symbolic,symbolic_grad,binding_validation}.rs` | Write the four new test files. |

Execution order:

1. **Core** alone (must land in tree before any call-site update compiles).
2. **A, B, C, D** in parallel (mutually independent).
3. **E** last (depends on all new API being callable and all call-site updates compiling).

Each group's subagent must end with its own validation green:

| Group | Validation command |
|---|---|
| Core | `cargo build -p tenferro-tensor -p tenferro` |
| A (`tenferro/tests`) | `cargo build --tests -p tenferro` |
| B (`tenferro/src`) | `cargo build -p tenferro` |
| C (`tenferro-tensor`, `tenferro-einsum`) | `cargo build -p tenferro-tensor -p tenferro-einsum` |
| D (docs) | `python3 scripts/check-docs-site.py` + `cargo doc --workspace --no-deps` |
| E (new tests) | `cargo test --workspace --release` of the four new test files |

The final commit runs the full pre-PR checklist per `AGENTS.md`:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

## Open questions

None at design time. Implementation may surface minor naming or signature
questions; these are to be resolved in the implementation plan.

## References

- Issue: tensor4all/tenferro-rs#658
- Prior spec (NaryEinsum op, shape-agnostic graph): `docs/superpowers/specs/2026-04-07-shape-agnostic-graph-design.md`
- Existing `NaryEinsum` op wiring: `tenferro-ops/src/std_tensor_op.rs:94-99`, `tenferro/src/einsum.rs:263-267`, `tenferro/src/exec.rs:576-667`
- Symbolic dim infrastructure: `tenferro/src/sym_dim.rs`, `tenferro/src/traced.rs:54,90-111`
