# TracedTensor over TraceContext (P2 of #2417)

## Decisions

- `TraceContext` is a cloneable `Rc<RefCell<TraceInner>>` handle; ops take
  `&self`, a re-entrant borrow is a typed error, and `finish(&self, outputs)`
  freezes a copy through `SemanticProgramBuilder::snapshot`. The extension-cache
  store is reached through `with_extension_caches(|caches| ..)` because a `&mut`
  borrow cannot escape the cell.
- `TracedTensor` is `(ctx, value)` plus a private, immutable cache of the value's
  `rank`, `dtype` and `shape_hint`. The cache is derived from `value_metadata`
  when the value is appended (`trace_ops::finish_value`), so it can never
  disagree with the program; `id`, `graph` and `val` are gone.
- Standard operations and constructors are written once in `trace_ops.rs`
  against a `(TraceContext, TraceValue)` pair. `TracedTensor` methods keep their
  names and bodies and delegate through the appenders.
- Symbolic extents map `DimExpr::InputDim { input_idx, axis }` to
  `SymDim::tensor_axis(input_idx as u64, axis)`, so `sym_size`, `reshape_sym` and
  `broadcast_in_dim_sym` round-trip through `SymDim::to_dim_expr` with the
  identity tensor map.
- Checkpoints are context state: `TraceContext::record_checkpoint_alias` appends
  a bound input and records `(input, target, bound_shape)`;
  `snapshot_substituting_aliases` replaces uses of the input with a prefix
  `Slice` of the target, which reproduces the old graph-level refinement.
- The graph merge path, the per-tensor scope chains and the graph-level extension
  helpers (`append_raw_op`, `analyze_extension_graph`, `apply_expanded_graph*`,
  `attach_expanded_shape_contract`) are deleted: `GraphCompiler` now compiles
  `TraceContext::snapshot(..)` output and keeps exactly one lowering path.

## Verification conclusions and constraints

- `crates/tenferro-runtime` compiles with the new representation; the remaining
  work is the `tenferro-ad` AD rewrite (snapshot + transform + `import` with an
  input remap, public `grad_many`/`value_and_grad`), the call-site migration
  (~190 files), then the workspace test run, coverage and the local A100 gate.
- Unused-import and dead-code warnings are expected until the removed modules
  (`metadata` scope chain, `checkpoint` graph nodes, `shape_constraint`) are
  deleted in the same series.
- No test or benchmark was run yet at this commit: the cut-over is not complete.

## Progress checkpoint (runtime and AD migrated)

- `TenferroTrace`/`TracedTensor`: `TraceContext` is the handle, `TracedTensor` is
  `(ctx, value)` with a derived cache; all runtime, linalg, fft, einsum, sparse and
  tropical libraries build on it.
- AD runs on snapshot → program transform → `import` with an input remap; the
  public multi-`wrt` API (`grad_many`, `value_and_grad`, `jvp_many`, `vjp_many`)
  is in place, and checkpoints are a context alias table with lazy prefix `Slice`
  substitution.
- Dead extension outputs are pruned at snapshot time, and the pruning renumbers
  kept inputs (input declarations are rewritten) so guards keep referring to the
  right program input.
- AD rules that need extents (slice, pad, reductions) required the semantic
  transform to import the source bindings; design D3's "structure only" import
  broke them at the time. Import now keeps the source's symbolic extents and only
  re-attaches the bindings, and the four rules that must emit literal lengths
  read them from the bindings (`SemanticProgramBuilder::resolved_extents`). The
  earlier note that the final remap alone kept concrete extents out of the traced
  program was wrong: the payloads copied from concretised metadata survived the
  remap, and a derivative of a bound symbolic input was specialised to that
  binding (review of #2432, finding 6). Rules whose payload is a literal
  (`Slice`, `Pad`, `DynamicSlice`, concatenate transpose) remain binding-specialised
  by construction; see the design note in `traced-over-trace-context.md`.

### Open items at this checkpoint

- `tenferro-ad` integration: 373/377 pass. Open: two contract-change tests
  (`binding_validation::compile_with_input_specs_validates_specs_without_tensor_values`
  now sees placeholders as program inputs instead of `UnboundPlaceholder`;
  `convenience_api::...cover_public_surface` now gets a symbolic `sym_shape`),
  `dynamic_truncate::dynamic_truncate_rejects_backend_size_binding_on_cpu`
  (input-index expectation), and `hvp::hvp_for_quadratic_form`, which still hits
  a shape guard naming program input 2 in a one-input program — a stale guard
  index in the second-order path that needs the same renumbering treatment as
  input declarations.
- Not started: the `tenferro-linalg`/`fft`/`einsum`/ext test suites, `docs/`
  (102 Markdown pages and `docs/tutorial-code`), `samples/`, then the workspace
  test run, coverage, the local A100 gate, the 5-commit split and the PR.

### Checkpoint 2 (AD green, remaining test migration)

- `tenferro-runtime` unit and integration suites pass (396 unit tests; the only
  runtime failure is the host-specific `binary_allocations` test tracked in #2422).
- `tenferro-ad` passes: 129 unit tests and 377 integration tests. The last
  failure was second-order AD: AD seeds created as symbolic leaves left
  `Reshape`/`BroadcastInDim` expressions unresolved at execution staging, so
  seeds are now bound with concrete extents (`TraceContext::bound_input_concrete`).
- Snapshot pruning keeps declared input order (a caller's `compile_with_input_specs`
  order is the program's input order), renumbers kept inputs, substitutes mapped
  inputs in guards, and discharges statically satisfied guards.
- Remaining: `tenferro-linalg`, `tenferro-fft`, `tenferro-einsum`, `ext/sparse`,
  `ext/tropical` test suites (mechanical constructor/preamble migration in
  progress; a few files were left structurally broken by a bulk rewrite and were
  reverted to be redone by hand: `fft_ndim.rs`, `tropical_ad.rs`, `sparse_ad.rs`,
  the einsum MPS benches), then `docs/tutorial-code` (6 binaries, reverted to be
  migrated by hand because the bulk rewrite changed `fn main()`), the 102
  Markdown pages, `samples/`, the workspace run, coverage, the local A100 gate,
  the five-commit split and the PR.
