# Eager semantic journal (P1 of #2417)

## Decisions

- `EagerTensor` no longer embeds a `TracedTensor`. It carries an immutable
  reference into a private op-application record
  (`crates/tenferro-ad/src/eager/semantic.rs`): `Leaf` (key, dtype, rank, bound
  data), `Op` (owned `StdTensorOp`, input values, output count) and `Program`
  (frozen derivative program plus its recorded inputs). A value is
  `(node, output_index)`, so a multi-output op keeps distinguishable values and
  residual identities.
- The record never infers metadata, decides identity, merges or lowers. One
  replay materializes the closure of the requested values into a
  `SemanticProgramBuilder` when a backward (or forward-mode) pass needs the
  canonical program; metadata, shape guards and semantic identity stay owned by
  the builder, and the derivative program is recorded as a `Program` node so a
  second-order pass inlines it instead of rebuilding a graph.
- Per-tensor `TracedTensor` state is not needed for this: `inputs_map` and
  `leaf_metas` are replaced by the leaf node's own binding and symbolic spec, and
  the deferred analysis pass disappears because the builder infers on append.
- Rejected: a frozen `FrozenProgram` per eager tensor. `finish` consumes the
  builder and `import` copies it, so per-op freezing is O(history) per op.
- `SemanticProgramBuilder::bind_input_retained` becomes a doc-hidden public hook
  so the record can bind its retained leaf values without copying them into an
  owned tensor (owner-scoped hook, design section 3.6/3.9 of
  `docs/design/traced-over-trace-context.md`).
- The now-unused runtime helpers `append_raw_eager_outputs` and
  `analyze_deferred_semantic_trace`, and the metadata-scope merge they needed,
  are deleted.

## Verification conclusions and constraints

- The existing eager AD suite (378 integration tests, 129 unit tests, doctests)
  passes with call sites migrated only: gradients, residual reuse, higher-order
  derivatives, checkpoint/truncate, shape churn, context isolation.
- New guards: `Send + Sync` on the record value, one node per applied op, one
  replay per backward pass, multi-output identity, rejection of a malformed
  recorded output count, and a replay/fingerprint equality check against the
  traced path (symbolic `InputDim` leaf extents, not concrete ones).
- Paired benchmark evidence (Apple M5 Max, `CpuEngine::serial()`, macOS has no
  `taskset`; host load 3.8-10 during the run): `eager_ad_transform_cache`
  78.5 µs cold and 33.3 µs warm and `eager_backward_shape_churn/16` ~0.82 ms are
  unchanged within the measured A/A drift; every `eager_dispatch_baseline` cell
  (0.55-1.29 µs) moves within +-3%, which is inside that drift, so no per-op
  speedup is claimed here. A quiet-host paired run is the follow-up if
  microsecond-level attribution is needed.
- Pre-existing, unrelated: `tenferro-runtime --test binary_allocations
  canonical_fallback_contraction_does_not_copy_operands` fails identically at
  `096c0ed7a` on this host.
