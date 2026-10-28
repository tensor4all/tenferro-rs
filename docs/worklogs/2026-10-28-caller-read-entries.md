# Eager and traced callers hand the read entries their operand (#2223)

## Decisions

- The eager dispatch (`crates/tenferro-ad/src/eager_exec.rs`) now calls the
  `*_read` hook for extract/embed diagonal, tril, triu, slice, pad, reverse,
  concatenate, scatter, dynamic-slice, dynamic-update-slice and the pad/slice
  branches of `PadToMatch` and `DynamicTruncate`. It previously resolved a
  borrowed view through `to_contiguous_read` first, so the native read entries
  of the CUDA and CPU backends never saw an eager view operand.
- Promotion is unchanged: a dtype mismatch still stages and converts the
  operand, because that conversion is the value change itself. Only the
  same-dtype path lost the copy, so `concrete_promoted_read_to_dtype` had no
  caller left and was removed. `Convert` keeps its materializing boundary
  (`cast` takes an owned tensor) and is the staging positive control in the
  dispatch test.
- An operation whose *output* must be a fresh owned tensor still materializes:
  `PadToMatch` without padding and the dynamic-truncate size value are value
  reads, not operand reads.
- Traced dispatch: the backend entries already pass `get_read(slots, ..)`
  (`crates/tenferro-runtime/src/exec/dispatch.rs`), which the
  `reads_input_view` contract test pins. The two host ops did not:
  `DynamicTruncate` staged operand 0 before the owned slice, and `PadToMatch`
  staged both the operand and the shape-only reference before the owned pad.
  They now use `get_read` + `slice_read`/`pad_read`; only the truncate size
  scalar and the unpadded `PadToMatch` output materialize, and
  `reads_input_view` now covers both host ops instead of `ShapeOf` alone.
- The CUDA staging assertion stays where the counter is observable. `staged_reads`
  is crate-private to `tenferro-gpu`; exposing it to a `tenferro-ad` device test
  would need a new public accessor or a test-only feature, so the unit keeps the
  two halves apart instead: the dispatch spy proves the caller hands the view
  over untouched, and `tenferro-gpu`'s own test proves the engine's read entries
  stage nothing.

## Verification conclusions and constraints

- Caller staging: `RecordingEngine` counts `to_contiguous_read` and forwards the
  in-scope read entries to its CPU engine. `eager_read_entries_consume_borrowed_views_without_staging`
  holds the count at zero for every in-scope op with an owned `TensorRead::Tensor`,
  a compact borrowed `TensorRead::from_view(..)`, and an offset/non-compact view,
  and compares all three value sets. `traced_host_ops_do_not_stage_a_borrowed_operand_or_shape_reference`
  does the same for the traced `DynamicTruncate` and `PadToMatch` host paths and
  keeps the unpadded branch's single output materialization as the positive
  control. `eager_promoting_read_entries_convert_only_the_mismatched_operand`
  pins that a mixed-dtype concatenate/scatter/dynamic-update-slice stages one
  conversion and leaves the already-promoted operand borrowed. Reverting one arm
  to `concrete_tensor_read` fails these tests, so the evidence is a negative
  control, not a dead counter.
- Value parity: `crates/tenferro-ad/tests/integration/read_entry_parity.rs` binds
  one logical rank-3 tensor to a traced placeholder as a compact borrowed read
  and as the strided view, and matches both against the eager result on an
  untracked (view-valued) eager input for the structural, indexing, concatenate,
  dynamic-truncate and pad-to-match entries.
- Device: `test_gpu_eager_structural_reads_agree_on_a_strided_device_view` runs
  slice, reverse and concatenate through the eager API on an A100 and matches a
  strided device view against its compact copy. It pins values only; the eager
  path may still take its ownership copy in `duplicate_value_in_session` where a
  new eager value has to own the view's root.
- The claim "the eager CUDA path stages no operand view for these entries" is
  restricted to the entries U3 made native (`slice`, `dynamic_slice`, `pad`,
  `reverse`, `extract_diagonal`, `embed_diagonal`, gather's operand, `scatter`,
  `concatenate`) and is the composition of the caller-side count above and
  `tenferro-gpu`'s `staged_reads` unit test on the same entries; no single
  cross-crate assertion exists, by choice (no new public surface in this phase).
  The remaining staging boundaries are outside that claim: `tril`/`triu` keep the
  compact-only kernel and the read-default copy, index operands (gather/scatter
  indices, dynamic-slice and dynamic-update-slice starts) are still materialized
  for the index kernel, and `Convert`/`cast` takes an owned tensor.
