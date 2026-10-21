# Concrete tensor read surface (#2214)

## Decisions
- One operand descriptor (`Operand { data, shape, strides, offset }`, `tenferro-cpu`) feeds each
  CPU indexing kernel, so the owned and the `*_read` entries share one body and a view is read in
  place. Tri/diagonal masks use strided-source entries added upstream (strided-rs#325) instead of
  a tenferro-local traversal.
- Read defaults pass an owned tensor through and stage only a view. Staging an owned tensor was a
  copy for nothing, and the host default refuses device-placed tensors.
- The session composite (`take_along_axis`, the softmax family) calls the read hooks rather than
  materializing its inputs, which is what lets the typed `take_along_axis` borrow its tensor.
- `dynamic_update_slice` (issue item 9) is added to both sugar surfaces at once, with its own
  read hook, rather than typed-only: the parity test enforces one spelling across both.
- Published crates in the CPU stack depend on each other with caret requirements (workspace
  `CODING_RULES.md`). Exact pins forced a release of the whole chain for every strided patch.

## Verification conclusions and constraints
- `read_entries_consume_views_without_staging` (tenferro-cpu) pins zero staging and values for
  every CPU read entry on a transposed view; `materializing_read_defaults_stage_views_only`
  (tenferro-tensor) pins the default contract.
- Only the CPU engine was changed. CUDA overrides only `gather_read`, and WebGPU none, so other
  entries there still stage views. GPU read kernels are tracked in #2205, #1930 and #2082.
