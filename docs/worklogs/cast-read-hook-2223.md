# `cast_read`: one read hook for the dtype projection (#2223)

## Decisions

- **The read form is additive.** `cast_read` gets a temporary staging default
  (`stage_view` then owned `cast`), and `convert_read` is provided as lattice
  validation followed by `cast_read`. The owned `cast`/`convert` and the default
  go away in the same change that deletes the other owned twins (#2223 PR 3,
  0.12 window), so no backend breaks here and the public `Engine` sub-traits keep
  their shape.
- **One conversion table, keyed on the source dtype.** The CPU cast matrix moved
  from a `(from, to)` match over `&Tensor` into a private `CastFrom` trait
  implemented per source scalar, parameterized by a `StridedView`. The owned path
  reaches it through `host_view`, the view path through `read_host_view`, so a
  second 7×7 matrix never exists. The shared output sets the source placement:
  the same-dtype arm used `duplicate()`, which preserves placement, while the
  cross-dtype pooled arms silently dropped it, and one rule for both is the
  stricter contract.
- **The float/complex→int range check walks logical elements.** It previously
  validated `typed_host_data` of the backing buffer, which for a view is the
  whole allocation — a NaN outside a slice would have refused the cast. The walk
  is an index odometer over the `StridedView`, so it is layout-exact for owned
  and borrowed input alike and allocates nothing.
- **CUDA accepts one staging copy (D6).** The catalogue `Convert` entry takes one
  flat span and no layout, so `cast_read` resolves a borrowed view through
  `device_read` (one copy, counted by `staged_reads`) and runs the existing flat
  kernel plus `ValidateRealCast`. A layout-taking entry is a cross-repo
  `t4a-cuda-kernels` change and is out of scope.
- **The whole `Convert` path hands over a `TensorRead`.** Traced dispatch, the
  eager executor, the eager promotion path and the erased core body
  (`api/body.rs`) all call `cast_read`/`convert_read`. `ensure_backend_inputs`,
  `ensure_owned` and the owned `get` accessor had no other caller and are
  deleted, which is why the fold predicate no longer excuses `Convert` and a
  metadata view now folds into it.

## Review record

- Plan review: `ollama-cloud/glm-5.3-flash`, three rounds. Round 1 `CHANGES`
  (placement dropped on the same-dtype pooled path; the staging count is
  crate-private and cannot be asserted from `tenferro-runtime`; four existing
  tests invalidated; the erased core body omitted), round 2 `CHANGES` (the fold
  assertion is `(0, Transpose)`, and the counting engine needs a second
  constructor), round 3 `APPROVE`.
- Post-implementation review of the finished diff: same model, `CHANGES`. Its one
  IMPORTANT finding was that nothing failed if the executor re-staged the view
  inside `execute_convert`; the traced test above was added for it and verified
  to fail (`left: 1, right: 0`) against a stage-then-cast `execute_convert` before
  being restored. Its minor items: a stale `get_read` doc line (fixed), and a
  plan line that still described `get` as renamed rather than deleted (the plan's
  §5 now records the deletion, matching the code).

## Verification conclusions and constraints

- Values: every destination dtype casts a compact, transposed, offset and
  negative-stride view exactly like the compaction-then-cast shape it replaces;
  the same-dtype view cast returns fresh compact storage.
- Refusals: NaN, ±inf and out-of-range values refuse identically through both
  entries, and a NaN outside a sliced view no longer refuses the cast.
- Staging: `cast_read` of a view leaves `CpuEngine::read_stagings` at zero, while
  the previous shape (`to_contiguous_read` + `cast`) raises it by one; that pair
  is the control for the count test. Local measurement of
  `route_matrix/cast_f64_f32` (serial engine, 4 cores pinned, 4096 f64 elements):
  the staging shape 2.37 µs, `cast_read` 1.12 µs, owned `cast` 0.81 µs. This is a
  dev-machine number, not the hosted benchmark runner, and it replaces the #2285
  profiling hypothesis with a measurement of the copy the hook removes.
- CUDA: compile-checked only. The A100 lane was not run on this host; the change
  there is the mechanical forwarding D6 describes.
- Still open in #2223: the owned-twin deletion (PR 3) and the caller migration
  (PR 2), including the einsum diagonal copy and the golden error-parity table.
