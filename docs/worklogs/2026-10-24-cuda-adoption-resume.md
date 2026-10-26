# Resuming #2208: seam buffer, indexing slice, and the order of the rest

Branch `feat/2208-cuda-adoption` off `origin/main` `9f54e10da`. One PR at the end; no
topping-up pushes.

## Baseline (recorded before any change)

Host: one A100 80GB PCIe, driver 580.126.09, CUDA toolkit 12.6, cuTENSOR 2.5.0.
`cargo check -j 16 -p tenferro-gpu --features cuda-next --all-targets` clean,
`seam_storage` (4), `seam_transfers` (12), `seam_publication` (7), `seam_launch` (4),
`seam_structural` (13), `seam_vendor` (4), `seam_contraction` (4) green.

`seam_transfers::an_unserved_dtype_is_rejected_and_returns_the_package` was already red on
`origin/main`: PR #2262 gave Bool a transfer route and left the assertion that Bool is
refused. Replaced by a Bool round-trip test.

## The seam carries the allocation (commit 1083ff701)

`gpu::Backend` names the allocation handle a backend owns and `gpu::Buffer<B>` owns it
together with the identity every consumer needs (`domain`, `allocation`, `device`) and the
byte length. `CudaStorage` *is* that buffer, built by `fresh_storage(runtime, bytes,
allocation)` from the handle the runtime allocator returned, so a handle and its identity
cannot disagree. The seam's unused `Program`/`OpDescriptor`/`KernelSource`/`VendorOps`
placeholders were removed; they come back with their first implementor rather than sitting
unused beside a second typed path.

## The indexing slice (commit f33277ff4)

`TensorIndexing` on the new engine serves `slice`, `pad`, `dynamic_slice`, `concatenate`
and `reverse`. `reverse` is folded into strided materialization (negated strides plus the
offset that keeps the view in span), which is the same rule the earlier units used instead
of adding a kernel. Still refused: `gather`/`scatter` (need the `GatherConfig`/`ScatterConfig`
to plan mapping) and `dynamic_update_slice` (no windowed-write kernel exists; the CubeCL
path does not serve it either). Float start indices for `dynamic_slice` are refused until
the exactness check (`validate_float_indices` + `extract_invalid_float_index`) is ported.

## Why the surface comes before the generic seam pass

The approved design puts the seam reshape first. The order inside this branch is inverted
for two reasons, and the endpoint is unchanged:

1. The neutral kernel vocabulary must be written once. Adding `TensorIndexing` and
   `TensorDot` after the vocabulary exists would force a second pass over both.
2. The refactor is a move. Doing it over a complete surface means every family is converted
   once, with its tests already guarding it, instead of converting a family and then
   extending its converted form.

The seam still gets its CUDA implementor in the same PR: `Buffer<B>` already carries the
allocation, and `KernelSource<B>`/the vendor requests land with the CUDA lowering when the
vocabulary is complete. `structural.rs` then loses its `t4a` imports — that is the check
that the shared surface is really backend-neutral.

## Independent review of the indexing slice (gpt-6.1-sol, post mode)

The review found one blocker and several real defects; all were verified against the source
and fixed in the same branch, with the two new edge-case tests covering them:

- `dynamic_slice` derived the starts rank from the *output* and never checked the starts
  tensor, so a short starts tensor made the kernel read past its allocation. It now requires
  one start per operand axis.
- `gather` indexed an empty `window_dims` for a degenerate configuration (a panic, not an
  error) and converted `collapsed_slice_dims`/`offset_dims`/`start_index_map` into their
  complements unvalidated, which erased duplicate or out-of-range axes. Both lists are
  checked before any shape is derived.
- An empty domain reached the runtime as a zero-grid launch and was rejected; element-driven
  families now skip it (a reduction is the exception, since its identity fill is the
  kernel's). Padding an empty operand returns the zero-filled allocation, which is exactly
  what padding nothing means.
- `padded_shape` used `(n - 1) * (interior + 1) + 1` for an empty axis and evaluated
  `interior + 1` unchecked; it now matches the kernel planner's zero-extent base with checked
  arithmetic.
- `concatenate` accumulated the axis extent and the running offset unchecked, and reported
  mixed dtype/rank as `Unsupported`; it now uses checked arithmetic and the trait's
  `dtype_mismatch`/`rank_mismatch`/`shape_mismatch`/`axis_out_of_bounds` codes.
- `scatter` waited on the operand copy before enqueuing the accumulation; the copy is now
  only enqueued, since stream order already covers it.
- The seam doc claimed the transfer/publication/dispatch code is already reusable by a second
  backend; the claim now states what is true today (the allocation passes through the seam,
  the execution vocabulary lands with its first implementor).
- `fresh_storage` took the byte length as an argument and could disagree with the allocation;
  it reads the length from the allocation.

Deferred, recorded here rather than silently dropped:

- Every new indexing route goes through `Layout::new`, i.e. rank ≤ 8, while the kernel crate
  already ships the `_any`/`LayoutRef` entries. Rank > 8 must be routed through the general
  path before CubeCL is deleted.
- CubeCL serves F32/F64 indices for gather, scatter and dynamic slice and complex data for the
  supported indexing families; the new path refuses them until the float exactness check and
  the complex dispatch land.
- The inherited read methods stage borrowed views through `to_contiguous_read`, which refuses
  every borrowed view on this path, while the CubeCL path canonicalizes a same-device
  numeric view. The read hooks need the same treatment before the cutover.

## Independent review of the `TensorDot` step (gpt-6.1-sol, post mode)

Fixes made after the review, each verified against the shared contract:

- **Output order was wrong.** `dot_general_output_shape` (and therefore every other backend)
  puts the free axes first and the *batch axes last*; the mode assignment put batch first, and
  the batched test asserted its own wrong order. Both now follow the shared order, and the
  test's flat-index reference was corrected.
- The configuration is validated by the shared `dot_general_output_shape`, which also refuses
  an axis that holds two roles (a batch axis that is also a contracting axis); the local
  per-list validation that let one role overwrite the other is gone.
- `dot_general_read_into_accum` validates through `validate_dot_general_accumulation`, so a
  coefficient of the wrong dtype is reported instead of being converted to a real zero.
- An empty contraction (a zero contracted extent, or a zero output extent) returns the zeroed
  allocation for the overwriting forms and `beta * output` for the accumulating ones — scaled in
  place through the catalogue's `MUL`, which also serves the `beta = 0` fill — instead of
  reaching cuTENSOR's descriptors, which require positive extents.
- Both operand dtypes and residencies are checked before the destination allocation.
- A vendor failure keeps the seam's classification (`Unsupported` stays `Unsupported`) instead
  of being flattened into `RuntimeState` with a stringified source.
- The plan cache is bounded to 32 plans with oldest-first retirement, so a varying-shape loop
  cannot retain one device workspace per shape for the engine's lifetime. Reusing one
  stream-owned workspace instead of one per plan stays open and is recorded here.

## Float indices are checked, the transfer surface joins the engine contract

- `validate_exact_indices` runs the kernel crate's three validation entries
  (`init_validation` → `validate_float_indices` → read the flag) before a gather, a scatter or
  a dynamic slice reads a float index tensor, so a NaN, an infinity, a fraction or a value past
  the dtype's exact range is refused with the offending element's position. The ABI resets a
  separate value slot as a double: aliasing it with the flag word clobbered the verdict, which
  the first version did and a direct probe (`flag` left at `UINT32_MAX` for exact indices,
  `0` for a fractional one) caught.
- Found by the independent review of that step, and fixed: the two launch wrappers were `pub`
  on safe code and could read past a caller's buffer (now `pub(crate)`; the other dispatcher
  methods that predate this branch still are `pub` and deserve the same treatment); the
  ABI's `init_validation` launch and its value slot were unnecessary, since the host upload
  already writes the sentinel the validator compares against; the explicit `synchronize`
  before `download_bytes` was redundant, that helper already waits on its own stream; and
  `TensorDeviceTransfer`'s storage preconditions now report `RuntimeState` like the other
  backends instead of `Unsupported`. The test was strengthened to assert the window values,
  the offending element in the message, NaN/infinity/out-of-range for both float dtypes, and
  the scatter route.
- `TensorDeviceTransfer` is implemented for the new engine.
- `Engine`/`ExecutionEngine` are not: `Engine` requires `TensorBuffer`, `TensorFusion`,
  `TensorAnalytic` and `SessionCachedDot` in addition to the traits this path has, so the
  runtime wiring needs those four surfaces first — that is the next unit, together with the
  decision of which runtime engine id and registration the new engine uses.

## The engine surface and the dtype policy

- `Engine` is now implementable for the new engine: `TensorAnalytic` (ten unary operations from
  the catalogue's own `UnaryOp` plus binary power), `TensorFusion`/`TensorBuffer`/
  `SessionCachedDot` (defaults only) and `gather_read` are in place, so `Engine` compiles.
  Complex absolute value lives in `TensorElementwise`, not `TensorAnalytic`, and lands in the
  real dtype; `native_engine` keeps the trait default, since no typed session is exposed
  before the cutover. `ExecutionEngine` and the registration/preparer that the runtime's prepared-program
  model needs are not: they replace the CubeCL preparer rather than sitting beside it, so they
  belong with the U6 cutover where there is exactly one engine id and one registration.
- `kernel_dtype` no longer keeps a second dtype policy: it maps the seven preset dtypes and lets
  the kernel crate's planners decide what each entry serves, which is the single source the
  design asks for. A planner refusal about the operation or dtype now surfaces as `Unsupported`
  instead of `RuntimeState`.
- `seam_dtype` covers the policy: complex conjugation and the complex absolute value (including
  its real output dtype), C32 arithmetic, integer arithmetic, an analytic operation the
  catalogue refuses for integers reported as unsupported, the analytic mappings and float,
  scalar and integer power (including `0^0` and a negative exponent as a numerical failure),
  and the float-to-integer cast check.
- The cast routes needed one more guard the review caught: the ABI requires
  `t4a_cuda_validate_real_cast` before a floating source is converted to a signed integer
  (C++ float-to-int casts are undefined for NaN, an infinity and an out-of-range value), and
  widening the dtype policy had made the complex routes reachable too. `cast` now runs that
  check with the ABI's acceptance table (single precision below `2^31` exclusive, double
  below `2^31 - 1` inclusive, both below `2^63` exclusive, the real component of a complex
  source) and reports the offending element.

## The extension hook is designed, not implemented

`docs/design/cuda-extension-hook-2208.md` records what `tenferro-fft` and `tenferro-linalg`
call on the CubeCL raw session (counted per primitive), how each maps onto
`CudaRuntime::with_raw_locked`, the four ownership rules the hook has to keep, and the
sequencing: hook → fft → linalg → the `cuda::` flip with `ExecutionEngine`, the registration,
the capability table and the CubeCL deletion. That is the bulk of what is left, and it is a
move verified by the extension crates' own A100 tests rather than by new design.

- The cuTENSOR handle now owns **one** workspace, grown to the largest requirement seen, instead
  of one allocation per cached plan: a varying-shape loop used to retain a device buffer for
  every shape. Growing it waits for the stream first, since the old buffer's plans ran against
  it; the plan cache stays bounded at 32 plans.
- One review finding is accepted rather than fixed now, with the reason: the pre-existing
  `Dispatcher` methods stay `pub` because `seam_dispatch` exercises them as an external crate —
  the clean fix is to fold that test target into the crate when the CI inventory is rebuilt at
  U6.

## What rank > 8 will cost (recorded before starting it)

The kernel crate's general path does not take a wider layout in the launch parameter block: the
`_any` entries read the geometry — extents then strides, `LayoutRef::pack` order — from
**device memory**, and the mirrors (`CLayoutAny`, `CAxisValuesAny`, `CGatherPlanAny`) carry raw
device pointers to those arrays. So rank > 8 needs one shared mechanism in the launch core
before any family can use it:

1. pack a layout's geometry into a device allocation and register it with the scope;
2. open the raw scope with the operands *and* the geometry registered, so the mirrors can be
   built from `scope.device_ptr` inside it;
3. launch the `_any` symbol of the entry (`GeneralEntry::ALL` maps each entry to its general
   name; `KernelModules::function` binds by the fast name today and must choose by the planned
   `RankVariant`);
4. keep the geometry alive until the same completion as the operands.

That is the same shape for every family, so it lands once in `launch.rs`/`dispatch.rs` and then
each family converts mechanically.

## The general path: rank > 8 through the `_any` entries

`KernelModules` gained `in_scope` (bind a symbol from an entry's module, enumerate the launch
inside the runtime's raw scope, where a general-path parameter can name device addresses) and
`function_named`. `CudaEngine::materialize_view` now chooses by rank: at most
`t4a_cuda_kernels::MAX_RANK` axes keep the fast parameter-block layout, a wider view uploads its
extents and strides, registers them with the same scope as the operands, builds `CLayoutAny`
from `scope.device_ptr`, and launches `Entry::general().name()`. `transpose_read` and `reverse`
share that helper, so both serve a rank-9 view; `seam_structural`'s new
`a_rank_nine_transpose_uses_the_general_path` proves it against a host reference.

Found by the independent review of that step and fixed: the wide path returned early on an empty
launch before validating that both operands belong to this runtime; it now takes the operand
allocations first. `KernelModules::function` delegates to `function_named`, and the planning and
marshalling use `LayoutRef::new` directly instead of a temporary `WideLayout`. A second rank-9 test
covers `reverse`, which carries negated strides and a non-zero offset, so a dropped offset would
fail it.

`broadcast_in_dim` follows: it packs both layouts and its destination-axis map, builds
`CLayoutAny` twice plus `CAxisValuesAny`, and launches `t4a_cuda_broadcast_in_dim_any`. A rank-2
source into a rank-9 destination is the new test. The geometry helper (`WideGeometry`) is shared
by both families. Found by the independent review of that step and fixed: `compact_strides` now
reports an overflowing extent product instead of panicking (it is shared, so every caller benefits),
the destination is allocated only after the planner has accepted the dimension map, and the test
maps one axis past the fast limit and permutes rather than merely inserting singletons, so a
truncated destination or a transposed map would fail it.

`concatenate_copy` follows as well: the wide branch packs the destination layout once and each
input's layout per input, plans each placement through the same planner (so the grid geometry and
the validation come from one source), and launches `t4a_cuda_concatenate_copy_any`. A rank-9
concatenation is the test, along the leading axis so the inputs interleave (a wrong axis or offset
cannot pass as a flat append), with an empty input in the sequence. The review of that step also
caught that an empty input planned a zero grid the wide path still submitted — the runtime rejects
that — so an empty placement is planned for its validation and not submitted, keeping the running
offset. `dynamic_slice` (two layouts, the starts already device-resident) and the diagonal pair
(`extract_diagonal`/`embed_diagonal_copy`, two layouts and the axis ids) follow the same pattern,
each with a rank-9 test. The review of that step found two real defects: the embed symbol walks the
*source* elements, so its trailing length is the input length (the fast path already passed it;
the new wide branch did not), and the extraction's output shape assumed the diagonal keeps its
input axis — removing a lower axis shifts it down by one, which the shared derivation now accounts
for, and the test uses `(8, 1)` with unequal extents, a spectator axis and a non-trivial diagonal
placement so a shifted axis fails it.

`slice` and `pad` follow through one helper (`window_wide`): both take two layouts and two
per-axis value arrays through the general path, and their fast symbols' third array — pad's
`edge_high` — is planner-only, because `t4a_cuda_pad_any` folds that extent into the destination
layout and does not read it (the kernel source states this; passing it as a third mirror shifted
the whole argument list and wrote a partial window, which the rank-9 test caught before the
commit). A rank-9 slice with a column step and a rank-9 pad with interior padding on a non-singleton axis
plus edge padding on the trailing axis are the two tests (the review noted the first version of the
pad test used zero interior padding, which would pass even if the interior array were dropped). The
same review caught the empty-window shortcut bypassing the operand's domain validation, so both
windows now take their allocations before that early return.

`reduce` follows: the wide branch mirrors the fast one pass for pass — the keepdims destination
per reduced axis, the direct strategy's single launch, and the split strategy's partials pass plus
its combining pass, whose partials layout the crate computes and which therefore needs its own
packed geometry. A rank-9 sum over one axis, a multi-axis sum (which squeezes two unit axes) and a
product reduction are the tests. The review of that step found three real defects, all of which the
fast path shared: a multi-axis sum of squares squared the squared sums again (now only the first
pass folds the squares, the others add them, in both loops), every pass allocated input-sized
scratch even for the direct strategy (now only a split allocates, exactly its partials layout), and
the fixtures only ever exercised the unit strategy (now 33, 513 and 8449-long axes reach the plane
and split strategies, plus a two-axis sum-of-squares regression).

`gather` and `scatter_add` follow: three device-resident layouts each, plus the plan's arrays,
which `pack_gather_plan`/`pack_scatter_plan` return as one packed vector — so the four (gather) or
three (scatter) array pointers `CGatherPlanAny`/`CScatterPlanAny` take are offsets inside one
uploaded buffer, taken from the scope's base address. The test's window rides operand axis 8 while the start selects axis 7, so the window,
offset, start-map and slice-size arrays are all non-empty and distinct — the first version collapsed
every axis, which left all four pointers at the same address and would have passed with the order
swapped (the review caught that), and an additive scatter mirrors it with distinct window, update
and scatter arrays.

### The general path now covers the whole tensor surface

Every entry the tensor surface dispatches is either rank-free (its parameters carry no layout:
`CopyExact`, the elementwise family, `Compare`, `Select`, `Clamp`, `Convert`, `Tril`/`Triu`, the two
validation entries, and the fused-expression emitter, which takes pointers and an element count
only) or has a wide branch (movement, broadcast, concatenate, slice, pad, dynamic slice, both
diagonal entries, reduce and reduce-partials, gather and scatter-add). The `_any` entries in the
catalogue that no surface route names — the view entries, `FillOne`, `MatrixAdjoint`, and the linalg
auxiliaries — need nothing here until a route calls them.

What each remaining entry would need, for reference:

| Family | Geometry to pack besides the operand layouts |
|---|---|
| `extract_diagonal` / `embed_diagonal_copy` | the two axis ids |
| `dynamic_slice` | nothing extra (the starts are already device-resident) |
| `fill_zero_view` / `contiguous_to_view` / `strided_to_strided` | nothing extra |

Each is a `_wide` variant of its dispatcher wrapper plus a branch in the tensor-surface caller,
verified by extending the rank-9 test to that family. Nothing about them is a design question.

## Remaining, in order

1. The rank>8 `_any`/`LayoutRef` routing for the indexing and dot families (the float-index
   exactness check landed).
2. `TensorDot` on `cuda_next::Cutensor`, extending `Contraction` with `beta`.
3. The neutral kernel/vendor vocabulary plus the CUDA lowering; `structural.rs` loses every
   `t4a` import.
4. `Engine`/`ExecutionEngine`/`TensorBackendCapability` on the new engine.
5. C32/C64 and the per-op capability matrix with accepted/rejected tests.
6. cuFFT and cuSOLVER onto the new path.
7. U6: delete CubeCL, cut the feature/CI/doc/doctest surface over, fold the seam tests into
   the CI inventory.
8. U7: full local A100 gate, the 1-thread/1-worker comparison, independent review, one PR.
