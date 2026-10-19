# GPU performance invariants strengthened by #2112

The umbrella [#2112](https://github.com/tensor4all/tenferro-rs/issues/2112)
names four invariants and states that the work must strengthen them in
`docs/design/`, not only in code comments. This records what the merged work
([#2127](https://github.com/tensor4all/tenferro-rs/pull/2127)) actually changed
for each, with the measured evidence.

## I1 — boundary: the runtime does not materialize by default

An executor receives `TensorRead`s and decides per operation whether it consumes
the strides or materializes once (`borrowed-extension-reads.md`).

- **Changed**: the eager gather took *every* view operand through
  `to_contiguous_read` before the kernel, so a pooled or borrowed embedding table
  was copied once per call. `Engine::gather_read` now takes both operands
  as reads: the default materializes and delegates (so every backend keeps
  working), and the CUDA session binds the operand where it is when the device can
  read it through its own strides, falling back to the owned path otherwise. The
  eager executor passes the reads through.
- **Evidence** (`gpu/perf_issues`, A100, f32, hidden 1024, 8 tokens, warm, 128
  iterations per sample): 8192 rows 96.1 us -> **18.1 us**, 65536 -> **17.0 us**,
  248320 rows 2245.3 us -> **19.2 us**; the `cudarc-row-copies` reference is
  24.6 us. The cost no longer depends on the table size.
- **Rule for new code**: an operation family that can consume a borrowed operand
  takes reads and binds them with their own strides; only the fallback path may
  call `to_contiguous_read`.

## I2 — resource ownership: the owner releases, and a drop leaves no silent retention

An allocated resource is released by its owner; when the owner is gone the memory
returns to the driver, or the caller gets a typed `Err`. Never a silent
retention, and never `Ok` for an allocation that did not happen.

- **Changed**: an output allocation whose tensor is dropped returns its handle to
  a size-keyed free list owned by the runtime that owns the device, bounded at
  64 MiB per runtime (#2156 moved it out of the former process-global
  `OUTPUT_POOLS`), and `alloc_output` takes an exact-length handle before asking
  the device thread. Reuse is write-after-write on one stream, which CUDA orders,
  and a handle only reaches the list once no live tensor references it.
- **Evidence**: `add` on a decode-sized tensor 12.0 us -> **9.94 us** per op
  (retained outputs) and **9.11 us** with the result dropped per op, against a
  measured launch-only floor of **5.36 us**.
- **Still open (recorded, not claimed)**: the trim arm
  (`pool_retention_f64_trim-after-drop`) shows the pinned 0.10.2 line's
  `memory_cleanup` does not return this sequence's pages to the driver, and the
  dedicated allocation mode that would (`MemoryAllocationMode::Dedicated`) exists
  only in the 0.11 fork release.

## I3 — plan identity: a plan is identified by its effective geometry

A prepared plan is identified by its effective geometry (dtype, extents, strides,
*effective* alignment, operators) and by nothing about the operand's
representation (owned vs borrowed view).

- **Changed**: the cuTENSOR plan key and descriptor took their alignment
  requirement from the operand's representation (owned -> the 256-byte allocation
  alignment, view -> the scalar size), so one logical contraction kept two plan
  entries and cuTENSOR was told a weaker alignment for the view. The requirement
  now comes from the resolved effective address
  (`effective_descriptor_alignment`), bounded by the allocation promise and
  floored at the scalar size; the kind-based helpers and the now-tautological
  validation are gone.
- **Evidence**: `cuda_cutensor_plan_is_shared_by_an_owned_and_an_equivalent_view_operand`
  (A100) contracts the same geometry owned and through a view with the same
  effective strides and asserts one plan entry with a hit for the view;
  `alignment_tests` pins the address-to-alignment rule without a device.

## I4 — degenerate algebra: rewritten once, in the semantic layer

A `dot_general` with no contraction semantics is rewritten before it reaches a
backend, once, in the semantic layer.

- **Changed**: nothing rewrote those forms, so every backend ran its general
  contraction machinery over a degenerate shape. `rewrite_degenerate_dots` now
  runs unconditionally in `optimize_exec_program`: a pure Hadamard becomes
  `Multiply`, an outer product two `BroadcastInDim`s plus `Multiply`, a scalar
  operand a `Transpose` plus a broadcast `Multiply`, and a fully contracting form
  `Multiply` plus `ReduceSum`. Conjugation is preserved as an explicit `Conj`, and
  a shape the rewrite does not model falls through unchanged.
- **Deliberately not gated** by `OptimizerConfig::dot_decomposer`: that flag
  selects canonicalization, which materializes layouts, while this rewrite
  materializes nothing. `OptimizerConfig::VERSION` moved to 3.
- **Evidence**: tests assert the lowered op kinds and that no degenerate
  `dot_general` survives; the runtime suite is 426 passed / 0 failed. The
  benchmark rows for the acceptance (`hadamard_f32_1024x1024_dot-general` against
  `..._multiply`) are measured in `gpu/perf_issues`.

## What this record does not claim

Item 7's acceptance (per-op host cost within 1-2 us of the measured launch-only
floor) is **not** met: the measured floor is 5.36 us and the eager per-op cost is
9.11 us, the remaining difference being the eager layer's per-op value/tensor
bookkeeping. Item 3's dedicated-allocation half needs the 0.11 pins. Both are
recorded in the umbrella as follow-ups, not as landed work.
