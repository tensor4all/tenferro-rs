# Incremental Householder QR small-append performance

**Issue:** [#1750](https://github.com/tensor4all/tenferro-rs/issues/1750)

## Goal

Explain and remove tenferro-owned overhead in the initial-rank-5 plus nine
width-3 appends (final rank 32) at 64, 128, and 256 rows. Do not add an
SRC-specific API.

## Measurement before optimization

Keep the frozen #1735 benchmark unchanged. Add a separate benchmark with the
exact downstream schedule and report these timed components independently:

1. time only compact `append_columns` within the complete workflow;
2. time only `r()` within the same workflow;
3. time only the newly appended `q_columns` within the same workflow;
4. time the complete append + R + selected-Q workflow.

The component lanes execute the identical interleaved operation sequence and
sum only their selected `Instant` intervals. This keeps state/cache conditions
matched, so `complete ~= append + R + selected-Q` is the reconciliation equation.

Run the four components through one reused concrete backend session. Add a
concrete complete-workflow lane that opens a backend session for every operation,
matching the rejected adapter before its Matrix conversions. Add an eager
complete-workflow lane using one reused `EagerRuntime`; these lanes isolate
session-entry and eager-dispatch cost. Pin `QrGauge::Raw` in every
output lane, matching the rejected downstream adapter; remeasure an accepted
adapter if it requires another gauge. F64 and C64 are performance dtypes so
the real matrix path and complex SRC sketch path are both represented; the
existing #1735 tests retain correctness coverage for F32/F64/C32/C64.
Input generation, initial factorization, and final correctness checks remain
outside timing. The initial reproducer benchmark is CPU-only, one-thread, and
release-mode because #1750's measured regression is the CPU tensor4all adapter.
The existing #1735 gate retains CUDA correctness and scaling coverage; add a
matching CUDA small-workflow lane only after a measured CUDA regression or a
shared API candidate makes it relevant.

All lanes live in one binary and commit. Each timing sample follows an untimed
50 ms pinned-core clock warmup and is a fixed batch of independently reset
nine-append sequences. Choose one batch size before measurement and use it for
every lane;
a process median below 1 ms is `INCONCLUSIVE`. Record at least seven process
medians, alternate lane order, and classify process-median CoV above 10% or a
CPU-frequency validity failure as `INCONCLUSIVE`; the sequential runner owns
this cross-process classification from the benchmark records. Record every raw
sample, provider, dtype, row count, schedule, batch size, thread settings, and
the exact tenferro commit; the benchmark refuses to run without the commit. Correctness checks use reconstruction and
orthogonality residuals after the final append.

## Decision rule

Classify costs before changing APIs or kernels. Compare row scaling and verify
that complete-workflow time agrees with the separately measured component sum:

- If all concrete components have a similar size-insensitive floor, optimize
  shared dispatch, validation, allocation, or session plumbing before kernels.
- If append grows with matrix work and dominates after subtracting that floor,
  optimize the compact append backend.
- If R extraction dominates, optimize or narrow the triangular-factor access.
- If selected-Q dominates, optimize reflector application for the requested
  range.
- If the fresh-session-to-reused-session delta dominates, keep one session or
  add a generic batching/fusion seam rather than changing reflector kernels.
- If the eager-to-fresh-session delta dominates, optimize eager dispatch rather
  than the backend session or reflector kernels.
- If downstream inverse-adjoint maintenance dominates, design a generic
  triangular solve/update operation only after a matching benchmark exists.

Promote a candidate only if its paired complete-workflow median improves by at
least 10%, no component median regresses by more than 5%, and correctness,
dtype support, AD semantics, and placement remain unchanged. Re-run the
tensor4all SRC comparison before recommending replacement of BCGS2.

## Known design mismatch to verify

The #1735 design said append would consume and return state where Rust permits
buffer reuse, but the shipped concrete API takes `&self` and the CPU providers
copy the old packed state and coefficient vector into new outputs. The #1735
performance gate timed append only on much taller SRC-derived matrices and
explicitly omitted inverse-adjoint maintenance. It therefore established the
reflector algorithm's scaling, not competitiveness of the intended small public
workflow.
