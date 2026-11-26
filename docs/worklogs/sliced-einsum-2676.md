# Sliced einsum v0 (#2676)

## Decisions

- **Plan object, opt-in.** `SlicedEinsumPlan` with `SliceOptions`, beside
  `ConcreteEinsumPlan`, rather than a field on `EinsumOptions`: the slice count
  and per-slice cost are read before execution, and an unmet target is a
  planning error (#2676 discussion, settled by the maintainer). Existing
  einsum entries never slice.
- **Target unit: log2 elements** of the largest intermediate of one slice.
  It is omeco's `sc_target`, so the slicer optimizes the checked quantity,
  and it does not depend on dtype. Bytes were rejected for this reason; the
  conversion is one `log2(bytes / size_of)`.
- **The bound is checked by tenferro, not trusted from omeco.** Per-slice
  costs (largest intermediate, peak live intermediates, flops) are computed on
  the tree tenferro will run, at the true extents. omeco's slicer also counts
  operands and can stop short or over-slice; the check catches both.
- **Explicit orders are kept.** For `Tree`/`Nested`/`Path`/`False` the slicer
  runs without annealing (`niters = 0`) and only its labels are used; `Auto`
  takes the slicer's re-annealed tree.
- **Output labels.** omeco 0.2.6 has no way to exclude labels and picked an
  output label in a test network (`abc,cde,efa,bdf->b`, target 2^7). Instead
  of vendoring or patching omeco, planning reruns the slicer on the network
  without output labels and with the target lowered by log2 of the output's
  size, which is a sound bound. A label-exclusion option in omeco would make
  this exact; that is an upstream request, not a blocker.
- **Slice views keep rank.** A sliced axis stays with extent 1 instead of
  being removed, so the per-slice subscripts are the original ones and
  repeated or broadcast labels need no rewriting. Host views are rebased to
  offset 0 so the engine's offset-keyed binary plan cache sees one layout.
- **Accumulation through the existing `_into` path**: first slice with the
  caller's `Dest` mode, the rest with `beta = 1`, via the plan's
  `run_reads_into_accum`. No new kernel.

## Verification conclusions and constraints

- Correctness against the unsliced einsum on random hyperedge networks for
  f64/f32/c64/c32 (rel. error ≤ 1e-12 / 1e-5), explicit labels, every
  contracted label sliced, an extent-1 label, `ConjMask`, explicit orders with
  a target, and the slicer under a target two bits below the unsliced tree.
- Not bit-identical to the unsliced contraction in general: the sum over the
  sliced labels moves outermost. Bit-identical on integer-valued data and
  between reruns, and ascending ranges into one destination are bit-identical
  to the whole loop.
- Memory: the CPU executor's intermediate arena per slice equals the planned
  peak (`tests/sliced_memory.rs`), and a warm loop allocates no
  intermediates. A thread's first GEMM reserves a 512 MiB thread-local page
  pool inside faer's x86 GEMM backend (`private_gemm_x86`); it is per thread,
  independent of slicing, and the memory test warms the thread first.
- Plan reuse: one CPU numerical-plan miss for all slices, and no new engine
  binary-plan misses on a second run (also for a two-operand network, where
  the engine cache keys on the offset).
- Constraints: slicer-chosen labels are not reproducible across plannings
  (omeco 0.2.6 breaks ties in `HashMap` order); traced/eager (AD), CUDA plan-key
  checks and output-label slicing are not in v0.
