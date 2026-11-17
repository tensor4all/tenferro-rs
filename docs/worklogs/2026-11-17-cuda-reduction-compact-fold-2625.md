# One pass per contiguous run: the CUDA reduction's compact fold (#2625)

Issue: tensor4all/tenferro-rs#2625. Branch `perf/2625-cuda-axis1-reduction`, base `origin/main`
`aebe8cb3`. Host: A100 80GB PCIe, driver 580.178.04, CUDA 12.6.

## Decisions

- `reduce_axes` now derives the shape and the pass list it hands the catalogue, instead of folding
  one logical axis per launch. Two rules, both from the compact column-major layout a `Tensor`
  always has:
  - **merge**: a maximal run of consecutive reduced axes is one axis of the run's product extent.
    `stride[k + 1] == extent[k] * stride[k]`, so the run addresses the same elements in the same
    order; folding it once is the same fold with one launch instead of N, and when the run is the
    contiguous one the plane strategy coalesces.
  - **split**: a strided axis whose length has a factor pair `a x b` with `2 <= a, b <= 32` becomes
    `[.., a, b, ..]`. The plane strategy puts the lanes on the reduce axis, which on a strided axis
    reads a row apart; both halves fit one plane, so both passes stay on the unit strategy, whose
    lanes vary the contiguous output axis. `split_policy` already splits at `len >= 512`, so this
    trades a partials pass for a keepdims intermediate instead of adding a launch.
- Pass order stays descending, and a split inserts the outer factor right after its inner one, so a
  later pass reads the smaller intermediate. The re-factoring is only valid on a compact
  column-major region, so `reduce_axes` compares the read's real element strides
  (`CudaEngine::strides_of`, the accessor this file already uses) with `compact_strides(shape)` and
  keeps the logical shape and order when they differ — the derivation is checked, not assumed.
- Untouched on purpose: `reduce_axes_wide` (rank > 8), the empty-fold and zero-extent branches, the
  keepdims destination trick (#2406), one `prove_launch` per standalone call, `REDUCTION_PLANE_WIDTH`
  (the literal 32 became a named constant so the split rule and the planner cannot drift apart), and
  the `t4a-cuda-kernels` pin. No ABI, no public API, no new kernel.
- Sum-of-squares composition is unchanged and still keyed to the emitted order: the first pass folds
  squares, later passes add.

## Verification conclusions and constraints

- Tests: new `crates/tenferro-gpu/tests/reduce_axes_layouts.rs` (6 cases: both axes over a 7-shape
  boundary matrix, every multi-axis choice and repeated axes, a transposed view, `norm_squared` on
  512x512/1024x1024/[17,3,5], a split above a lower pass with one and with two splittable axes, and
  determinism), plus a host-lane unit test of the presentation arithmetic in `structural/tests.rs`.
  Registered in `scripts/ci/cuda_test_partition.tsv` (6 `gpu` rows, 1 `host` row). Existing
  `seam_full_reduction` (4), `seam_empty_reduction` (3), `seam_structural` (33), `blas1_cuda` (12)
  and the crate's 48 unit tests pass.
- The matrix discriminates: pushing the logical axis index instead of the merged one fails all six
  cases, and shifting the lower passes instead of the already-planned higher ones fails the
  `a_split_axis_above_a_lower_pass_keeps_every_index_right` case (found by the reviewer, fixed, and
  pinned by that case).
- Fixture note: the test values are `1.0 +- 0.01` and the sum/prod/sumsq tolerance is relative to the
  CPU value. With `[0.5, 1.5)` values a product over a million terms decays to ~1e-18, and a
  unit-magnitude floor in the assertion would have accepted an identically zero result (reviewer
  finding).

- `cargo fmt --all --check`, `cargo clippy -p tenferro-gpu --features cuda --all-targets` (only the
  workspace's pre-existing manifest warnings), `check-duplicate-test-messages.py`, the 514
  `scripts/ci/tests` cases, `check-operation-categories`, `audit-engine-entry`, `check-contraction-kernels`,
  `test-doc-consistency`, `check-public-boundary-inventory` pass.
- `RUSTDOCFLAGS=-D warnings cargo doc -p tenferro-gpu --features cuda` fails on a broken
  intra-doc link in `cuda/identity.rs:99` (`CudaEngine::supports`); it fails identically at the base
  commit, so it is pre-existing and not touched here.
- Device evidence (`nsys`, 20 calls of `reduce_min`/`reduce_sum` on `[1024, 1024]` f64, axis 1, all
  four arms in one quiet window): before, one `t4a_cuda_reduce_partials` (16.5-19.4 us) plus one
  `t4a_cuda_reduce` (4.0-4.3 us) per call; after, two `t4a_cuda_reduce` and no partials.
- Harness pairing (`pair-d`, direct arm, medians of 15 samples): `reduce_sum` 36.87 -> 25.26 us
  (-31.5%), `reduce_prod` 36.44 -> 25.87 (-29.0%), `reduce_min` 37.06 -> 35.51 (-4.2%),
  `reduce_max` 37.10 -> 35.38 (-4.6%), `norm_f64_512x512` 57.73 -> 43.19 (-25.2%); the eager arm
  moves the same way, and `pytorch-cuda` is unchanged (7.0-7.3 us, norm 9.3-9.4).
- Release-profile probe against `main` (200 reps, CPUs 0-1, one-launch `abs` floor unchanged at
  18.2 -> 18.0 us): `sum` 36.84 -> 25.96 us, `prod` 36.66 -> 25.97, `min` 37.14 -> 26.86, `max`
  37.50 -> 26.96, both-axes 47.02 -> 26.66, `norm_squared 512x512` 37.31 -> 23.32.
- A discarded pair is kept for the record: `pair-c-base`/`pair-c-after` both measured 25-26 us for
  `sum` because cargo fingerprints a path dependency by mtime under the symlinked
  `extern/tenferro-rs`, so flipping the symlink reuses the previous arm's binary. `pair-d` touches the
  measured sources first; `pair-d-base` reproduces the independent baselines (37.0-39.2 us).
- **The issue's baseline revision costs 11.7 us per `min`/`max` pass where `sum` costs 6.2 us**, so
  the same two-pass fold leaves extrema's device total unchanged there (23.2 -> 23.5 us) even though
  the partials pass is gone. `main` does not have that asymmetry (6.6 vs 6.0 us), so the wall-clock
  win for extrema only appears on `main`. This is a property of the pinned revision's kernel cost,
  not of the fold, and it is why the measured `reduce_{min,max}` cells sit above the predeclared
  24-32 us band while `reduce_{sum,prod}` land inside it.
- Limits kept on purpose: a strided axis with no factor pair inside one plane (e.g. 1000, 1031) is
  left to the plane strategy, as is any axis below the planner's 512 split floor; merging is refused
  when an extent product overflows; a presentation that would exceed `MAX_RANK` falls back to the
  logical shape and order.
- Association changes for merged and split sums/products, within the `n * u` bound the tests assert;
  extrema are order-independent; every fold stays deterministic across calls.
