# Tight batched CPU linalg loops and fused solve lowering (issue #1878)

## Decisions

- Batched LAPACK `lu_factor`, `lu_solve_prepared`, `triangular_solve`,
  `full_piv_lu` solve, `eigh`, `eigh_values`, `svd`, `svd_values` and
  `svd_full` now run one workspace query per call, reuse pooled scratch across
  the batch and write results straight into the batched outputs. No per matrix
  tensor or `Vec` is allocated inside the batch loop. Threading stays with the
  provider; the loops are sequential over the batch.
- `lu_solve_prepared` fuses the pivot apply into the provider solve: one
  `getrs` per matrix for every flag pair. The conjugate only case solves with
  `'N'` on conjugated right hand sides and conjugates the result back.
- faer `lu_factor` and `lu_solve_prepared` moved to packed kernels under the
  same discipline. The other faer paths (solve, triangular, full pivot, eigh,
  svd) still go through the existing batched result helpers.
- Traced `solve` and tracked eager `solve` emit a single `LuFactorSolve` op
  with outputs `(x, packed_lu, pivots)`. On CPU it is one fused kernel whose
  factor scratch becomes the LU output, so a primal only program pays no more
  than the plain `Solve` kernel. The backend default, used by CUDA, composes
  `lu_factor` and `lu_solve_prepared`.
- `LuFactorSolve` is never pruned to `Solve`. The first version pruned it when
  the factor outputs were unused, but traced AD prunes the source program
  before differentiating it. The prune stripped the factors and the reverse
  graph refactored A, three solves in total instead of one factorization. With
  no prune, reverse mode reuses the saved factors and factors A exactly once.
  The untracked eager surface still constructs `Solve`, because nothing
  differentiates it.
- The eager backward rebuild overhead is out of scope here and stays with
  #1758.

## Baseline fixes found by the new tests

- Accelerate `getc2` leaves the last `IPIV`/`JPIV` entries unset; the in place
  wrapper now sets them to N.
- faer `lu_factor` on an empty batch returned one parity entry because the
  batch count was clamped to at least one. An empty batch now has no parity
  entries; an unbatched empty matrix still has one.

## Verification conclusions and constraints

- Residual oracle sweeps cover batch sizes including empty, n in 1, 2, 3, 8,
  33, f32/f64/c64/c128, every transpose and conjugate flag, forced pivoting and
  a singular batch member, on both the faer and LAPACK providers, with faer and
  LAPACK agreeing on solutions and pivots.
- Traced VJP and JVP of `solve` with respect to both A and B match central
  finite differences for real and complex inputs, and the reverse graph
  contains exactly one factorization. Tracked eager backward on a batched
  pivoting system matches finite differences.
- No wall clock measurement was taken; the coordinator owns timing. Benches
  were only compiled.
- CUDA still runs the split default composition for `LuFactorSolve`.
