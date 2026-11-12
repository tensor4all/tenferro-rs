# `EinsumOptimize::False` folds left to right again

## Decisions

- Fix the plan resolution, not the documentation. Three places already state the contract —
  the `EinsumOptimize::False` variant doc, `EinsumOptions::optimize`, and
  `docs/guides/einsum.md` — and the existing unit test for `False` already expects the fold's
  pairs for three operands. The implementation was the outlier.
- Keep `EinsumPlanSpec::LeftToRight` and change only how it resolves into a tree. The spec is
  built while shapes may still be symbolic, so it cannot carry fixed pairs; changing the
  resolution leaves the wire payload, its hash, its equality and the plan-cache identity
  untouched.
- Do not change `jax_path_to_v1_pairs`. `EinsumOptimize::Path` documents append-at-end as its
  meaning, and making the resolver keep a result at the front would change a public contract
  (`Path`'s) instead of the one that was broken (`False`'s). `LeftToRight` stops routing through
  the JAX position list at all.
- Rejected: documenting `False` as pairwise. The pairwise order contracts operands that share no
  label as soon as four operands are involved, which is never what "no optimization, written
  order" can be taken to mean.

## Verification conclusions and constraints

- With the fix, `False` on `"ij,jk,kl,lm,m->i"` (d=256, f64, serial engine, release) takes
  3.33 ms against 186.9 ms before, and its result is bitwise identical to the same order written
  as an explicit `EinsumOptimize::Path`. For `"ij,jk,kl,lm,mn,no,o->i"`: 5.52 ms against 78.5 ms.
- Planning-level regression guards: the exact five-operand pair sequence, and a property test
  over four chain notations that no step contracts two operands with disjoint label sets. Both
  fail on the pre-fix code. Both are classified `host` in `scripts/ci/cuda_test_partition.tsv`:
  the CUDA archive job validates its inventory exhaustively and fails on an unclassified test.
- Not covered here: whether the automatic planner still wins on other shapes, and the GPU
  providers, which share the resolved tree but are validated by their own lanes.
