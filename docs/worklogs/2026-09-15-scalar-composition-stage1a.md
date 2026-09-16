# Work log: scalar composition stage 1a — one dtype tag, one numerical body

Session summary for the first part of the staged scalar-composition plan recorded
in `docs/design/scalar-composition.md`. Verified against `origin/main` at
`28dfc7e3` and rebased onto `bca2d54a` before the PR checks.

## Context read

- `crates/tenferro-tensor-core/src/lib.rs` — `DType`, sealed `TensorScalar`,
  `HostTensor<T>`, `HostTensorView`.
- `crates/tenferro-tensor/src/types.rs` — the second `DType`, the second
  `TensorScalar`, its `private::Sealed`, and `impl_tensor_scalar!`.
- `crates/tenferro-tensor/src/lib.rs`, `src/backend.rs`, `src/error.rs`,
  `src/tests/backend_default_read_tests.rs`,
  `crates/tenferro-runtime/src/{error,typed_tensor}.rs`,
  `crates/tenferro-gpu/src/cubecl/tests/mod.rs` — the four `core_dtype()`
  copies and their 44 call sites.
- `crates/tenferro-internal-cpu-kernels/src/elementwise.rs` — `replay_binary`,
  `replay_scalar_left`, `replay_scalar_right`, `typed_*_with_pool`,
  `PooledUninitOutput`.
- `crates/tenferro-cpu-basic/src/buffer_pool.rs` and
  `src/pooled_uninit_output.rs` — `PoolScalar` with its sealed per-type pool
  fields and `PooledUninitOutput`.
- Issues #1787, #1785, #1788, #1789, #1790, #1793, #1706; `REPOSITORY_RULES.md`.

## Decisions

- **One tag.** `tenferro_tensor::DType` is a re-export of
  `tenferro_tensor_core::DType`. The two enums had identical variants and
  identical derives, so the split bought no layering freedom and forced every
  boundary between the two crates to convert a tag to itself. The seven preset
  types were enumerated twice; now they are enumerated once.
- **Delete the conversion layer, not just the enum.** With one tag, `core_dtype`
  is the identity, so all four hand-written copies and their 44 call sites are
  removed instead of being rewritten.
- **Keep `TensorScalar::dtype()`.** It has 605 call sites in crate source and
  830 with tests. Removing it is not a reviewable first step; what changes is
  where the tag is defined, not whether the accessor exists.
- **Drop the pool bound from the numerical body, keep it on allocation.** The
  replay helpers wrap `strided-kernel`'s `zip_map2_into` and `map_into`, which
  need only `Copy`, yet they required the sealed `PoolScalar`. `PoolScalar` was
  not used inside the replay body. `PooledUninitOutput` does allocate through the
  sealed typed pool and keeps its bound. The numerical body is therefore
  element-type agnostic, and the sealed pool stays a resource boundary owned by
  #1789 rather than being opened here.
- **Two parts, recorded in the design doc.** 1a is the mechanical
  simplification above. 1b adds the open scalar contract, the single preset
  table, the shared erased dispatch, and the external proof crate.

## Rejected alternatives

- Removing `TensorScalar::dtype()` (605 call sites) as part of this change.
- Opening `PoolScalar` or the typed `BufferPool` to external types here; that is
  the #1789 resource-ownership decision and would pull a resource redesign into
  a mechanical simplification.
- Making `PoolScalar` a storage minimum so external types could reuse it.
- Adding a parallel extension-scalar trait beside the existing machinery.

## Verification

- `cargo check --workspace --all-targets`: 0 errors, 0 warnings.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-tensor tensor_shape_and_dtype_cover_all_variants'`: passed (fmt,
  clippy with `-D warnings`, focused test).
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --dry-run --llm-skipped-reason "local deterministic review"`: pass.
- Measured net change: 45 added, 113 deleted, net −68 lines across Rust sources.
  The 100-line stage 1 target is not yet met; part 1b still adds the open
  contract and its tests, and the design doc records that the number is a
  measured result rather than a promise.

## Residual risk

- `crates/tenferro-tensor/tests/ui/storage` reports 10 of 14 `trybuild`
  mismatches in this worktree both with and without these changes, because the
  committed `.stderr` fixtures were generated under a different path prefix.
  Pre-existing environment artifact, not a consequence of this diff.
- The `strided-kernel` replay path is left with the same runtime behaviour; only
  the bound changed. No machine-code sharing between scalar types is claimed.
- Stage 1 does not yet prove that an external scalar traverses the erased
  `Tensor` layer, provider dispatch, or the AD graph.
