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

## Remaining, in order

1. `gather`/`scatter` plan mapping, and the float-start exactness check for `dynamic_slice`.
2. `TensorDot` on `cuda_next::Cutensor`, extending `Contraction` with `beta`.
3. The neutral kernel/vendor vocabulary plus the CUDA lowering; `structural.rs` loses every
   `t4a` import.
4. `Engine`/`ExecutionEngine`/`TensorBackendCapability` on the new engine.
5. C32/C64 and the per-op capability matrix with accepted/rejected tests.
6. cuFFT and cuSOLVER onto the new path.
7. U6: delete CubeCL, cut the feature/CI/doc/doctest surface over, fold the seam tests into
   the CI inventory.
8. U7: full local A100 gate, the 1-thread/1-worker comparison, independent review, one PR.
