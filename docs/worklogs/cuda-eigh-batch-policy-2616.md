# CUDA eigh batch policy (#2616)

- Issue: <https://github.com/tensor4all/tenferro-rs/issues/2616>
- Branch: `fix/2616-cuda-eigh-batch-policy`

## Decision

`EighOptions.batch_policy` (`EighBatchPolicy::{AllowPerMatrixFallback, RequireBatched}`) makes
"one cuSOLVER launch per batch" an explicit, refusable contract, and
`tenferro_linalg::cuda_batched_eigh_available()` exposes the capability ahead of the call. The
default stays `AllowPerMatrixFallback`, so routing is unchanged; `RequireBatched` turns the
per-matrix `syevd` loop into a typed `Unsupported` error that names the missing capability and the
fallback it refused.

`RequireBatched` is rejected on a real batch by every provider that has no batched routine (CPU
default hooks, CPU lane). The issue text names only the CUDA capability check, but a
backend-neutral option that one provider silently ignores is the reported bug in another
provider's clothes, and R-api-design-01 forbids an implicit fallback.

## Alternatives rejected

- A new `EighDriver` variant: launch count is not an algorithm family, and a provider without a
  batched routine cannot honour it.
- Making `cusolverDnXsyevBatched` a mandatory load symbol: turns a lost optimisation into a total
  cuSOLVER load failure (#2413).
- `-> bool` capability query that folds a cuSOLVER load failure into `false`: conflates "no
  capability" with "no library".
- A new public `CudaLinalgExt` trait for one bit (`mod gpu` is private, so the crate root function
  is the smaller surface).

## Known limits

- Traced pruning of `Eigh` to values-only `EighVals` drops the policy; the values-only op carries
  no policy, so a `RequireBatched` eigh whose values feed AD may recompute per matrix.
- Every toolkit installed on the development A100 provides `cusolverDnXsyevBatched`, so the
  unavailable path is covered at the selector level (version/symbol arguments) plus #2413's
  recorded cuSOLVER 11.6.1 evidence, not by a live old-library run.

## Verification conclusions

- The default policy reproduces the pre-#2616 routing at every order and batch size, and
  `RequireBatched` changes no result it can honour: on an A100 with cuSOLVER 12.6.2 the strict and
  permissive policies return the same values and vectors.
- The refusal is reached, and names `cusolverDnXsyevBatched` plus the declined
  `AllowPerMatrixFallback`, at the selector level for both inability reasons (unresolved symbols,
  version below 11.7.1), on the CPU default hooks, and in a CPU lane; a single matrix and a batch of
  one are never refused.
- Pinned by `api/tenferro-linalg{,+autodiff-cuda}.txt`: the only additions are `EighBatchPolicy`,
  `EighOptions::batch_policy` (field and setter), and the `cuda`-gated
  `cuda_batched_eigh_available`. `EighBatchPolicy` stays out of `prelude`, matching `EighDriver`
  and `EighGauge`.

## Constraints and unverified parts

- The `Ok(false)` answer of `cuda_batched_eigh_available`, and therefore the live refusal on CUDA,
  could not be exercised on the development A100: every installed toolkit provides
  `cusolverDnXsyevBatched`, and the one older system cuSOLVER (11.4.3) lacks a symbol this crate
  requires, so it fails to load instead of reporting a lower version. The unavailable branch is
  covered by the selector unit tests and by #2413's recorded cuSOLVER 11.6.1 run.
- The guide and design prose state the policy is dropped by `Eigh`→`EighVals` pruning; that
  is the shipped behaviour, not a placeholder for a later fix. Revisiting it means adding the
  policy to the values-only op.
