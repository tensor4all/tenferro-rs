# Linalg prepared solve work log

Date: 2026-06-02

## Session summary

This change reshapes linalg solve-related execution so eager, traced, and AD
paths can reuse one LU factorization instead of repeatedly materializing public
`P/L/U` outputs:

- traced and eager `solve` now emit internal `LuFactor` plus
  `LuSolvePrepared`,
- traced `slogdet` consumes packed LU and parity instead of public `lu()`
  outputs,
- matrix norm `ord=2/-2` uses an internal singular-values-only op,
- CUDA solve consumes packed LU and I32 pivots directly, applying pivots on the
  device and using cuBLAS triangular solves,
- linalg AD emits prepared solve linearization and transpose rules,
- the runtime compiler now records extension output dtype/shape metadata per
  output slot, which supports mixed-output extensions such as packed LU plus
  I32 pivots.

The user-visible linalg wrappers remain unchanged. Internal prepared operations
are intentionally not exported as public wrapper APIs or hidden extension
payload constructors.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| `REPOSITORY_RULES.md` | Confirm public-surface, extension dispatch, AD, GPU, and work-log rules. | Kept prepared helpers internal and avoided hidden fallback paths. |
| Shared tensor4all rules | Confirm cross-project public API, layering, performance, and numerical rules. | Fixed the compiler/runtime abstraction instead of adding a local workaround. |
| `docs/design/gpu-backend-design.md` | Check CUDA ownership, launch, placement, and unsupported-op contracts. | Kept all pivot application and packed solve work on device. |
| `docs/design/tensor-prims.md` | Check current backend/linalg surface ownership. | Left public linalg wrappers unchanged and used the linalg extension backend surface. |
| `docs/spec/backend-contract.md` | Check compiled execution metadata contract. | Updated stale single-output dtype wording for mixed-dtype extension outputs. |
| `../jax/jax/_src/lax/linalg.py` | Compare solve, LU solve, SVD, QR, and triangular solve structure. | Mirrored JAX's split between factorization and prepared solve for solve sensitivities; added singular-values-only SVD for norm. |

## Reference code

JAX implements `solve` by computing `lu(stop_gradient(a))` once and passing
that factorization into `custom_linear_solve`, with both `solve` and
`transpose_solve` delegating to `lu_solve`. JAX's `lu_solve` applies the
permutation and then uses triangular solves over packed LU.

JAX SVD has a `compute_uv=False` mode for singular values only. tenferro did
not have a public `svd_values` wrapper in this change; it added only an
internal extension op so norm does not allocate U/VT when it only needs
singular values.

QR, Cholesky, Eigh, and triangular solve did not show the same "public
factorization materializes extra outputs before solve" problem. Their current
tenferro paths still have ordinary performance work to do, but this batch did
not find the same factor-reuse design flaw there.

## Decisions made

- **Use internal prepared ops rather than new public APIs.** `LuFactor`,
  `LuSolvePrepared`, and `SvdVals` are extension payloads used by wrappers and
  AD rules. They are not exposed as user-facing linalg functions, and their
  payload construction is crate-local rather than a `#[doc(hidden)]` public
  support surface.
- **Store extension metadata per output slot.** `LuFactor` returns
  `packed_lu`, I32 `pivots`, and `parity`; the compiler must not assume a
  single dtype for every output of one instruction.
- **Use bool flags at the backend trait boundary.** The public backend trait
  needs a hidden prepared-solve hook for runtime dispatch, but it does not need
  to expose a public `LuSolveMode` type.
- **Keep CUDA solve on the backend.** CUDA prepared solve applies pivots and
  runs triangular solves on device. It returns explicit errors for unsupported
  variants instead of falling back to CPU or rebuilding a backend.
- **Do not optimize CPU prepared solve in this batch.** CPU default prepared
  solve preserves correctness by delegating to `solve`; CUDA is the path where
  the observed regression was severe.

## Rejected or deferred alternatives

- **No public `lu_factor` / `lu_solve` wrappers in this PR.** They may be useful
  later, but this batch only needs internal factor reuse.
- **No squash of prepared ops into public `solve`.** AD needs to reference the
  factorized boundary explicitly, matching the traced linearization model.
- **No full CPU packed-LU solve optimization yet.** It is useful follow-up work
  but not required to remove the CUDA eager regression.
- **No FFT CUDA implementation in this PR.** FFT GPU support is tracked as
  follow-up work.

## Verification performed

- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `cargo test -p tenferro-runtime`
- `cargo test -p tenferro-linalg`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo test -p tenferro-linalg --features cuda`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda -- --ignored`

## Remaining risks

- CPU prepared LU factor reuse remains a performance follow-up.
- FFT CUDA execution remains unsupported and is tracked separately.
