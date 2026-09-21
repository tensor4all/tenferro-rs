# `EighOptions::driver` for cuSOLVER eigensolver selection (issue #1846)

## Decisions

- `EighDriver { Auto, Syevd, Syevj }` mirrors `SvdDriver` from #1839/#1845:
  a field on `EighOptions` with a `.driver(..)` builder, hashed into the
  `LinalgOp::Eigh` / `EighVals` payload, preserved across the
  `Eigh -> EighVals` pruning, and reached through defaulted `LinalgBackend`
  hooks (`eigh_with_options_read`, `eigh_values_with_driver`,
  `eigh_values_with_driver_read`) so no existing backend implementation
  breaks. CPU providers ignore it.
- `Auto` maps to `syevd`/`heevd` at every size and batch count, which is
  exactly the pre-driver behavior. Unlike the SVD policy there is no size
  threshold, because no measurement justifies one.
  - Rejected: giving `Auto` a size rule that picks Jacobi for small batches.
    It would change the default for existing callers, and the issue's
    non-goals rule it out. A caller who wants the batched routine asks for it.
- Inside `EighDriver::Syevj` the backend selects `syevjBatched` whenever
  cuSOLVER accepts it (`batch_total > 1` and `n <= 32`), otherwise per-matrix
  `syevj`. This is a routine choice inside one driver, not a fourth public
  variant: callers should not have to know the cuSOLVER dimension limit.
- The batched entry point is the reason this API exists. cuSOLVER ships no
  divide-and-conquer batched eigensolver, so a single-launch batch solve is
  unreachable from any default policy.

## Verification conclusions and constraints

- The driver policy is unit-tested without a GPU: `Auto` and `Syevd` map to
  `syevd` at every `(n, batch)` probed, `Syevj` takes the batched routine only
  for a real batch within the dimension limit, and falls back to per-matrix
  Jacobi for single matrices and above the limit.
- CPU: `cpu_ignores_explicit_eigh_driver_on_concrete_and_traced_paths` checks
  that a forced driver executes and matches the default through the owned,
  borrowed, values-only, and traced-pruned paths. Following the #1845 finding,
  the bit-for-bit comparison stays inside one entry point and the
  vectors-vs-values comparison uses a tolerance; the test passes under both
  `cpu-faer` and `cpu-blas`.
- GPU: the `#[ignore]` CUDA tests force each driver at `n = 8`, `n = 64`, and
  a `n = 6, batch = 4` case that routes through `syevjBatched`, checking
  `V diag(w) V^T` reconstruction, ascending eigenvalues, agreement across the
  owned/borrowed/values-only entry points, and agreement with the CPU
  provider. They pass on an NVIDIA A100 80GB PCIe (driver 580.126.09, CUDA
  12.6) together with the pre-existing CUDA eigh tests.
- Measured on that A100 through the public API (median of 5, after one
  warm-up), with the Jacobi spectrum checked against the `syevd` spectrum on
  the same input:

  | case | `Syevd` (= `Auto`) | `Syevj` | ratio | agreement |
  |---|---|---|---|---|
  | batched f64 `n=8`, batch 1024 | 105.18 ms | 0.29 ms | 363x faster | 3.4e-15 |
  | batched f64 `n=32`, batch 256 | 69.66 ms | 0.44 ms | 158x faster | 9.0e-15 |
  | dense f64 `n=512` | 6.29 ms | 10.14 ms | 1.6x slower | 1.4e-13 |
  | dense f64 `n=1024` | 14.37 ms | 30.91 ms | 2.2x slower | 3.1e-13 |
  | ten-decade c64 `n=256` | 3.62 ms | 45.90 ms | 13x slower | 2.4e-13 |
  | ten-decade c64 `n=512` | 7.60 ms | 157.12 ms | 21x slower | 8.3e-13 |

  An earlier revision of this branch had only per-matrix `syevj`, and under
  that implementation Jacobi lost every case, including the batched ones
  (290.99 ms against 104.77 ms at `n=8`, batch 1024). The batched rows above
  are therefore a property of `syevjBatched`, not of Jacobi as such: at
  `n = 8` the divide-and-conquer path spends about 102 us per matrix, which is
  launch overhead, not arithmetic.
- Constraints: `tenferro-linalg` compiles GPU linear algebra only under the
  `cuda` feature. `syevjBatched` is capped at `n = 32` by cuSOLVER. The Jacobi
  parameter object keeps cuSOLVER's default tolerance and sweep limit; the
  driver selects a routine and does not retune convergence. The measurement
  host had CUDA 12.6, not the 12.8 that enables the full CubeCL feature set.
