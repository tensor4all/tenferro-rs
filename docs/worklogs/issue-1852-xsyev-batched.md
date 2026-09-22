# Batched eigh through `cusolverDnXsyevBatched` (issue #1852)

Corrects two claims this repository shipped in #1846 / #1850 and makes the
batched eigh default fast without a driver knob. The finding, the manual
citations, and the standalone cuSOLVER measurements are
[@exAClior's](https://github.com/exAClior) in
[issue #1852](https://github.com/tensor4all/tenferro-rs/issues/1852).

## What was wrong

- **`n <= 32` was not a cuSOLVER limit.** #1850 described
  `CUSOLVER_SYEVJ_BATCHED_MAX_DIM` as the order beyond which cuSOLVER rejects
  the batched Jacobi entry point, in the constant's doc comment, the FFI
  wrapper, the design doc, and the #1846 work log. The cuSOLVER manual
  documents a 32 limit for `gesvdjBatched`, the *SVD* Jacobi routine; the
  `syevjBatched` entry has no such bound. #1846 flagged this as needing
  confirmation and the implementation then asserted it as fact without
  checking. Verified here by raising the guard and running batched Jacobi at
  order 64 and 128 on an A100: both succeed, with reconstruction, eigenvalue
  ordering and CPU agreement holding.
- **"cuSOLVER has no divide-and-conquer batched counterpart" was false.**
  `cusolverDnXsyevBatched` exists in the 64-bit API from cuSOLVER 11.7.1.
  Verified with `nm -D` against `libcusolver.so.11.7.1.2` from CUDA 12.6 —
  the same library the #1850 measurements were taken on. This mattered
  because that claim was the central justification for adding `EighDriver`
  at all.

## Decisions

- Batched input always takes a batched routine; the order no longer enters the
  decision, so `CUSOLVER_SYEVJ_BATCHED_MAX_DIM` is deleted rather than raised.
  The driver alone selects which batched routine: `Auto`/`Syevd` take
  `XsyevBatched`, `Syevj` takes `syevjBatched`. A single matrix has no launch
  overhead to amortize and keeps the per-matrix entry points.
  - Rejected: raising the threshold to 128, the issue's option (2). Each
    batched routine beats its own per-matrix loop at every order measured, so
    any threshold would only pick a slower path.
- `Auto` is restated as *the fastest divide-and-conquer routine available*
  rather than *bit-identical to the pre-driver behaviour*. Batched results can
  now differ from the per-matrix loop in the last ULPs. This is a deliberate
  change to the #1850 contract, whose non-goals said the default would not
  change; the 100x-280x gain at equal accuracy justifies revisiting it.
- The supported CUDA floor moves from 12.4 to 12.6.2 (cuSOLVER >= 11.7.1),
  with `XsyevBatched` loaded as a mandatory symbol.
  - Rejected: an optional symbol with a runtime fallback. The cuSOLVER loader
    treats every symbol as mandatory, so this would need a second loading
    mode and a second batched code path to keep tested.
  - Rejected: keeping the 12.4 floor with a mandatory symbol. A CUDA 12.4 user
    would then fail to load cuSOLVER at all, so every linalg operation would
    break rather than one batched path being slower.
  - The floor is a *library* requirement. CUDA 12.x supports the same GPU
    architectures throughout and tenferro `dlopen`s cuSOLVER, so a cuSOLVER
    package update satisfies it without touching the driver or the GPU. GPU CI
    history supports the move: across 28 sampled on-pod runs the 12.4 baseline
    tier was selected 0 times, with every host reporting driver CUDA API 12.8
    or 13.0.

## Verification conclusions and constraints

- The policy is unit-tested without a GPU: every driver sends a real batch to
  a batched routine at every order probed, and single matrices keep the
  per-matrix routines. A source-contract test asserts
  `CUSOLVER_SYEVJ_BATCHED_MAX_DIM` is not reintroduced.
- On an A100 80GB PCIe (driver 580.126.09, CUDA 12.6), the `#[ignore]` CUDA
  eigh tests pass, including new cases at order 33 and 64 for both drivers —
  orders the old guard excluded — and a check that the batched `Auto` spectrum
  matches the CPU provider within 1e-9.
- Measured through the public API on that A100 (median of 5 after one warm-up),
  against the #1850 numbers from the same machine:

  | case | `Auto` before (#1850) | `Auto` now | `Syevj` now |
  |---|---|---|---|
  | f64 order 8, batch 1024 | 105.18 ms | **0.37 ms** | 0.27 ms |
  | f64 order 32, batch 256 | 69.66 ms | **0.69 ms** | 0.38 ms |
  | f64 order 64, batch 256 | not measured | **1.78 ms** | 1.72 ms |
  | f64 order 128, batch 128 | not measured | **3.53 ms** | 15.70 ms |

  So the default gained about 100x-280x on batches, and Jacobi's remaining
  advantage is confined to roughly order 32 and below, where it is a fraction
  of a millisecond. Single-matrix and wide-spectrum behaviour is unchanged.
- Constraint: `XsyevBatched` is a 64-bit-API routine reusing the params plus
  host/device workspace pattern that `Xgesvdp` introduced in #1851. The host
  workspace is left uninitialized because cuSOLVER owns it, and it outlives
  the launch because the solver-status download in the same scope is a host
  barrier.
