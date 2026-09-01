# Incremental Householder QR Phase 4: CUDA (#1735)

## Session summary

Implemented device-native compact Householder QR for CUDA across factorization,
factor import, column append, R extraction, selected Q columns, and positive-
diagonal gauge. The implementation uses cuSOLVER GEQRF, cuBLAS
GEMV/GEMM/GER/GERU, and linalg-owned CubeCL kernels without host tensor payload
transfer or full-QR append fallback.

## Context reviewed

- `REPOSITORY_RULES.md` GPU, transfer, FFI, materialization, and validation rules
- `docs/design/incremental-householder-qr.md` and `gpu-backend-design.md`
- CUDA raw/CubeCL session ownership and stream-retention contracts
- existing CUDA QR, cuSOLVER/cuBLAS vtables, solver-info handling, and linalg
  hardware/source-contract tests
- Phase-2 compact CPU implementations and Phase-3 abstract-state AD

## Design review gate

The user-selected `reviewer-flash` review found three Important gaps in the
initial CUDA prose: missing cuBLAS GEMV/GER/GERU/GEMM bindings, unspecified
CUDA factor-import folding, and owned/read/typed QR gauge routing. The amended
design names the FFI/backend seams, device packed assembly, metadata-only
status downloads, shared gauge kernels, and `qr_with_options_read` hook.
Closure review returned **Correct-to-merge** before CUDA implementation.

## Decisions

- Factorization stores cuSOLVER GEQRF packed output and device tau directly.
- Reflector application builds one implicit-v vector per reflector, computes
  real contractions with GEMV and complex contractions with GEMM, folds tau
  into that contraction, and applies GER/GERU with device pointer mode.
- Append applies old Q-adjoint to the new block, factors only the trailing
  residual, and concatenates packed columns and coefficients on device.
- Factor import factors Q, folds its triangular factor into R with cuBLAS GEMM,
  and assembles packed state with a CubeCL kernel; it never QR-factorizes Q*R.
- Positive-diagonal phases are materialized into a device phase vector before
  separate Q/R scaling, preventing cross-thread diagonal read/write races.
- The only host downloads are `i32` solver/validation status values. Input
  factors and tensor outputs remain device-local.
- GPU Householder code lives in `gpu/linalg/householder_qr.rs`; backend routing,
  vendor FFI, and kernels stay in their existing ownership seams.

## Verification completed so far

- CUDA+autodiff feature compilation
- local NVIDIA A100 hardware reconstruction for F32/F64/C32/C64
- multiple append and square-to-wide transition
- real and complex factor import
- rank-deficient factorization, zero-column append, selected Q range, and
  wrong-placement rejection
- existing CUDA QR positive-diagonal owned/read path parity
- source contracts for no full refactorization, no host payload download,
  required cuBLAS symbols, public QR routing, and unsafe-block comments

## Post-diff review

The user-selected `reviewer-flash` review was split into bounded lanes after a
broad attempt timed out:

- Householder CUDA math/FFI-use lane found one Important pointer-mode leak on
  setup errors. Device pointer mode is now entered only after every fallible
  setup step and always reset before propagating computation errors. Closure
  review returned **Correct-to-merge**.
- FFI/kernel/routing lane found one Important batched-gauge bug: phases were
  initially shared from batch zero. Phase tensors now have shape
  `[k, batch...]`, and kernels map batch axes explicitly for arbitrary rank.
  A rank-3 complex and rank-4 real hardware test pass. Closure review returned
  **Correct-to-merge** with no remaining Critical or Important findings.

Minor review notes were also closed: pointer-mode docs describe host/device
scalar pointers, source contracts pin every new cuBLAS symbol, and tests add
empty Q ranges, wide starts, explicit orthogonality, rank-deficient factor
import, complex nonzero Q ranges, and multi-axis real batches.

## Candidate verification

- CUDA+autodiff feature compilation passed.
- Local NVIDIA A100 hardware batch: six compact Householder tests passed for
  F32/F64/C32/C64, multiple append and wide transitions, factor import,
  rank-deficient inputs, zero-column append, empty/nonzero selected-Q ranges,
  orthogonality, gauge, and placement errors.
- Existing CUDA QR positive-diagonal owned/read tests passed for unbatched C64,
  rank-3 batched C64, and rank-4 batched F64.
- All 168 CPU-faer linalg library tests and 228 integration tests passed.
- Source contracts for no full refactorization/host payload transfer, every
  required cuBLAS symbol, public QR routing, CUDA admission, and unsafe comments
  passed.

## Candidate gates

- `python3 scripts/ci/run_profile.py fmt` and documentation snippet checks
  passed.
- Focused default-feature clippy and all 27 GPU source-contract tests passed.
- CPU-BLAS-only, CPU-faer+BLAS, and CUDA+autodiff feature checks passed.
- `scripts/check-pr-fast.sh --coverage-reviewed` passed with the compact CUDA
  source contract as its focused test, including workspace/standalone clippy,
  formatting, and docs checks.
- Worktree deterministic repository-rules review passed; the external LLM lane
  was skipped with reason `local deterministic review`.

## Remaining gates

Committed-head repository-rules review, exact-commit checks, and hosted CI
remain pending. Phase-5 performance gates remain separate.
