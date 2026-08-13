# 2026-08-13 — single-call eager solve (`LinalgOp::Solve`)

## Session summary

Issue #1674: the eager `solve` ran two `apply_eager` calls (`LuFactor` +
`LuSolvePrepared`), paying the fixed per-call extension machinery twice
(~64 µs for a 2×2, measured pinned/release in issue #1672). Added a
partial-pivot `LinalgOp::Solve` (2 in → 1 out) and switched `eager_ext.rs::solve`
to a single `apply_linalg_eager` call. Result: **~64 → ~27.7 µs (−57%)**, no
regression in matmul/eigh/svd, eager AD forward/backward intact (JVP/VJP
verified against finite differences incl. complex adjoint).

## Context read

- `crates/tenferro-linalg/src/eager_ext.rs::solve` (composite),
  `extension.rs` (`LinalgOp` vocabulary + `execute_linalg` dispatch +
  `linalg_session_supported`), `ad/rules/solve.rs` (shared `LinearSolveOp`
  machinery), `ad.rs` / `ad/semantic.rs` (traced + eager AD registration).
- Cost structure from #1672: single apply_eager floor ~25-29 µs (session open
  ~3 µs + machinery + materialization + wrap); `execute_linalg` itself only
  ~10 µs of the 64 µs.

## Chosen design

- **New `LinalgOp::Solve` (partial pivoting)** instead of reusing the
  existing single-op `FullPivLuSolve`: full pivoting is slower and
  parallelizes poorly, and would silently change the documented partial-pivot
  numerics of `solve`, diverging from the concrete `LinalgBackend::solve`.
- Dispatch to the existing `backend.solve` (faer partial-pivot LU on CPU,
  cuSOLVER getrf + prepared pivot/triangular solves on CUDA in-session).
- AD: `LinearSolveOp::Solve` reuses `linearize_linear_solve` /
  `transpose_linear_solve`; the transpose forms `A^H` explicitly
  (`StdTensorOp::Transpose` on the conjugated matrix) since `Solve` has no
  transpose flag, then a plain partial-pivot solve. Registered in traced
  `ad.rs` and eager `semantic.rs` (`semantic_custom_transpose`).
- Traced `solve_in_graph` composite intentionally unchanged (no per-call
  overhead in the graph path).

## Rejected alternatives

- Reusing `FullPivLuSolve`: full pivoting (see above).
- Keeping the composite: pays the second session + machinery round for every
  eager solve.

## Residual risks

- CUDA eager `solve` now runs in-session via `LinalgOp::Solve` (was the
  LuFactor/LuSolvePrepared composite, also in-session); verified admitted and
  never returns `Unsupported` for F32/F64/C32/C64. GPU runtime validation of
  the new path is covered by the existing `cargo check --features cuda` and
  the gated GPU tests; actual GPU run requires a CUDA host.
- The eager AD solve tests were initially diag-only (could not detect an
  omitted transpose); strengthened with nonsymmetric + finite-difference +
  C64 adjoint coverage per review.
