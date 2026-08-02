# Issue 1589 CUDA wide SVD

## Session summary

Fixed the CUDA legacy `gesvd` fallback for wide matrices. cuSOLVER's legacy
driver requires `m >= n`, while tenferro previously passed large wide matrices
with their original dimensions.

## Context read

- issue #1589 and its linked cuSOLVER contract;
- `REPOSITORY_RULES.md` and `docs/design/gpu-backend-design.md`;
- the CUDA SVD factor, values-only, FFI, device-adjoint kernel, source-contract,
  and ignored hardware-test paths.

## Decisions made

- Keep the existing 1024 `gesvdj` selection threshold unchanged.
- Only for the legacy `gesvd` branch with `m < n`, factor `A^H` on CUDA.
- Reuse the existing one-pass real transpose / complex adjoint device kernel.
- For `A^H = U_B S V_B^H`, return `U_A = V_B` and `V_A^H = U_B^H` by
  adjointing the thin device factors. The values-only route needs only the
  adjointed input.
- Keep the original direct legacy route for tall matrices and preserve solver
  `info` checking and typed errors.

## Rejected alternatives

- Raising the Jacobi threshold would change numerical algorithm selection.
- Passing a wide matrix directly to legacy `gesvd` violates its shape contract.
- A host transpose or CPU SVD fallback would introduce a hidden device transfer.
- Composing separate transpose and conjugation allocations for complex inputs
  is unnecessary because the existing CUDA kernel performs the adjoint in one
  pass.

## Verification performed

- `cargo fmt --all --check`
- `cargo test -p tenferro-linalg`
- `cargo test -p tenferro-linalg --test gpu_linalg_source_contract`
- `cargo test -p tenferro-linalg --features cuda --test gpu_linalg --no-run`
- `git diff --check`

Portable source contracts verify orientation in both SVD routes and reject host
downloads. Ignored CUDA tests cover real and complex `8 x 1025` matrices with
reconstruction, thin-factor isometry, singular-value ordering, factor/value
parity, and a CPU values oracle. A `1025 x 8` test guards the direct tall route.

## Remaining risk

The CUDA tests compile locally but require a CUDA 12.8+ host for execution.
