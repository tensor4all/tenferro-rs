# Issue #1267: borrowed solve entry points

## Session summary

Added explicit `TensorRead` entry points for general and triangular linear solves. The backend trait defaults remain capability-honest by returning `Unsupported`; the CPU backend accepts owned and arbitrary valid host reads, canonicalizes them within host placement when required, and delegates to the existing provider-owned solve implementations.

## Sources reviewed

- GitHub issue #1267 and the merged `TensorRead` API decision from #1420.
- `AGENTS.md`, `CONTRIBUTING.md`, `REPOSITORY_RULES.md`, and the shared tensor4all Rust/repository rules.
- `LinalgBackend`, `CpuBackend`, existing unary `_read` hooks, Faer linalg adapters, and LAPACK solve/triangular-solve adapters.
- Existing read-layout, typed-error, singular-system, and backend-buffer tests.

## Decisions

- Added `solve_read` and `triangular_solve_read` without changing the owned methods.
- Kept default trait implementations as explicit `Unsupported` errors; materialization policy belongs to each backend.
- Validated dtype and CPU placement before canonicalization so zero-sized inputs and backend buffers cannot bypass the public boundary.
- Borrowed `TensorRead::Tensor` operands directly without copying. For `TensorRead::View`, used `CpuBackend`'s existing pooled `to_contiguous` path and then the owned solve methods. This keeps shape validation, vector-RHS restoration, provider dispatch, singular detection, and output placement single-sourced.
- Did not add CUDA overrides or any implicit CPU/GPU transfer.

## Alternatives considered

- Trait-level fallback to owned solves was rejected because the trait has no backend-neutral placement or materialization policy.
- `TensorView`-only signatures were rejected because `TensorRead` is the repository-wide borrowed-input contract.
- New Faer binary `MatRef` solve cores were deferred. Unary decompositions already have view cores, but binary solve implementations currently combine compact RHS copying, factorization, batching, and provider dispatch; separating those kernels is an independent performance refactor.

## Verification

- `cargo fmt --all -- --check`.
- `cargo clippy -p tenferro-linalg --all-targets --features autodiff -- -D warnings`.
- `python3 scripts/check-public-error-docs.py`.
- `cargo test -p tenferro-linalg --features autodiff`: 114 unit and 148 integration tests passed.
- `cargo test -p tenferro-linalg --doc --features autodiff`: 75 doctests passed.
- `RUSTFLAGS='-l dylib=openblas -l dylib=lapack' cargo test -p tenferro-linalg --no-default-features --features cpu-blas --lib solve_read_accepts_owned_and_strided_inputs_for_compiled_providers`: both BLAS borrowed-solve tests passed.

## Remaining risk

Borrowed CPU solves currently canonicalize explicit view operands even when a future provider adapter could consume one or both views directly. Owned `TensorRead::Tensor` operands are not copied. The behavior and placement semantics are correct, but a later benchmark-backed provider refactor may remove the remaining view copies without changing the public API.
