# API surface parity batch work log

Date: 2026-07-19

## Session summary

This batch resolves four accepted API-surface issues together: optional
reduction axes (#1279), eager linalg composites (#1265), concrete/read/typed
linalg extension traits (#1264), and eager FFT methods (#1266). The work keeps
one canonical operation vocabulary across concrete, eager, and traced layers
while preserving each layer's ownership, error, placement, and AD contracts.

The FFT work was intentionally delayed until the explicit backend-capability
refactor in #1424 passed review and CI. That PR was merged first, then its merge
commit was integrated into this branch before implementing the eager adapter.
Durable decisions are recorded in the batch design, linalg design, FFT backend
design, operation categories spec, and affected user guides.

## Classification ledger

| Issue | Classification | Working-hash evidence | Resolution |
| --- | --- | --- | --- |
| #1279 | Auto Fix after accepted contract | Eager and traced reduction surfaces used different axis representations and could not express all-axes versus empty-axes consistently. | `Option<&[usize]>`: `None` reduces all axes; `Some(&[])` is identity. All callers and docs migrated together. |
| #1265 | Auto Fix after accepted contract | Kernel eager linalg methods existed, while traced composites had no eager peers. | Added eager `slogdet`, `det`, `inv`, `eigvalsh`, `eigvals`, `pinv`, `pinv_with_rtol`, and `norm` from existing primitives. |
| #1264 | Auto Fix after accepted contract | Backend methods were the only direct dynamic surface and returned variable-length vectors for fixed-arity operations. | Added crate-root owned/read/typed extension traits, fixed tuples, sealed scalar mappings, and existing-placement read dispatch. |
| #1266 | Design Gate resolved by accepted contract and #1424 | Traced and concrete FFT APIs existed, but eager execution had no explicit `FftBackend` route. | After #1424 merged, added `EagerTensorFftExt` using the existing extension op/runtime and AD rules. |

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| Accepted issue comments for #1279, #1265, #1264, and #1266 | Freeze semantics before implementation. | Prevented compatibility shims and kept the batch limited to agreed surfaces. |
| `AGENTS.md`, `REPOSITORY_RULES.md`, and shared tensor4all Rust/docs rules | Confirm public API, validation, placement, docs, worklog, and merge requirements. | Added runnable examples, concrete error docs, shared validation helpers, and no-fallback tests. |
| Runtime eager/traced reduction implementations and callers | Trace the axis vocabulary across public APIs, AD, examples, and tutorials. | Migrated the canonical contract atomically instead of adding parallel overloads. |
| Linalg backend, traced, eager, and AD support modules | Identify existing kernels, composites, read hooks, and typed scalar contracts. | Reused backend primitives and AD-visible standard ops rather than adding kernels or transfers. |
| FFT extension op, runtime registration, backend capability, and cache design | Establish the post-#1424 execution boundary. | The eager adapter delegates only to the selected capability and shares validation with traced FFT. |
| Active linalg/FFT guides and design documents | Find stale entry-point and support claims. | Updated user selection tables, examples, typed-output rules, and placement behavior. |

CodeGraph was refreshed and used before direct source search for reduction,
linalg, eager-runtime, and FFT dependency tracing.

## Decisions made

- **Reduction absence and emptiness are distinct.** `None` means every axis;
  `Some(&[])` preserves the input. This matches the accepted contract across
  eager, traced, AD, docs, and examples without a legacy compatibility path.
- **Eager linalg composites remain compositions.** They reuse existing eager
  primitives so execution, error propagation, and AD recording stay visible to
  the established runtime instead of introducing hidden backend kernels.
- **Concrete linalg is receiver-first and backend-explicit.** Owned `Tensor`,
  borrowed `TensorRead`, and `TypedTensor<T>` use crate-root extension traits;
  decomposition arity is represented by tuples rather than output vectors.
- **Typed output dtype is part of the adapter contract.** `TensorScalar::Real`
  represents singular values, Hermitian eigenvalues, log-magnitudes, and norms;
  sealed `LinalgScalar::Complex` represents general eigen outputs. Backend
  dtype mismatches are reported rather than silently cast.
- **Borrowed linalg input stays borrowed until the provider boundary.** Read
  hooks are used directly. Complete-pivot solve composes read factorization,
  permutation matmul, and triangular read solves, including vector RHS reshape,
  without an adapter-level input clone or device transfer.
- **Eager FFT uses the explicit capability.** The adapter registers the same
  FFT runtime and applies the same `FftOp`. CPU delegates to its capability;
  other eager backend variants return `Unsupported` without upload, download,
  host reference, or a newly constructed CPU backend.
- **Known FFT validation is shared.** Eager and traced paths prepare the same
  validated op for dtype, rank, axis, transform length, and C2R spectrum shape.

## Rejected or deferred alternatives

- Legacy reduction overloads were rejected because they would preserve two
  meanings for an empty axis list and keep caller ambiguity alive.
- New eager linalg kernels or backend fallbacks were rejected; the requested
  operations already lower cleanly to supported primitives.
- Public typed linalg support for integer/bool tensors was rejected. The sealed
  scalar set is `f32`, `f64`, `Complex32`, and `Complex64`.
- Adapter-level cloning or implicit transfer for read linalg was rejected.
  Unsupported layout/placement remains a typed backend error.
- CUDA/cuFFT and real/complex FFT AD semantics remain out of scope. The eager
  surface exposes existing support only; unsupported rules remain explicit.

## Verification performed

Development followed a red/green cycle for each issue. Focused crate tests,
integration tests, doctests, clippy with `-D warnings`, guide snippet checks,
and the public error-doc audit were run while implementing each unit. The final
committed head is also checked through the repository PR gate and local
repository-rules review before publication.

- `cargo test -p tenferro-ad --features cpu-faer`
- `cargo test -p tenferro-linalg --features autodiff`
- `cargo test -p tenferro-fft --features autodiff`
- `cargo clippy -p tenferro-linalg --features autodiff --all-targets -- -D warnings`
- `cargo clippy -p tenferro-fft --features autodiff --all-targets -- -D warnings`
- `python3 scripts/check-public-error-docs.py --root-dir . --changed-from origin/main`
- `python3 scripts/check-doc-snippets.py --check`
- `python3 scripts/check-guide-dependency-snippets.py`
- `bash scripts/check-pr-fast.sh --coverage-reviewed` with the three focused
  crate tests above
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD`

## Remaining risks and follow-up

- The reduction signature migration is intentionally source-breaking; all
  in-repository callers are migrated, but downstream callers must wrap explicit
  axes in `Some(...)`.
- Typed linalg adapters validate provider output dtypes at runtime because the
  underlying backend contract remains dynamically typed.
- The concrete/read/typed linalg module is large because three public traits
  intentionally show their different ownership and output contracts directly;
  macro generation was avoided so rustdoc, feature behavior, and error
  boundaries remain reviewable.
- Eager GPU FFT remains unsupported until the selected GPU backend implements
  `FftBackend`; the adapter deliberately does not provide a fallback.
