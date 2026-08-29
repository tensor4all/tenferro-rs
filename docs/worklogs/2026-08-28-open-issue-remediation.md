# Open issue remediation

## Summary

Base: `origin/main` at `8e8d879d`.

This batch resolves verified documentation, invariant-marker, and public tensor
ownership defects from issues #1707, #1715, #1718, #1724, #1728, #1729, and
#1730. It does not implement the two design-gated performance investigations.

## Classification ledger

| Issue | Classification | Current evidence and disposition | Verification target |
| --- | --- | --- | --- |
| #1706 | Design Gate | The issue explicitly requires maintainer acceptance before implementation and predeclared cold-build experiments. Deferred unchanged. | Accepted design plus paired cold-build protocol |
| #1707 | Verify First | The pinned `strided-kernel` revision implements `i32`/`i64` erased sum/product with `wrapping_add`/`wrapping_mul`, and existing CPU overflow tests pass. Added adjacent tenferro invariants and a source-contract test rather than duplicating the optimized reduction traversal. | CPU overflow behavior and source-contract tests |
| #1715 | Auto Fix | Active agent-facing docs described syntax neutrally. They now name parse-time `Error::InvalidSubscripts` failures for omitted arrows and parenthesized ellipses. | Docs checks and mirror consistency |
| #1718 | Auto Fix | `flat_to_multi` retained a deliberate `dead_code` allowance without the required marker. Replaced its prose with the canonical adjacent `// INVARIANT:` marker. | Tensor tests and repository-rules review |
| #1720 | Design Gate | The proposed SVD policy depends on GPU-model measurements and an accepted selection policy. Deferred unchanged. | Accepted policy plus RTX/A100 benchmark matrix |
| #1724 | Auto Fix | The public raw CUDA API already supports PTX, CUBIN, NVRTC, scoped tensor borrowing, and launch. Added a downstream guide using only that surface and linked it from active GPU docs. | Docs checks and CUDA raw launch tests |
| #1728 | Auto Fix | JOSS publishes the faer paper as DOI `10.21105/joss.06099`. Added the backend-dependent citation requirement and full reference. | Link/docs checks |
| #1729 | Auto Fix | Two design docs and one neighboring active GPU guide still claimed host slice access panicked. Updated all three to the current `Error::RuntimeState` contract. | Stale-language scan and docs checks |
| #1730 | Auto Fix | `TypedTensor::into_parts` replaced backend extraction failure with an empty host vector. Made the canonical method fallible and added backend-storage regression coverage. | Focused tenferro-tensor test and doctest |

## Context reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, `CONTRIBUTING.md`
- shared tensor4all repository, Rust, performance, documentation, and provenance rules
- bug-fix and repository-remediation workflows
- `docs/design/gpu-extension-api.md` and the public `cuda::raw` implementation/tests
- einsum parser implementation and rejection tests
- JOSS faer article metadata and upstream `CITATION.cff`

## Decisions

- Kept integer reductions on the existing optimized `strided-kernel` plan. A
  second tenferro-owned scalar traversal would violate the CPU-kernel ownership
  contract and duplicate code; the exact dependency pin, runtime overflow
  tests, invariant markers, and source-contract test make the delegated
  wrapping contract explicit.
- Changed `TypedTensor::into_parts` directly to return `Result`; no compatibility
  shim was retained because the existing infallible behavior silently lost
  backend data.
- Reused the already hardware-exercised PTX/NVRTC public API rather than adding
  another CUDA runtime abstraction or dependency.

## Verification

Passed locally:

- `bash scripts/check-pr-fast.sh --coverage-reviewed --doc-snippets` with the
  focused tensor regression, CPU source-contract and overflow tests, and CUDA
  tutorial compile check;
- workspace plus standalone-extension formatting and clippy from the fast gate;
- the full local `docs` CI profile, including source-backed snippet checks,
  faer/BLAS interop examples, CUDA example compilation, rustdoc, and rendered
  site link/inventory validation;
- `cargo test -p tenferro-tensor --doc` (330 doctests);
- compile-only CUDA raw PTX/NVRTC tests and the source-backed custom-kernel
  tutorial binary; the docs CI profile now runs that compile check on GPU-less
  runners.

CUDA hardware execution was not run locally. Existing GPU tests cover PTX,
NVRTC, and CPU/CUDA integer-reduction parity when a CUDA device is available.

## Residual risks

- CUBIN execution remains architecture-specific and must be run with a matching
  `CUDA_ARCH`; the guide states this boundary.
- #1706 and #1720 remain open until their design and hardware gates are met.
