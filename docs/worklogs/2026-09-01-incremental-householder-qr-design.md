# Incremental Householder QR design (#1735)

## Session summary

Created the durable design for an opaque, backend-neutral incremental
Householder QR state. The design fixes the public state, append algorithm,
`from_factors` semantics, output gauge, primal rank-deficiency behavior,
semantic AD tangent, CPU/CUDA ownership, tests, performance gate, and phased
delivery.

## Context reviewed

- GitHub issue #1735 and its accepted phase plan
- `AGENTS.md`, `REPOSITORY_RULES.md`, and relevant shared Rust/numerical rules
- `docs/design/linalg.md`, `linalg-prims.md`, and `gpu-backend-design.md`
- `docs/spec/ad-contract.md`
- existing QR public, backend, CPU-faer, LAPACK, CUDA, gauge, and AD-manifest
  implementations
- faer elementary and blocked Householder APIs

## Decisions

- Keep provider-neutral scalar reflector coefficients and private packed state.
- Support matrix and compatible-factor construction in v1.
- Treat the opaque state's tangent as the accumulated matrix tangent; the
  coefficient tensor is auxiliary.
- Implement CPU-faer state construction with elementary Householders rather
  than converting faer's blocked coefficient matrix.
- Use cuSOLVER `geqrf` plus streamed cuBLAS `gemv` and `ger`/`geru` for CUDA
  reflector application because cuSOLVER has no `ormqr`/`unmqr`.
- Make positive-diagonal gauge provider-owned on CUDA and share the device-side
  implementation with existing CUDA QR.
- Deliver CPU primal, AD after oracle coverage, CUDA, and performance evidence
  as independently reviewed phases.

## Alternatives rejected or deferred

- Explicit-Q state and full-QR append: violate the compact-state and scaling
  requirements.
- Provider handles in public state: not portable across eager/traced execution.
- Host gauge or CPU fallback for CUDA: violates placement policy.
- Batched, pivoted, arbitrary-index Q extraction, and WebGPU/ROCm support:
  deferred from v1.

## Design review gate

Reviewer selection was made by the user: `reviewer-flash` (DeepSeek V4 Flash),
read-only, high thinking.

- Round 1: incomplete; 300-second timeout, no verdict.
- Round 2: **Findings-require-fix**. Important findings: cuSOLVER has no
  `ormqr`/`unmqr`; existing positive-diagonal gauge is host-only. Minor
  findings covered faer scalar coefficients, AD gauge consistency, and
  symbolic append split width.
- Round 3: **Correct-to-merge**. All Important findings and requested Minor
  clarifications were closed. Two non-blocking phase-4 wording/enforcement notes
  remain for the CUDA post-diff review.

No Rust implementation started before the Round-3 verdict.

### Phase 3 amendment review

Before AD implementation, `reviewer-flash` reviewed the Phase-3 amendment for
symbolic thin-Q recovery, fixed `Primary` residual roles, exact oracle families,
and metadata invariants. The bounded closure review returned
**Correct-to-merge** with no Critical or Important findings. Its Minor clarity
notes were folded into the durable design before implementation.

The oracle-first prerequisite merged separately as
`tensor4all/tensor-ad-oracles#25` (`8a4f95cd`). Its exact head passed the
`replay` and `regenerate` required checks; the oracle family covers all five
operations and all four scalar dtypes.

### Phase 4 amendment review

Before CUDA implementation, `reviewer-flash` identified missing concrete
cuBLAS GEMV/GER/GERU/GEMM bindings, an unspecified device-native
`from_factors` fold, and owned/read/typed QR gauge routing as Important design
gaps. The amendment names each FFI and backend seam, the packed assembly,
metadata-only solver-info downloads, the shared device gauge, and the
`qr_with_options_read` hook. Closure review returned **Correct-to-merge** with
no remaining Critical or Important findings.

## Verification

```text
git diff --check
bash scripts/check-pr-fast.sh
```

The docs-only fast gate passed, including `scripts/check-doc-snippets.py`.

## Remaining risks

- Phase-2 CPU code must preserve fixed reflector positions and backend session
  allocation/threading contracts.
- Phase-3 AD requires new oracle families before support is enabled.
- Phase-4 CUDA performance may require blocked WY application if the simple
  streamed reflector sequence misses the predeclared gate.
