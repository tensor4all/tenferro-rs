# Small Bug-Fix Remediation

## Session Summary

This batch triaged eight small repository-rule findings against the current
source, then grouped the confirmed repairs by owning subsystem. Seven findings
remain `Auto Fix` items implemented on `codex/batch-small-bugfixes`; one GPU
finding is a false positive whose host-side proof is now explicit and guarded;
and one additional issue was found stale because an earlier change on `main`
already removed the reported pattern. No GitHub issue was closed or otherwise
mutated during this session.

This is an in-progress ledger. The implementation commits are present, but the
full PR checklist, issue closure, PR creation, and merge are still pending.

## Context And Rules Read

- Online `tensor4all-agent-rules`: `rules/index.md`,
  `rules/common/repository.md`, `rules/common/docs-and-tests.md`, and
  `rules/rust/index.md`.
- Repository `AGENTS.md`, `CONTRIBUTING.md`, and `REPOSITORY_RULES.md`, with
  emphasis on public-boundary safety, checked allocation arithmetic, lock
  poison propagation, invariant markers, parallel backend semantics, work
  logs, and the pre-PR checklist.
- `ai/contribution-workflows/bugfix-pr.md` and
  `ai/contribution-workflows/repository-remediation.md`.
- The issue reports for #1325, #1351, #1290, #1350, #1349, #1330, #1352, and
  #1309; design-gated exclusions #1353, #1275, and #1276.
- `git log` and `git diff` for `origin/main..HEAD`, plus the current source and
  history relevant to #1309.

## Classification Ledger

| Issue | Classification | Current evidence | Close criterion and current result | Recurrence prevention |
| --- | --- | --- | --- | --- |
| #1325 | Auto Fix | Commits `0570bad9` and `3525c61e` make faer/LAPACK tensor assembly and scratch refill helpers fallible and propagate errors through their callers. | No production `.expect()` remains in the reported helpers; focused linalg tests and the full PR checklist must pass before closure. **Implemented; final verification pending.** | Unit tests exercise overflow helpers, and `cpu_linalg_source_contract.rs` guards the fallible helper boundary. |
| #1351 | Auto Fix | Commits `0570bad9` and `3525c61e` replace reported allocation products across faer and LAPACK assembly/workspace paths with checked element/byte calculations and typed errors. | Every reported `rows * cols` or equivalent allocation length is checked before allocation or FFI use. **Implemented; final verification pending.** | Helper unit tests cover overflow, while source-contract tests inventory the LAPACK allocation sites that are impractical to execute at extreme dimensions. |
| #1290 | Auto Fix | Commits `9ae171dd`, `2f103426`, and `c36c2c08` make single-output metadata registration and concatenate graph construction return `Result`, migrate workspace callers, and document fallible triangular traced methods. | Metadata registration failures reach callers without `.expect()` or a fallback value, with all affected callers and docs migrated. **Implemented; final verification pending.** | `runtime/tests/public_surface_contract.rs` requires fallible helpers and forbids `.expect()` in the reported registration bodies. |
| #1350 | Auto Fix | Commits `87c0979e`, `966ab814`, and `cd1a3b14` replace FFT cache poison recovery through `into_inner()` with an exact typed cache error and propagate it through FFT plan resolution. | A deliberately poisoned cache must produce the expected error through the public FFT path, and silent recovery must be absent. **Implemented; final verification pending.** | Concrete poison coverage plus public FFT source-contract/error-propagation tests require the exact failure path. |
| #1349 | Auto Fix | Commit `9f593ecd` routes `I64` DynamicTruncate sizes through integer clamping, keeping only floating dtypes on the finite/rounding path. | `2^53 + 1` remains exact on 64-bit hosts, negative values clamp to zero, and oversized values clamp to the axis extent. **Implemented; final verification pending.** | A focused unit test covers the value immediately above binary64 integer precision and the ordinary clamp boundaries. |
| #1330 | Auto Fix | Commits `7961fb8e`, `09b3b9d3`, `b1b834be`, and `2d2ef5ec` remove the CPU sign-only rejection, harden signed position calculations before narrowing, and align CPU/CUDA cropping behavior. | Valid negative edge padding crops on CPU and CUDA while invalid dimensions and index conversions remain checked. **Implemented; CUDA hardware verification pending.** | CPU value tests cover low/high cropping and overflow edges; GPU source contracts and ignored CubeCL value tests cover signed mapping and parity. |
| #1352 | False Positive | The kernel-local `update_window_len` product is bounded by host launch validation. Commits `81bcf014`, `5fb5e5bb`, `579a138d`, and `8c003f5e` add concrete `// INVARIANT:` markers and lexically prove every scatter launch reaches the checked window/batch calculation without proof aliases. | Each launch path must preserve a checked host product before the kernel's unchecked multiplication, and the marker must remain adjacent and re-verifiable. **Evidence recorded and guarded; final suite pending.** | `cubecl_launch_contract.rs` inventories launch sites, rejects unchecked or aliased proofs, and ties the kernel marker to host validation. |
| #1309 | Stale / Out Of Scope | This was already resolved on `main` by `9ad0fb1a` (`Enforce AD graph emission invariants (#1324)`). Current `one_like_fixed` at `crates/tenferro-linalg/src/ad/rules/support.rs:306` calls `ad_support::one_like`; the former `scalar_one_fixed` no longer exists, and `identity_matrix_fixed` calls `ad_support::identity_matrix`. | Neither reported helper emits `StdTensorOp::Exp` for a constant. Tests at `support.rs:829` and `support.rs:844` assert `Constant`/`BroadcastInDim` structure and reject analytic constant shortcuts. **Close as already fixed by #1324 after maintainer review; no batch code change needed.** | Shared semantic constant builders centralize dtype-aware emission, and graph-structure tests reject `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Expm1`, or `Log1p` as constant shortcuts. |

## Issue #1309 Evidence

The current source is stronger than the issue's proposed local visibility
change. `tenferro-internal-ops` exposes semantic builders through its supported
AD helper surface, and linalg delegates to that surface instead of reaching
into `zeros.rs`:

- `one_like_fixed` delegates to `ad_support::one_like(builder, dtype, anchor,
  anchor_rank)`.
- `scalar_one_fixed` and its `exp(anchor - anchor)` implementation were
  removed.
- `identity_matrix_fixed` delegates to `ad_support::identity_matrix`, which
  supplies the scalar one through semantic constant construction.
- `one_like_fixed_uses_semantic_constant_not_analytic_shortcut` expects exactly
  one `Constant` and one `BroadcastInDim` operation.
- `identity_matrix_fixed_uses_semantic_constant_not_analytic_shortcut` uses the
  same analytic-shortcut rejection helper.

This supports closing #1309 as stale/already fixed, citing commit
`9ad0fb1a96112bd1e0ee33a3c581198745923635` and its regression tests. This work
log does not perform that GitHub mutation.

## Scope And Design Exclusions

- #1353 is design-gated: choosing IEEE floating remainder semantics versus a
  new typed zero-divisor policy requires a binding cross-dtype specification.
- #1275 is design-gated: removing public CPU free functions or introducing a
  shared default backend changes public API and runtime/thread-pool ownership.
- #1276 is design-gated: renaming typed einsum view methods and adding a typed
  mutable-write abstraction changes the public API surface.

The batch does not change those policies or APIs. It also does not introduce a
new backend, operation family, dependency, feature flag, AD convention, or
coverage policy.

## Verification Status

The implementation commits contain focused unit, source-contract, and ignored
CUDA regression coverage described in the ledger. During work-log preparation,
the #1309 source and commit history were inspected directly, and the following
narrow checks were run:

- `cargo test -p tenferro-linalg --features autodiff
  one_like_fixed_uses_semantic_constant_not_analytic_shortcut --lib`: passed,
  one test.
- `git diff --check`: passed.
- `python3 scripts/check-docs-site.py`: snippet and guide dependency checks
  passed, but the overall command stopped because workspace rustdoc output had
  not yet been generated. It must be rerun after `cargo doc --workspace
  --no-deps` during final verification.

The following repository-required checks are **pending** and must not be
inferred from this ledger: release workspace tests, CI-parity clippy, LLVM
coverage plus the per-file threshold check, workspace rustdoc, the committed
repository-rules review, and CUDA ignored tests on supported NVIDIA hardware.
Issue closure, PR creation, branch-protection confirmation, non-squash
auto-merge configuration, and merge are also pending.

## Environment Notes

The linalg changes cover faer and LAPACK allocation/error paths. Full release
tests may depend on the host BLAS/LAPACK linker configuration; no result is
claimed here until that command is run. Coverage generation can require
substantial target-directory and JSON disk space, so available disk should be
checked before the final coverage run. CUDA value tests require the documented
CUDA toolkit and runtime libraries; source-contract coverage does not replace
that hardware run.

## Residual Risks

- The broad fallible traced-API migration touched runtime, AD, einsum, XLA,
  tutorials, and tests; only the full release workspace test can establish
  cross-crate completeness.
- GPU signed-padding tests are ignored without CUDA, so runtime parity remains
  hardware-gated even though CPU tests and source contracts cover the mapping.
- #1352 remains correct only while every scatter launch preserves the checked
  host proof; the lexical contract tests intentionally fail when a new launch
  path or proof alias is introduced.
- Extreme allocation failures are validated mainly through checked helpers and
  source contracts because constructing near-`usize::MAX` matrices is not a
  practical runtime test.
- The three excluded issues remain open design questions and must not be
  silently folded into this remediation PR.
