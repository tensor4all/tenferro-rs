# Small Bug-Fix Remediation

## Session Summary

This batch triaged eight small repository-rule findings against the current
source, then grouped the confirmed repairs by owning subsystem. Seven findings
remain `Auto Fix` items implemented on `codex/batch-small-bugfixes`; one GPU
finding is a false positive whose host-side proof is now explicit and guarded;
and one additional issue was found stale because an earlier change on `main`
already removed the reported pattern. Issue #1309 was closed after that source
and test evidence was verified.

The implementation and required verification are complete. PR creation, merge,
and closure of #1325, #1351, #1290, #1350, #1349, #1330, and #1352 remain
pending.

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
| #1325 | Auto Fix | Commits `0570bad9` and `3525c61e` make faer/LAPACK tensor assembly and scratch refill helpers fallible and propagate errors through their callers. | No production `.expect()` remains in the reported helpers, and the full verification suite passes. **Ready to close when the remediation PR merges.** | Unit tests exercise overflow helpers, and `cpu_linalg_source_contract.rs` guards the fallible helper boundary. |
| #1351 | Auto Fix | Commits `0570bad9` and `3525c61e` replace reported allocation products across faer and LAPACK assembly/workspace paths with checked element/byte calculations and typed errors. | Every reported `rows * cols` or equivalent allocation length is checked before allocation or FFI use, and the full verification suite passes. **Ready to close when the remediation PR merges.** | Helper unit tests cover overflow, while source-contract tests inventory the LAPACK allocation sites that are impractical to execute at extreme dimensions. |
| #1290 | Auto Fix | Commits `9ae171dd`, `2f103426`, and `c36c2c08` make single-output metadata registration and concatenate graph construction return `Result`, migrate workspace callers, and document fallible triangular traced methods. | Metadata registration failures reach callers without `.expect()` or a fallback value; affected callers, runtime/XLA doctests, and docs are migrated. **Ready to close when the remediation PR merges.** | `runtime/tests/public_surface_contract.rs` requires fallible helpers and forbids `.expect()` in the reported registration bodies. |
| #1350 | Auto Fix | Commits `87c0979e`, `966ab814`, and `cd1a3b14` replace FFT cache poison recovery through `into_inner()` with an exact typed cache error and propagate it through FFT plan resolution. | A deliberately poisoned cache produces the expected error through the public FFT path, silent recovery is absent, and the full verification suite passes. **Ready to close when the remediation PR merges.** | Concrete poison coverage plus public FFT source-contract/error-propagation tests require the exact failure path. |
| #1349 | Auto Fix | Commit `9f593ecd` routes `I64` DynamicTruncate sizes through integer clamping, keeping only floating dtypes on the finite/rounding path. | `2^53 + 1` remains exact on 64-bit hosts, negative values clamp to zero, oversized values clamp to the axis extent, and the full verification suite passes. **Ready to close when the remediation PR merges.** | A focused unit test covers the value immediately above binary64 integer precision and the ordinary clamp boundaries. |
| #1330 | Auto Fix | Commits `7961fb8e`, `09b3b9d3`, `b1b834be`, and `2d2ef5ec` remove the CPU sign-only rejection, harden signed position calculations before narrowing, and align CPU/CUDA cropping behavior. | Valid negative edge padding crops on CPU and CUDA while invalid dimensions and index conversions remain checked; the A100 ignored test passed after RED/GREEN mutation. **Ready to close when the remediation PR merges.** | CPU value tests cover low/high cropping and overflow edges; GPU source contracts and ignored CubeCL value tests cover signed mapping and parity. |
| #1352 | False Positive | The kernel-local `update_window_len` product is bounded by host launch validation. Commits `81bcf014`, `5fb5e5bb`, `579a138d`, and `8c003f5e` add concrete `// INVARIANT:` markers and lexically prove every scatter launch reaches the checked window/batch calculation without proof aliases. | Each launch path preserves a checked host product before the kernel's unchecked multiplication; CUDA checks and the full verification suite pass. **Ready to close with the remediation PR as a guarded false positive.** | `cubecl_launch_contract.rs` inventories launch sites, rejects unchecked or aliased proofs, and ties the kernel marker to host validation. |
| #1309 | Stale / Out Of Scope | This was already resolved on `main` by `9ad0fb1a` (`Enforce AD graph emission invariants (#1324)`). Current `one_like_fixed` at `crates/tenferro-linalg/src/ad/rules/support.rs:306` calls `ad_support::one_like`; the former `scalar_one_fixed` no longer exists, and `identity_matrix_fixed` calls `ad_support::identity_matrix`. | Neither reported helper emits `StdTensorOp::Exp` for a constant. Tests at `support.rs:829` and `support.rs:844` assert `Constant`/`BroadcastInDim` structure and reject analytic constant shortcuts. **Closed as already fixed by #1324 after source and test verification.** | Shared semantic constant builders centralize dtype-aware emission, and graph-structure tests reject `Exp`, `Log`, `Sin`, `Cos`, `Tanh`, `Sqrt`, `Rsqrt`, `Expm1`, or `Log1p` as constant shortcuts. |

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

Issue #1309 was closed as stale/already fixed, citing commit
`9ad0fb1a96112bd1e0ee33a3c581198745923635` and its regression tests.

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

## Verification Performed

- `cargo fmt --all --check`: passed.
- The CI-parity workspace and tropical clippy commands passed with warnings
  denied.
- `cargo test --workspace --release`: passed after migrating the remaining
  runtime and XLA doctest call sites to the fallible traced API.
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
  followed by `python3 scripts/check-coverage.py coverage.json`: passed for all
  150 measured files; three configured files were excluded.
- `cargo doc --workspace --no-deps` followed by
  `python3 scripts/check-docs-site.py`: passed, including four guide dependency
  checks and rustdoc output for all 13 workspace crates checked by the script.
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD
  --output-json /tmp/repository-rules-review.json` on the committed head:
  passed with zero findings and no waiver.
- The sensitive-detector finding in the GPU source-contract test was a lexical
  false positive. Renaming the incidental lexeme resolved it; no detector
  suppression, baseline entry, or waiver was added.
- On an NVIDIA A100, the signed-padding ignored regression test was mutation
  checked RED/GREEN and passed with the fix restored. The associated CUDA
  checks also passed.
- The focused #1309 semantic-constant regression test passed, and the issue was
  closed as already fixed by #1324.
- `git diff --check`: passed for the implementation state before this final
  work-log update and is rerun for the documentation commit.

PR creation, merge, and closure of #1325, #1351, #1290, #1350, #1349, #1330,
and #1352 remain pending until the remediation branch is merged.

## Environment Notes

The linalg changes cover faer and LAPACK allocation/error paths. The final
release test and coverage runs completed with the available BLAS/LAPACK linker
configuration. Coverage generation required substantial target-directory and
JSON disk space but completed successfully. CUDA value verification used an
NVIDIA A100 with the documented CUDA runtime setup.

## Residual Risks

- The broad fallible traced-API migration touched runtime, AD, einsum, XLA,
  tutorials, and tests. The release workspace test passed, but downstream code
  outside this workspace may still require the same source migration.
- GPU signed-padding runtime parity was verified on an A100; other CUDA device
  generations remain covered by the same CubeCL path but were not sampled.
- #1352 remains correct only while every scatter launch preserves the checked
  host proof; the lexical contract tests intentionally fail when a new launch
  path or proof alias is introduced.
- Extreme allocation failures are validated mainly through checked helpers and
  source contracts because constructing near-`usize::MAX` matrices is not a
  practical runtime test.
- The three excluded issues remain open design questions and must not be
  silently folded into this remediation PR.
