# Terasaki Bug Batch 4

## Summary

This work log records the follow-up remediation for GitHub issues #1212 and
#1222 through #1232. The batch focuses on user-visible numerical and shape
semantics, shared helper extraction where repeated mistakes had the same root,
and regression coverage for the issue reproducers plus same-class audit
findings.

Claude review was explicitly cancelled by the user before this implementation
pass because earlier attempts were too slow, so no Claude review was run for
this batch.

## Context Read

- `README.md`
- `REPOSITORY_RULES.md`
- Shared tensor4all rules: common repository, common performance,
  common docs-and-tests, Rust performance, Rust numerical
- GitHub issues #1212 and #1222 through #1232
- Prior work log `docs/worklogs/2026-06-26-issue-1209-1220-remediation.md`

## Classification Ledger

| Issue/finding | Classification | Resolution |
| --- | --- | --- |
| #1212 degenerate `Clamp` AD | Maintainer decision | User chose permissive `lower > upper` semantics. CPU clamp was aligned with `min(max(x, lower), upper)`, and AD coverage now checks degenerate bounds. |
| #1222 FFT `transpose_rule` ignores role | Auto fix / shared-helper parity | Added role-aware active-input helper usage and FFT coverage so `Primary` and inactive linearized paths do not emit invalid cotangents. |
| #1223 `DynamicTruncate` / `PadToMatch` dtype promotion | Auto fix | Runtime dtype inference now preserves the data operand dtype instead of promoting with shape/control operands. |
| #1224 `transpose_extract_diag` axis order | Auto fix | Diagonal transpose now maps rectangular reversed axes correctly. |
| #1225 XLA dot-general transpose skip | Auto fix | StableHLO lowering skips output transpose only when the permutation is identity and the shape already matches. |
| #1226 `TracedTensor::pad_to_match` shape hint | Auto fix | Traced metadata now uses the padded input shape rather than the reference tensor shape as the result hint. |
| #1227 einsum repeated output labels | Auto fix / shared helper | Added occurrence-aware label mapping and reused it across traced/eager/extension paths. |
| #1228 empty layout contiguity | Auto fix | Empty views are treated as contiguous even when degenerate strides appear before larger axes, with tensor-core and tensor-view coverage. |
| #1229 FFT C2C VJP length mismatch | Auto fix | C2C adjoints restore input length using runtime shape, dynamic truncate, and pad-to-match instead of returning the transformed length. |
| #1230 XLA F32 non-finite literals | Auto fix | StableHLO scalar formatting preserves f32 bit width and NaN payloads. |
| #1231 complex ordered comparisons | Explicit rejection | Runtime and CPU ordered operations now reject complex dtypes with diagnostics; users should compute abs/norm explicitly before ordering. |
| #1232 eager `svd`/`eigh` eps | Auto fix / shared constant | Eager and traced decomposition AD paths share `DEFAULT_DECOMPOSITION_AD_EPS = 1e-12`, preventing NaN gradients for vector-valued observables. |

## Shared Helpers And Preventive Measures

- `linear_transpose_input_active` centralizes the `OperationRole` /
  `active_mask` policy for standard AD transpose rules. This avoids each rule
  rediscovering how `Primary` direct-rule tests and inactive linearized inputs
  should behave.
- `map_label_occurrences` centralizes occurrence-sensitive einsum label
  mapping so repeated labels cannot accidentally reuse the first matching axis.
- `reject_complex_ordered_dtypes` in runtime shape inference and CPU dispatch
  keeps the no-total-order decision at public operation boundaries instead of
  relying on missing low-level trait impls.
- `DEFAULT_DECOMPOSITION_AD_EPS` keeps eager and traced linalg decomposition AD
  defaults in one place.

## Verification

Focused red/green and regression checks were run for each issue family,
including dtype propagation, clamp AD, extract-diag AD, FFT inactive-mode and
length-changing VJP paths, einsum repeated labels, XLA lowering, complex ordered
operation rejection, empty-layout contiguity, and eager linalg finite-gradient
coverage.

Final pre-PR workspace verification:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace --no-fail-fast`
- `cargo doc --workspace --no-deps`
- `cargo test --doc --release --workspace`

## Residual Risks

- Complex equality remains exact equality where equality is not an ordered
  operation; ordered comparisons and extrema are rejected instead of defining a
  magnitude order.
- The FFT C2C transpose path now handles runtime length restoration, but broader
  real FFT length-changing variants should still be audited when those APIs are
  extended.
