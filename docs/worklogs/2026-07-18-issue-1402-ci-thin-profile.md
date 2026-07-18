# Issue #1402 CI Thin Profile

## Session summary

Switched hosted Rust CI from `--release` to workspace `[profile.ci]`:
`inherits = "test"`, `incremental = false`, `strip = "symbols"`. Local
`[profile.dev]` / `[profile.test]` remain incremental. Shared command source is
`scripts/ci/run_profile.py`; CUDA archives in `runpod-gpu-test.yml` and
`CI_gpu.yml` use `cargo nextest archive --cargo-profile ci`.

## Context read

- [#1402](https://github.com/tensor4all/tenferro-rs/issues/1402) and parent #1401
- Local macOS PoC comments on #1402 (thin compile/run, `strip="debuginfo"` no-op,
  `strip="symbols"` 14 GiB → 8.3 GiB)
- `docs/worklogs/2026-07-17-issue-1397-build-artifact-reduction.md`
- `scripts/ci/run_profile.py` and CI workflow contracts

## Chosen design

- Explicit `[profile.ci]` rather than env-only overrides, so hosted cold builds
  are named and reviewable.
- Keep `--cargo-profile ci` / `--profile ci` in `run_profile.py` so YAML callers
  stay thin.
- Bump CUDA archive / rust-cache keys so previous `--release` artifacts are not
  reused.
- Log `du`/`df` after workspace and coverage jobs for acceptance evidence.

## Rejected alternatives

- Changing default `[profile.test]` to non-incremental/stripped: would slow
  local AI edit-test loops.
- `strip = "debuginfo"` alone: no size change after `debug = 0` on macOS PoC.
- Keeping `--release` for coverage only: would leave the heaviest cold-compile
  path on the old profile without a measured reason.

## Residual risks

- Hosted Linux peak disk and free-space still need confirmation in CI logs.
- `strip = "symbols"` weakens CI backtraces; test names/assertions remain.
- Coverage with `debug=0` + strip must still pass thresholds on GitHub runners.
- CUDA/PJRT archive wall-clock target (<10 minutes) is measured after merge into
  the GPU workflow path, not locally on macOS.
