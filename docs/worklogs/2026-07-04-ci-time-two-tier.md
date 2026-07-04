# CI Time Reduction: Two-Tier PR/main Workflow

## Session Summary

Reduced pull-request CI wall-clock time without increasing GPU-runner billing.
The GPU runner was already gated behind the non-GPU checks, so the constraint
"only use the GPU runner after other tests pass" was preserved throughout. The
work restructures three areas:

1. **Parallelize the CUDA test archive build.** `cuda-archive` runs on a cheap
   non-GPU runner but was gated behind `pre-gpu-gate`, putting its ~9 min
   compile serially on the critical path. It now builds in parallel with the
   non-GPU CI. Only the expensive GPU runner (`cuda-run`) stays gated.
2. **Two-tier workspace tests.** Pull requests run `cpu-faer` always and
   `cpu-blas` only when backend-relevant paths change (dynamic matrix). Push to
   `main` runs the full `cpu-faer` + `cpu-blas` matrix (comprehensive tier), so
   every backend is exercised on the merged result.
3. **Extract extension/sample tutorial tests into one job.** `tropical`,
   `sparse`, and the KdV sample are standalone workspaces that recompile the
   tenferro graph from scratch (~11 min combined). They previously ran inside
   *both* backend legs of the matrix; they now run once in a dedicated
   `ext-samples` job (backend-agnostic), with `rust-cache` pointed at their
   target dirs.

The GPU gate's prerequisite list was also trimmed from 8 checks to 5, keeping
the checks that best predict a broken PR (fmt, clippy, coverage, the workspace
test aggregate, repository policy) and dropping three that still block merge via
branch protection but need not hold up the GPU runner (docs-site, blas inject,
tensor core dependency boundary).

## Context Read

- `.github/workflows/{ci.yml,ci-pr-workspace-tests.yml,CI_gpu.yml,review_bot.yml}`.
- `scripts/check-coverage.py` and `coverage-thresholds.json` (per-file absolute
  thresholds, default 80, 32 overrides, 11 exclusions; supports `--report-only`).
- Branch-protection required checks (via API): `rustfmt`, `coverage`,
  `docs-site`, `cargo test (blas inject)`, `CI gate (PR workspace tests)`,
  `CI GPU gate`.
- Actual check-run names on a recent PR head (to confirm gate name matching and
  that renamed matrix legs are not required checks).

## Measured Baseline (single-run samples, not rigorous benchmarks)

Per PR, the three workflows:

| Workflow | Wall time | Notes |
|---|---|---|
| CI (`ci.yml`) | ~11 min | slowest job = coverage (11.4 min) |
| CI PR workspace tests | ~25 min | cpu-faer / cpu-blas matrix (parallel) |
| CI_gpu | ~39 min | **critical path** |

CI_gpu breakdown (run 28688363421):

```
0..25min   pre-gpu-gate (waits for the non-GPU checks)
25..34min  cuda-archive (9 min compile, cheap runner, serial after the gate)
34..39min  cuda-run (5 min, ubuntu-gpu)
39min      cuda-gate
```

Per-backend workspace-test breakdown: `nextest --workspace` ~12 min, tropical
ext ~5.5 min, sparse ext ~5.3 min, doc/kdv/setup ~2 min. The ext tutorial steps
used the same command in both legs (they ignore `matrix.feature_args`), so they
were duplicated across backends and their target dirs were not covered by
`rust-cache`.

## Chosen Design

- **`CI_gpu.yml`**: remove `needs: [pre-gpu-gate]` and the archive `if:` gate so
  `cuda-archive` runs in parallel; `cuda-run` keeps `needs: [pre-gpu-gate,
  cuda-archive]`. Trim `pre-gpu-gate.required` to `repository rules review`,
  `rustfmt`, `clippy`, `coverage`, `CI gate (PR workspace tests)`.
- **`ci-pr-workspace-tests.yml`**: add `push: [main]`; add a `changes` job that
  emits the backend matrix as JSON (`[faer]` or `[faer, blas]`); `ci-maintainer`
  consumes it via `fromJSON`; new `ext-samples` job; `ci-gate` now also requires
  `ext-samples`. Concurrency group keyed per-PR for PRs, per-commit for pushes.

Projected critical path after the change: `pre-gpu-gate` waits on
`max(CI ~11, workspace ~14) ≈ 14 min` (cpu-blas usually skipped, ext extracted),
archive is already built in parallel, `cuda-run` +5 min ⇒ **CI_gpu ≈ 20 min**
(from ~39). Numbers are estimates pending post-merge confirmation.

## Decisions And Rejected Alternatives

Explicit maintainer decisions this session:

- **Coverage stays a hard, blocking, per-PR check.** Rejected switching PRs to
  `--report-only` (with strict enforcement moved to main) and rejected switching
  to diff/patch coverage. Coverage friction is accepted as the cost of the
  quality bar. Coverage is not the time bottleneck, so no coverage change was
  needed for the time goal.
- **Comprehensive tier lives on `push: main` only** (no nightly schedule).
- **`cpu-blas` runs full when triggered**, not a backend-sensitive subset.
  Rejected the subset because compile time dominates test-execution time in
  these jobs, so subsetting saves little while adding a fragile "which tests are
  backend-sensitive" boundary. The savings come from skipping the lane entirely
  when no backend path changed.
- **GPU gate keeps `coverage`** even though it is not the timing bottleneck,
  because coverage fails frequently for this repo and gating on it avoids
  spending GPU minutes on PRs that will fail coverage anyway.

## Residual Risks

- **Opt3 (ext target-dir caching) benefit is unconfirmed.** `rust-cache` may
  clean the tenferro path-dependency artifacts from the standalone-workspace
  caches; if so the ext build stays cold. Worst case is no speedup (harmless),
  and wall-clock is already bounded by the cpu-faer leg. Confirm from CI timing.
- **`main` may go red post-merge** if a merged change breaks the `cpu-blas` lane
  that its PR skipped. This is the accepted trade-off of running the full matrix
  only on `main` push. The full matrix still runs on any PR that touches backend
  paths, so backend-affecting changes are caught pre-merge.
- **Path-based `cpu-blas` selection** depends on the backend path list in the
  `changes` job (tenferro-cpu, tenferro-linalg, tenferro-tensor,
  tenferro-tensor-core, Cargo.lock/toml, the workflow file). If a future backend
  crate is added, extend this list.
- Renamed matrix leg check names (`CI (same-repo / heavy / …)`) are **not**
  branch-protection required checks; only `CI gate (PR workspace tests)` and
  `CI GPU gate` are, and both job names are preserved verbatim.

## Verification

- `actionlint 1.7.12` on both changed workflows: clean except the pre-existing
  `ubuntu-gpu` custom self-hosted label warning (unchanged line, false positive).
- Confirmed all 5 trimmed GPU-gate names and all 6 branch-protection required
  check names still map to real reported checks.
- Rust build/test/coverage/doc checklist items are N/A: this change touches only
  `.github/workflows/` and this worklog, with no Rust source changes.
