# Phase 8 PR benchmark and CUDA gate

## Scope

This records the local PR-before-creation evidence after Phase 8 of the
execution-engine provider work. Phase 8 itself stops at the public XLA
`CompiledGraph` boundary; it does not add a new CUDA runtime engine and does
not change the eager benchmark source.

The formal same-repository PR GPU gate remains the trusted RunPod
`CI GPU gate`, which runs the archived CUDA and PJRT test binaries on a
reviewed CUDA host and publishes the required GitHub check.

## Host

- Date: 2026-07-25 JST.
- CPU: AMD EPYC 7713P 64-Core Processor, 64 cores, one socket, one NUMA node.
- GPU: NVIDIA A100 80GB PCIe.
- Driver: NVIDIA 580.126.09, reported CUDA API 13.0.
- Local installed CUDA toolkits: 12.0, 12.5, and 12.6. CUDA 12.8 NVRTC was not
  installed on this host.

## CPU eager benchmark

The Phase 1 eager gate benchmark binary was first built without CPU pinning to
avoid making release compilation single-core. The measured benchmark binary was
then run pinned to CPU 0:

```console
RAYON_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  cargo bench -p tenferro-ad --bench eager_dispatch_baseline --no-run

RAYON_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 target/release/deps/eager_dispatch_baseline-8e66b963d6115af8 --bench
```

The Criterion run completed successfully with 2 seconds warm-up, 5 seconds
measurement, and 100 samples per case. Compared to the 2026-07-20
`origin/main` worklog table in
`docs/worklogs/2026-07-20-eager-main-baseline.md`, every comparable median was
within +2.3% and every comparable upper 95% confidence bound was within +2.7%.
The largest increases were:

| Case | Baseline median | Candidate median | Candidate 95% CI | Median delta |
|---|---:|---:|---:|---:|
| materialized `reduce_sum_f64` size 1 | 9.161 us | 9.367 us | 9.333-9.409 us | +2.3% |
| materialized `neg_f64` size 1 | 8.866 us | 9.065 us | 9.041-9.079 us | +2.2% |
| materialized `neg_f64` size 8 | 9.046 us | 9.230 us | 9.218-9.257 us | +2.0% |

The worklog table does not contain the pre-refactor `slice_f64` rows, so those
cases are recorded as successful local Criterion measurements but are not used
for a worklog-table relative comparison here.

This is narrower than the immutable three-pair A/B protocol because the
current worktree does not retain a same-session `origin/main` Criterion result
set. It is sufficient local PR-before-creation evidence that Phase 7-8 did not
show a reproduced eager-path regression; the hosted PR checks remain the
promotion gate for CI integration.

## CUDA smoke and tests

The official smoke script selected CUDA runtime tier 12.8 from the CUDA 13.0
driver, but local execution with `--skip-nvrtc-install` failed before compile
because this shared host does not have CUDA 12.8 NVRTC:

```console
python3 scripts/ci/cuda_smoke_test.py --skip-nvrtc-install --min-vram-gb 16
```

The same smoke proof passed against the locally installed CUDA 12.6 tier:

```console
python3 scripts/ci/cuda_smoke_test.py \
  --skip-nvrtc-install \
  --min-runtime-version 12.4 \
  --full-runtime-version 12.6 \
  --min-vram-gb 16
```

Observed output included driver CUDA API 13.0, selected runtime tier 12.6, loaded
NVRTC 12.6, compute capability 8.0, 79.2 GB VRAM, successful NVRTC
compilation, PTX load, launch, synchronize, readback, and `SMOKE PASS`.

The CUDA package set used by the RunPod archive was also exercised locally:

```console
CUDA_PATH=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
RAYON_NUM_THREADS=1 \
NEXTEST_TEST_THREADS=1 \
  cargo test -p tenferro-gpu -p tenferro-ad -p tenferro-linalg --features cuda

CUDA_PATH=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-} \
RAYON_NUM_THREADS=1 \
  cargo test -p tenferro-gpu -p tenferro-ad -p tenferro-linalg --features cuda -- --ignored --test-threads=1
```

Both commands completed with exit code 0. The ignored GPU run covered the local
CUDA device paths: 1 ignored AD GPU fusion test, 108 ignored `tenferro-gpu`
CUDA tests, and 15 ignored `tenferro-linalg` CUDA linalg tests.

## Follow-up gate

After the PR is opened, babysit the required hosted checks, especially
`CI GPU gate`. The hosted RunPod path is still required to prove the repository
archive and PJRT flow on the workflow-selected CUDA runtime tier.

## Post-PR CI repair notes

PR CI at head `771b1b99` exposed integration issues that were not caught by
the Phase 8 local pre-PR checks:

- Extension AD bridge transpose reached the JVP operation families emitted by
  tropical and sparse linearization. The repair registers direct JVP-family
  semantic VJP/linear-transpose rules for the tangent inputs while preserving
  unsupported active-primal diagnostics.
- `ExtensionCacheStore::clear()` intentionally clears retained entries while
  preserving cumulative hit/miss/eviction/clear counters. The FFT cache test
  now asserts those two parts of the contract separately.
- The generated dependency SVG had drifted from the source DOT inventory and
  was regenerated.
- `cpu-blas` classifies the built-in BLAS provider as
  `GlobalOrUncontrolled`, so strict external-managed domain tests must install
  controlled test provider bundles at construction time. The production
  validation remains strict. Separately, provider-default-exclusive backend
  sessions now keep the public session callback outside the Tenferro-managed
  Rayon entry.

Fresh local verification after these repairs:

```console
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'cargo test -p tenferro-ad --test integration placement_bound_eager -- --nocapture' \
  --test 'cargo test -p tenferro-linalg cpu::tests::single_entry:: -- --nocapture' \
  --test 'cargo test -p tenferro-runtime graph::executor::tests::preflight::compiled_graph_dot_uses_the_installed_cpu_provider_slot_once -- --nocapture' \
  --test 'cargo test -p tenferro-cpu --test provider_boundary_allocation_tests warmed_public_session_request_provider_dispatch_does_not_allocate -- --nocapture' \
  --test 'cargo test -p tenferro-fft fft_plan_cache_is_bounded_lru_and_reports_known_retention -- --nocapture' \
  --test 'cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff --test tropical_ad' \
  --test 'cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff --test sparse_ad' \
  --test 'cargo check -p tenferro-cpu --tests --no-default-features --features cpu-blas' \
  --test 'cargo check -p tenferro-ad --test integration --no-default-features --features cpu-blas' \
  --test 'cargo check -p tenferro-linalg --lib --tests --no-default-features --features cpu-blas' \
  --test 'cargo check -p tenferro-runtime --lib --tests --no-default-features --features cpu-blas'
```

The command completed with `fast PR checks passed`. Full linked `cpu-blas`
execution remains CI-owned on hosts with BLAS libraries available.
