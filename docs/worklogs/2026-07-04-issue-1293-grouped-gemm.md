# Issue 1293 Grouped GEMM

## Session Summary

PR #1297 implements a backend-internal grouped GEMM path for CPU execution.
The public tensor API is unchanged; the new hook lives behind
`BackendCachedDot`/`SessionCachedDot` and accepts validated matrix jobs over
shared flat buffers.

After comparing an independent batched-`dot_general` prototype with #1297, the
PR keeps #1297's broader backend-hook design and folds in the prototype's
reviewed details: OpenBLAS group coalescing, explicit `Vec`/`SmallVec`
rationale, execution-local raw pointer descriptors, and a sorted output-overlap
validation pass.

## Context Read

- Issue #1293 discussion, including Rayon/faer behavior, OpenBLAS
  `gemm_batch`, descriptor caching, and BLAS thread-count concerns.
- PR #1295 merge state before implementation.
- PR #1297 implementation branch and local prototype diff.
- Shared tensor4all repository, performance, docs/test, and Rust performance
  rules.
- `AGENTS.md` and `REPOSITORY_RULES.md`, especially CPU provider features,
  faer/Rayon policy, BLAS provider threading policy, public-surface discipline,
  FFI safety, work-log requirements, and benchmark rules.

## Decisions

- Keep the grouped operation as hidden backend/session plumbing instead of a
  user-facing tensor API. The hook can serve batched `dot_general` lowering and
  future compiler/runtime dispatch without exposing raw offsets publicly.
- Validate dtype, scalar compatibility, buffer ranges, and pairwise-disjoint
  output ranges before provider execution. Output overlap is rejected; input
  reuse is allowed.
- Use faer with outer Rayon job parallelism under the owning `CpuContext`; each
  child GEMM receives `faer::Par::Seq` to avoid nested parallel fan-out.
- Use OpenBLAS `cblas_*gemm_batch` as the first BLAS provider candidate for
  small grouped jobs. The short benchmark showed OpenBLAS batch overhead can
  lose on medium or mixed-size jobs, so the provider falls back to the existing
  per-job BLAS path outside the measured small-job regime. The OpenBLAS path
  supports real and complex dtypes when `blas-openblas` is enabled; generic
  BLAS keeps the sequential provider fallback.
- Cache only structural metadata. Raw pointers are rebuilt per execution because
  tensor views and buffer-pool allocations can change between calls.
- Keep runtime job descriptors and OpenBLAS descriptor arrays in reserved
  `Vec`s, not `SmallVec`s. Group/job count is runtime-dependent and can exceed
  any inline threshold; OpenBLAS also requires stable contiguous C arrays for
  the immediate FFI call.

## Benchmarks

The PR adds:

```bash
cargo bench -p tenferro-cpu --bench grouped_gemm --features cpu-faer
cargo bench -p tenferro-cpu --bench grouped_gemm --no-default-features --features blas-openblas
```

Benchmark cases compare grouped execution with the existing N-call
`dot_general_read_into_accum` loop over:

- `uniform_small`: 64 jobs of 8x8x8
- `mixed_large_small`: one 64x64x64 job plus 31 jobs of 8x8x8
- `medium_blocks`: 16 jobs of 32x32x32

For OpenBLAS runs, provider thread variables should be pinned explicitly by the
caller, for example `OPENBLAS_NUM_THREADS=4`. There is no stable portable API
for reading the active OpenMP/OpenBLAS thread count, so benchmark commands must
record the environment used for the run.

Short local faer run with `--sample-size 10 --warm-up-time 0.1
--measurement-time 0.2`:

| Case | Grouped hook | Sequential N-call |
| ---- | ------------ | ----------------- |
| uniform_small | 111.99 us | 954.56 us |
| mixed_large_small | 161.47 us | 5.3152 ms |
| medium_blocks | 270.23 us | 80.764 ms |

Short local OpenBLAS run with `OPENBLAS_NUM_THREADS=4 --sample-size 10
--warm-up-time 0.1 --measurement-time 0.2` after the small-job `gemm_batch`
heuristic:

| Case | Grouped hook | Sequential N-call |
| ---- | ------------ | ----------------- |
| uniform_small | 40.706 us | 130.23 us |
| mixed_large_small | 21.267 us | 79.706 us |
| medium_blocks | 82.261 us | 98.769 us |

Before the heuristic, OpenBLAS `gemm_batch` was still faster for
`uniform_small` but lost on `mixed_large_small` and `medium_blocks`. The final
provider keeps `gemm_batch` for small jobs and uses the existing per-job BLAS
path for larger jobs.

## Verification

- `cargo fmt --all --check`
- `git diff --check`
- `cargo test -p tenferro-tensor grouped_gemm`
- `cargo test -p tenferro-cpu grouped --features cpu-faer`
- `OPENBLAS_NUM_THREADS=4 cargo test -p tenferro-cpu --no-default-features --features blas-openblas grouped`
- `OPENBLAS_NUM_THREADS=4 cargo test -p tenferro-cpu --no-default-features --features blas-openblas openblas_gemm_batch_heuristic_keeps_medium_jobs_on_sequential_path --lib`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `OPENBLAS_NUM_THREADS=4 cargo clippy -p tenferro-cpu --no-default-features --features blas-openblas --all-targets -- -D warnings`
- Short faer and OpenBLAS benchmark runs listed above.

## Residual Risks

- The OpenBLAS `gemm_batch` FFI uses OpenBLAS extension symbols that are not
  exposed by `cblas-sys`.
- Group coalescing currently preserves input job order and combines consecutive
  jobs with identical BLAS metadata. This improves uniform small batches without
  adding a reorder table; non-consecutive equal-shape jobs remain a possible
  future optimization.
- The hidden grouped hook uses flat-buffer offsets and assumes column-major
  matrix blocks. Callers should continue to build those jobs from validated
  layout/planning metadata rather than ad hoc public inputs.
