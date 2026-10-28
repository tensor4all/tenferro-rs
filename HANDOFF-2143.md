# HANDOFF — #2143: merged; the acceptance measurements remain

**Untracked local handoff note. Do NOT `git add` this file and do NOT commit it.**

- Branch/worktree: `cpu_issue_burn` in `/home/shinaoka/tensor4all/tenferro-rs`
  (rebased on `origin/main` `9655b4e0d`, which already carries #2149).
- Commits: `be683609a` (six work items), `a10342d7b` (design record).
- PR: https://github.com/tensor4all/tenferro-rs/pull/2153 — **MERGED** as
  `e28d1e56e` (merged `--merge`, head branch deleted). #2152 (`43b41fa6e`) had
  already landed underneath it, so the merge commit carries both.
- Post-merge verification on `origin/main` (`e28d1e56e`):
  `cargo check --workspace --all-targets` clean; the CUDA test-partition entry
  for `cuda_axpby_accepts_strided_destination_regions` survived #2152's edits to
  `scripts/ci/cuda_test_partition.tsv`; the focused A100 `--ignored axpby` filter
  re-run on the merged state passes (3 lib + 4 `blas1_cuda` tests, incl. the new
  strided-destination region test).
- Remaining work is tracked publicly in the #2143 issue comment
  (issuecomment-6159163333). The benchmarks are being run on another machine.
- Local state: this worktree is detached at `origin/main`; the local
  `cpu_issue_burn` branch and its remote were deleted after the merge.
- Decision record (in the PR): `docs/design/cpu-structural-2143.md`.
- Working rules: repo `AGENTS.md`, `REPOSITORY_RULES.md`, `PERFORMANCE_TIPS.md`.
  Read those before touching anything. Build with `-j 16`.

## What is already done and locally verified

Items 1 (#2021), 3 (#2040), 4 (#1838), 5 (#2039), 6 (#2145, on top of upstream
#2149), and item 2 (#2029) needs no code change. Verified locally: workspace
`check`/`clippy` clean, `fmt`; `tenferro-tensor` `borrowed_view_allocation_tests`
(zero-allocation metadata views), `tenferro-cpu-basic` `view_allocation`
(zero-allocation `typed_view`, rank 9, canonical stored strides), `tenferro-cpu`
blas1 (strided/offset destination region, `MATERIALIZATIONS == 0`, alias and
rejection tests), `tenferro-fft` (allocation contract, lanes, reuse),
`tenferro-runtime` (426/174/511); CUDA `--ignored axpby` on the local A100
(driver 580.126.09, CUDA 12.6, cuTENSOR 2.5.0) exit 0 with 7 tests ok; doctests.
Pre-existing local-only failure: `tenferro-tensor --test storage_compile_contract`
fails because `kache` rewrites rustc paths (fails the same on pristine
`origin/main`; `CARGO_BUILD_RUSTC_WRAPPER=""` gives `2 passed; 0 failed`).

## Remaining work before merge

1. **Acceptance measurements** (the reason auto-merge is off). Use
   `tensor4all/tenferro-benchmark` `mwe/cpu_followup` at its frozen revision and
   the #2143 protocol (release, pinned idle core, effective thread count
   asserted, median of repeated batches, 1T and 4T compared in the same run):
   - #2029: reshape materialization 1T/4T **and** ≤1.5x PyTorch at 4T
     (`reshape_f64_concrete-shared_n33554432_to8192x4096` vs `_pytorch_*`).
   - #2021: session softmax reproducer (`[512, 512, 8]` f32) before/after, and a
     composite-level counter test asserting `broadcast_in_dim_read` is not
     entered on that path (add to `crates/tenferro-runtime/tests/`).
   - #2040: the 12 `followup_metadata_*` rows for no regression, and the
     `metadata_view_alloc_counter` row (already asserted in-tree).
   - #2039: the 8 `followup_fft_*` rows against `mkl-dfti` for no regression; the
     residual is the RustFFT-vs-DFTI engine difference, recorded as the reviewed
     reason — do not try to close the ratio in this PR.
   - Register/confirm the case IDs in `tensor4all/tenferro-benchmark#177` if any
     row is missing (`cpu_perop_view_alloc_counter`).
2. **Local PR gates**: `bash scripts/check-pr-fast.sh --coverage-reviewed --test
   '<focused test>'` and `python3 scripts/repository-rules-review.py --base
   origin/main --head HEAD --output-json /tmp/repository-rules-review.json
   --dry-run --llm-skipped-reason "local deterministic review"` on the committed
   head; full workspace tests and `cargo doc --workspace --no-deps`.
3. **Independent post-review** of the finished diff (`independent-review` skill,
   post mode, `openai/gpt-6.1-sol`). Verify each finding against source before
   acting; an unsupported finding is not a reason to add code.
4. **PR discipline**: no topping-up pushes. If the measurements require code
   changes, replace the branch with one finished state in a single push, or
   withdraw #2153 and reopen when complete. Spend the hosted GPU gate once, on
   that finished state.
5. **DONE (#2152 merged underneath #2153)**: the rebase, the CUDA test-partition
   check and the focused A100 `axpby` re-run were all performed on the merged
   state; see the post-merge verification above. Re-check
   `crates/tenferro-ad/src/eager_ops.rs` against the additive
   `CompositeOps::broadcast_in_dim_owned` default method if it changes again.
6. **DONE**: merged without auto-merge, `--merge` (not squash), head branch
   deleted. Nothing further is required on the PR itself.

## Separate, do not bundle into #2153

- **#2150**: CPU `reduce_prod` returns NaN for a long axis whose product is
  exactly representable (lane-wise SIMD partial products overflow). Own PR,
  own branch from `origin/main`. Path: `crates/tenferro-cpu/src/reduction.rs`
  and the product reduction in
  `crates/tenferro-internal-cpu-kernels/src/elementwise.rs`; tests in
  `crates/tenferro-cpu/src/reduction/tests/mod.rs` and
  `crates/tenferro-cpu/src/tests/cpu_tests/basic_ops.rs`. First reproduce at
  len 1000 (`2.0, 0.5, 2.0, 0.5, …` must reduce to `1.0`), then decide whether
  the defect lives in tenferro's kernel or in strided-rs (REPOSITORY_RULES.md:
  a generally useful missing primitive goes to strided-rs first).
- Out of scope for #2143: #2139, #2140, #2141, #2142, GPU FFT, any `strided-rs`
  change or pin bump, any new provider/`CpuContext`/`Pool` surface.
