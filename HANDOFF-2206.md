# HANDOFF: tenferro-rs #2206 (+ #2207) implementation, 2026-10-20 ~00:45 JST

Stopped by maintainer request (token budget). Not committed to git; this file lives in the integration worktree only.

## Where things are

- Design (settled): tenferro-rs#2206 body (current state; comments are history), tensor4all-meta
  `docs/design/core-boundary-and-execution-ownership.md` (PRs #32-#37 merged). Bug #2207 (memoized device
  addresses vs CubeCL relocation) is fixed in the GPU branch below.
- Autopilot run: proxy-log `shinaoka-advisor/proxy-log/2026-10-20-tenferro-rs-2206.md` (plan, plan revision after
  pre-review, all decisions; tier-C items parked), run dir `shinaoka-advisor/work/runs/tenferro-rs-2206/`
  (task files, worker logs, reviews `pre.out`, `post-upstream.out`).
- Integration branch: `feat/2206-caller-engines`, worktree `/home/shinaoka/tensor4all/.worktrees/tenferro-2206`,
  HEAD `4cc49fed0`. Not pushed; `develop` not updated (workspace does not build end-to-end yet).
  Merged so far: S1 rename BackendSession->Engine, runtime R (prepare/run/Output/ExecutionEngine), CpuEngine C,
  CPU adapter integration, strided/provider pin bump + strided `with_execution_pool` switch, eager A
  (EagerRuntime takes engines), GPU G (CudaEngine::open, #2207 sweep, raw scope, home stream fix; A100 suite green),
  einsum E1, runtime/tensor tests T1, buffer-pool in-flight accounting removal.
  Known green on HEAD: tenferro-tensor, tenferro-runtime (lib+tests), tenferro-cpu(-basic) (all tests),
  tenferro-ad (default features; 1 trybuild test fails only on this host's /kache paths), tenferro-einsum
  (default + autodiff). tenferro-gpu green on its own branch (merged).

## Upstream (DONE, published with maintainer permission)
strided-basic 0.6.2 (strided-rs#323/#324, tag strided-basic-v0.6.2), tcontract-exec 0.3.1 + tcontract 0.3.2
(tcontract-rs#133), cpueinsum + cpueinsum-blas 0.3.2 (cpueinsum-rs#34), tlinalg + tlinalg-blas 0.3.2
(tlinalg-rs#67/#68). tenferro already pins these on the integration branch.

## In-flight units (branches off the integration branch; merge each into feat/2206-caller-engines)

| unit | worktree (`.worktrees/…`) | branch | state |
|---|---|---|---|
| L linalg (fork, stopped) | tenferro-2206-linalg | feat/2206-linalg | 2 commits (2d2efa93c src, 39b5d6ed7 tests); **19 uncommitted files**; doctests 198/199 (1 lib.rs failure); CUDA linalg A100 run not confirmed. Next: review/commit the dirty files, fix the lib.rs doctest, run `--features cuda -- --ignored` linalg, faer GEMM width-before-install (task in prompt), update cuda_test_partition.tsv rows. |
| F fft (fork, stopped) | tenferro-2206-fft | feat/2206-fft | 1 commit 016c19c9c (CPU side); clean tree. Next: CUDA FFT test targets, cuFFT plan ownership in engine, A100 run, partition rows. |
| G2 GPU follow-ups (fork, stopped) | tenferro-2206-gpu2 | feat/2206-gpu2 | commits d15f58085 (ad registers CUDA/WebGPU engine kinds), 5d285a107 (publication copies into any engine on the same device); **7 uncommitted files** (was adding prospective admission to sized transfers). Remaining: cross-engine transfer route (both-side ordering; acceptance tests), unfinished-memory accounting + broken device, WebGPU/Apple bridge (currently `compile_error!` under `webgpu` -> macOS CI profile fails). |
| E4a xla + ext proof crates (DeepSeek, still running) | tenferro-2206-xla-ext | feat/2206-xla-ext | worker merged feat/2206-linalg into its branch itself; 24 dirty files at stop time. Check `work/runs/tenferro-rs-2206/e4a-xla-ext.status` (STATUS line) and review before merging. |
| E4b ext/sparse + tropical (DeepSeek, still running) | tenferro-2206-sparse | feat/2206-sparse | just started. Status: `e4b-sparse.status`. |

DeepSeek workers continue without the orchestrator; their status files get a final `STATUS:` line. They never push.

## Remaining work after those units
1. einsum CUDA/WebGPU integration tests (`tests/integration/cuda_eager_tensor.rs`, `webgpu_eager_tensor.rs`, CUDA arm of `einsum_session_supported`).
2. samples/cubecl-kernel (use `with_raw`/`with_cubecl` + `cubecl-interop`), docs/tutorial-code, README, guides
   (`docs/guides/cuda-sessions-and-transfers.md`, eager guides), shipped skills cheatsheets (`.agents/.claude/.kimi`),
   `scripts/check-guide-dependency-snippets.py` snippets (bulk mechanical: Haiku on disjoint file sets).
3. Gates/scripts: `scripts/audit-session-entry.py` (+allowlist/tests; mechanisms removed), `scripts/check-pr-fast.sh` call,
   `scripts/run-session-route-performance-gate.sh` / `capture-session-route-baseline.py` (bench arms renamed to engine/*),
   REPOSITORY_RULES.md ownership text (`execution-entry: CpuBackend`; asserted by `backend_capability_contracts.rs:483`),
   PERFORMANCE_TIPS.md; retire superseded design notes (held session, phase lease, child ticket, shared scope) and add one
   tenferro design note linking the meta doc.
4. Acceptance tests from the #2206 body not yet written (nested k-engine thread count, small-op thread identity /
   pool-entry counts across strided/faer/tlinalg/contraction paths, GPU raw scope ordering, bounded unfinished work).
5. Local gate (`python3 scripts/ci/run_profile.py` fmt clippy workspace-faer workspace-blas blas-inject extensions docs
   ci-config; cuda/webgpu/cuda+webgpu checks; CUDA inventory; A100 run), post review (GPT/Astra), measurement vs main
   (dispatch bench at explicit 1T, same run), then push to develop and one PR develop->main.

## Parked for the maintainer (tier C, proxy-log)
- Prepare-time GPU kernel compilation (warm-up) — recommended: move back to #1885.
- tcontract-exec `WholeBatchVendor` removal — deferred (breaking); recommended: next breaking tcontract-exec release.

## Notes
- Core budget: 16 cores total (maintainer 2026-10-20); forks used taskset blocks 0-3/8-11/12-15, workers 4-5/6-7.
- No backward compatibility, no transitional shims (maintainer). Names are not important.
- Never write in the main checkout `/home/shinaoka/tensor4all/tenferro-rs` (it holds unrelated uncommitted work).
