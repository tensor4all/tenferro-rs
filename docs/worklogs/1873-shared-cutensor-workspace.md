# Shared cuTENSOR contraction scratch per stream slot (#1873)

## Decisions

Base: `c3bf88cc3`; branch: `fix/1873-shared-cutensor-workspace`.

- Replace the per-plan, per-stream-slot scratch buffers with one lazily grown
  workspace per physical stream slot, shared by every cached contraction plan
  on that slot. A plan now records only the scratch size it needs.
- The plan cache owns the shared scratch, and its mutex still serializes host
  enqueues, so reuse on a slot stays ordered by the physical CUDA stream.
- Keep #1809's event-based retirement, but change its trigger: individual plan
  eviction no longer drops scratch. Growth, a lowered retention cap,
  whole-cache eviction, clear, and backend teardown retire the allocation.
  Growth allocates the replacement before dropping the current buffer, so a
  failed allocation cannot cost a slot its usable scratch.
- Exclude shared scratch from the plan and extension-cache byte budgets. That
  budget drove the reported failure: one scratch allocation per plan made the
  contraction entry exceed the 64 MiB default and evict the whole typed entry,
  discarding every plan and the configured plan bound with it.
- Bound retained scratch with a separate, configurable backend-wide cap
  instead of the cache budget. `0` disables retention; it is not "unlimited".
  The cap bounds what the backend keeps for reuse, never what a contraction may
  use: a request that does not fit the remaining headroom runs in an
  exact-size temporary workspace that is retired afterwards, and lowering the
  cap releases retained buffers without evicting a plan. Other slots are never
  evicted to make room. There is no cap-specific error path.
- Default cap: 10 GiB. There is no single optimal cap, so the knob is what
  matters. The default is a finite policy value chosen to be permissive enough
  that a large workload is not silently degraded by the default, and it is
  documented as neither a practical memory protection nor a reservation; on a
  device with less free memory than the cap, it simply never binds. The
  maintainer raised this from the initial 1 GiB after review: the cap only ever
  bounds idle retention, so a permissive default trades idle device memory for
  not losing scratch reuse, and the high-water statistic makes the real demand
  observable. The cost is that a large peak leaves that much reserved until
  `clear_cuda_extension_cache` or a smaller cap is set. `u64::MAX`
  (effectively unlimited by default) was still rejected: it leaves no practical
  protection in place for a caller that never configures the cap.
- Growth rounds up to a power of two with a 1 MiB floor, falls back to the
  exact request when the rounded capacity does not fit, and falls back to a
  temporary workspace when the request does not fit either. Rounding overflow
  takes the exact-size path rather than failing.
- The cap is stored on the backend, not in the extension-cache entry, so it
  survives `clear_cuda_extension_cache`, typed-entry eviction, and recreation,
  and is shared by clones of the backend.
- Keep the existing 64-plan default, the existing cache controls, and the
  existing `cuda` feature. No new dependency. Add
  `CudaBackend::cutensor_workspace_stats`, `CudaBackend::cutensor_workspace_bytes`,
  and the cap getter/setter; `clear_cuda_extension_cache` releases the scratch.
- Add `CudaBackend::cutensor_workspace_temporary_uses` so a caller can tell
  whether the cap is binding. The retained high-water alone cannot show this:
  when the cap binds, the high-water is capped and the real requirement is
  invisible. The counter is a backend-level cumulative `AtomicU64` (shared by
  clones, not reset by a cache clear), so it costs one relaxed add only on the
  already-degraded temporary path.

Rejected alternatives:

- Keeping per-plan scratch and only removing the workspace bytes from the byte
  budget. That stops the whole-cache eviction but still retains one scratch
  allocation per cached shape, which is the memory and retirement cost the
  report measured.
- Refusing a contraction that exceeds the cap. Scratch is required to execute a
  contraction, so refusing would turn a memory-policy knob into a correctness
  failure. The design keeps retention and executability separate.
- A per-slot cap instead of a backend-wide total. The total expresses any
  per-slot split and does not have to know the stream slot count; the cost is
  documented slot starvation when the cap is below the steady-state working
  set.
- Counting shared scratch in the extension-cache budget or stats. It would
  reintroduce eviction pressure for a single allocation that every plan
  depends on.

## Verification conclusions and constraints

- New regression test `cuda_cutensor_shared_workspace_does_not_evict_plan_cache_by_bytes`
  reproduces the report's shape: two plans, a 2-entry bound, and a 64 KiB
  extension-cache byte budget below the shared-workspace floor. It asserts two
  misses then a hit, zero evictions, a preserved plan bound, retained
  extension-cache bytes below the budget, scratch at or above the 1 MiB floor,
  scratch released by `clear_cuda_extension_cache`, and numerical agreement
  with the CPU backend after the clear. It also asserts that the default cap is
  not binding (zero temporary uses).
- New cap tests, all executed on an A100 80GB PCIe (driver 580.126.09,
  cuTENSOR 2.5.0):
  `cuda_cutensor_zero_retention_cap_runs_without_retaining_scratch` (zero
  retained entries and bytes, plan reuse and numerical agreement intact, and
  exactly one `cutensor_workspace_temporary_uses` per contraction),
  `cuda_cutensor_small_retention_cap_is_respected_and_keeps_plans` (retained
  bytes at or below the cap, plans intact),
  `cuda_cutensor_shrinking_retention_cap_releases_scratch_and_keeps_plans`
  (cap lowered to zero releases scratch, plan entries and eviction count
  unchanged, the plan still hits and still matches the CPU reference), and
  `cuda_cutensor_default_retention_cap_survives_cache_clear`.
- Device-free host tests cover the capacity rounding, the retention decision
  table (reuse, rounded retain, exact-size retain, temporary, zero retention,
  and `u64::MAX`-adjacent arithmetic), and the 10 GiB default. The public
  getters are exercised on a constructed backend in the CUDA lane; no fake
  runtime is introduced.
- All 25 CUDA GEMM unit tests (`cubecl::tests::gemm_tests`,
  `cubecl::tests::gemm_accum_tests`), including the f32/f64/C32/C64 numerical
  cases and the updated eviction/retirement tests, pass on that GPU.
- The shared-scratch control methods carry runnable doctests that gate on
  `gpu_available()` and fall through on a host without a CUDA driver. They are
  real usage examples rather than `no_run` or path-only ones, and they cannot
  panic where the driver library is absent.
- Host-side gates pass: `cargo check -p tenferro-gpu --features cuda
  --all-targets`, `cargo test -p tenferro-gpu --features cuda --lib`
  (104 passed, 185 ignored), CUDA-feature all-target clippy with
  `-D warnings`, `cargo fmt --check`, the source-contract integration tests,
  and `scripts/check-pr-fast.sh`.
- Repository-rules review verdict: pass. It raises one remaining decision: the
  cap's default is a policy value, and the pool's bound is a retention cap
  rather than a device-memory cap. Both are recorded in the design doc and
  rustdoc; the value is left for maintainer review.
- Allocation failure preserving the existing buffer is enforced by the
  allocate-before-replace ordering and asserted by the source-contract test;
  it is not triggered on hardware, because forcing a real device allocation
  failure on an 80 GB card is not a practical test.
- Barrier-fallback counts can be observed with
  `CudaBackend::cutensor_workspace_retirement_stats`, but zero fallbacks is not
  an acceptance criterion. The only added counter is
  `cutensor_workspace_temporary_uses`, which is a cap-policy diagnostic rather
  than a retirement or performance counter.
- Not measured: this change is not claimed as an end-to-end speedup. The
  report's caller counts come from code predating #1809 and its timing
  comparison changed several settings, so neither isolates this change. The
  10 GiB default is not derived from a measured workspace distribution.
- Pre-existing environment limitation: the trybuild UI test
  `session_contract::execution_session_capability_cannot_project_or_escape_owner_borrow`
  fails in this checkout because the build cache rewrites compiler paths in the
  emitted diagnostics; it is unrelated to this change.
