# CUDA permutation materialization: the layouts the native entries cannot address (#2528 stage 3)

## Decisions

- The rows #2551 and #2581 could not reach are the ones whose coalesced view is still
  above `t4a_cuda_kernels::MAX_RANK` (8): `reverse_18d_3` (rank 18, `3^18`) and
  `reverse_23d_128x2` (rank 23). Their source strides are strictly descending, so no two
  destination axes merge: #2581's merge rule and #2551's pair-swap test both refuse, and
  the view falls to `t4a_cuda_kernels`' device-geometry entries, which decode every axis
  of every element. 8.6 GB of traffic in 30–85 ms is 0.1–0.3 TB/s, i.e. index arithmetic,
  not bandwidth.
- Three in-repo repairs were checked and rejected:
  - *merge axes*: impossible, no adjacent destination axis continues in the source;
  - *compose adjacent pair swaps*: a rank-18 reversal needs 9 of them, ~96 ms at the
    measured per-pass cost — 3x worse than the 30 ms it would replace;
  - *collapse the reversal to a flat negative-stride copy*: false. Reversing the axis order
    is a digit reversal in mixed radix; `dst[f] = src[N - 1 - f]` would need `s[k] = -c[k]`,
    while the view has `s[k] = c[r - 1 - k]`.
- The fix is `cutensorPermute` — the issue's own alternative, and the implementation PR
  #1509 added before the CubeCL era removed it with the old engine (#2312). Selecting it
  only where the native entries cannot serve a view keeps the no-silent-fallback rule
  (`REPOSITORY_RULES.md`, "GPU Backend Contract"): a view the tiled or by-value entries serve
  never reaches the vendor, and a routed view reaches a native kernel only when the *setup* of
  the vendor route failed (`PermuteFailure::Unusable` — no library, no handle, a refused
  descriptor/operation/plan), where the catalogue's wide entry is a complete provider for the
  same layout and the fallback is counted in `vendor_permutation_fallbacks`. A failed vendor
  *call* is an error, never a second attempt: the destination may be partly written.
- The route predicate (`cuda/structural.rs::vendor_permutation_applies`) is deliberately
  narrow — rank above `MAX_RANK` after merging, F32/F64/C32/C64, no zero or negative
  stride, a 16-byte-aligned base, different source and destination allocations, and a rank
  within the widest this repository demonstrates (24). A wider predicate would have moved
  `rotation_6d` (rank 6) and both `tn_light` views (rank ≤ 8) off `materialize_strided`,
  which they already serve, and would have made cuTENSOR mandatory for layouts that work
  vendor-free today.
- `copy_read_into`'s per-element `strided_to_strided` on a permuted destination takes the
  same vendor call when the reordered view is above the rank limit, with the caller's
  buffer: no allocation, no fill. A fresh destination is allocated zeroed instead, because
  the raw scope that registers operands for the library can only register initialised
  buffers; that fill is the whole difference between the fresh and the reused arms below.
- The permutation plan shares the handle's 32-entry plan cache with contractions under a
  reserved key prefix (`-1`, where a contraction key starts with a dtype tag in `0..=4`),
  and `cutensorPermute` takes no workspace, so no workspace buffer is allocated for one.
- Deliberately not in this PR: a native N-D tiled permute kernel (two tiled axes and one
  outer axis). It is the durable vendor-free fix and needs a new `t4a-cuda-kernels` entry,
  an ABI revision and a crates.io release before this repo can pin it — two PRs plus a
  release chain, which the issue's "one PR" and `profile/escalate.md` item 4 exclude.
  It is the recommended follow-up if the maintainer does not want the vendor owning
  permutation.

## Profile (A100 80GB PCIe, driver 580.178.04, CUDA 12.6.3, `nsys profile -t cuda`,
`PATTERN_ID=reverse_18d_3`, one suite process per binary)

Before (`origin/main` = `aebe8cb32`):

| kernel | instances | avg |
|---|---|---|
| `t4a_cuda_strided_to_strided_any` | 19 | 61.02 ms |
| `t4a_cuda_materialize_strided_any` | 38 | 28.41 ms |
| `cutensor_internal_namespace::permute_coop<…>` | 19 | 5.01 ms |

After (this branch):

| kernel | instances | avg |
|---|---|---|
| `cutensor_internal_namespace::permute_coop<(int)80, double, …>` | 76 | 5.03 ms |

All four arms' timed calls are one vendor permute kernel; no native materialization or
strided-to-strided kernel remains on the row.

## Measurement

Protocol (mirrors #2581): baseline `aebe8cb32` and candidate measured in the same session
with the same harness binary build, 5 processes per binary interleaved B, C, B, C, …,
`taskset -c 8-15`, the GPU confirmed idle before every process, no retries,
`LD_LIBRARY_PATH`/`TENFERRO_CUTENSOR_PATH` pointing at the installed cuTENSOR 2.8.1,
`GPU_BENCH_DEVICE=0`. Each process's `median_ms` is one sample; the table reports the
median of five. rustc 1.100.0. Raw records: `gpu-permutation-base.jsonl`,
`gpu-permutation-cand.jsonl` (170 rows each, `BENCH_OUTPUT` JSON Lines), run order and one `exit 0` per process in
`protocol.txt` (the GPU-idle refusal would have aborted the run instead). Every row reports `status: ok`, `correctness: passed`,
`verification_mode: sampled` in all 10 processes.

| row | arm | before | after | ratio | cuTENSOR arm |
|---|---|---|---|---|---|
| transpose_2d_32768_16384 | transpose | 5.587 | 5.587 | 1.00 | 5.401 |
| | to-contiguous | 5.731 | 5.733 | 1.00 | |
| | destination-reuse | 5.580 | 5.576 | 1.00 | |
| transpose_3d_1024_1024_512_102 | transpose | 5.309 | 5.307 | 1.00 | 5.309 |
| | to-contiguous | 5.433 | 5.433 | 1.00 | |
| | destination-reuse | 5.287 | 5.289 | 1.00 | |
| transpose_3d_1024_1024_512_201 | transpose | 5.212 | 5.213 | 1.00 | 5.336 |
| | to-contiguous | 5.316 | 5.315 | 1.00 | |
| | destination-reuse | 5.179 | 5.182 | 1.00 | |
| rotation_6d_64_32_32_32_16_16 | transpose | 7.252 | 7.252 | 1.00 | 5.403 |
| | to-contiguous | 7.277 | 7.277 | 1.00 | |
| | destination-reuse | 7.251 | 7.248 | 1.00 | |
| **reverse_23d_128x2** | transpose | 42.108 | **8.120** | **5.19** | 5.657 |
| | to-contiguous | 42.106 | **8.271** | **5.09** | |
| | destination-reuse | 85.055 | **5.671** | **15.00** | |
| **reverse_18d_3** | transpose | 30.159 | **6.777** | **4.45** | 5.027 |
| | to-contiguous | 30.121 | **6.788** | **4.44** | |
| | destination-reuse | 61.192 | **5.037** | **12.15** | |
| cyclic_18d_3 | transpose | 9.907 | 9.908 | 1.00 | 4.780 |
| | to-contiguous | 9.907 | 9.914 | 1.00 | |
| | destination-reuse | 9.903 | 9.901 | 1.00 | |
| tn_light_415_24d_scattered | to-contiguous | 8.285 | 8.276 | 1.00 | 5.477 |
| tn_light_415_24d_contiguous | transpose | 6.145 | 6.151 | 1.00 | 5.206 |
| | to-contiguous | 6.205 | 6.197 | 1.00 | |
| | destination-reuse | 6.145 | 6.141 | 1.00 | |

Against the cuTENSOR arm on the same run: the two reused-destination arms are 1.002x and
1.002x of it, and the fresh-destination arms are the vendor arm plus one destination fill
(8.12/8.27 ≈ 5.66 + 2.4 ms for 4.29 GB; 6.78/6.79 ≈ 5.03 + 1.75 ms for 3.1 GB), which is
the cost #2581 measured for the same fill. No cell regressed: the two control rows and
every untouched row are within 0.5 % of baseline.

## What this does not fix

- `cyclic_18d_3` (2.08x behind the vendor) is #2581's tiled transpose with a merged source
  run only 3 wide, so 3 of a tile's 16 lanes carry data; `rotation_6d` (1.34x) and both
  `tn_light` views (1.18x/1.53x) stay on `materialize_strided`. All four need a better
  native kernel (option A above or a tile-config sweep), not a routing change — deliberately
  outside the vendor predicate, since the vendor must not take over layouts the native
  entries serve.
- `tn_light_415_24d_scattered` has no destination-reuse arm in the suite (the harness runs
  that arm only where the destination is a permuted compact span).

## Measurement-harness adaptation

The benchmark worktree (`tbench-2528`, `tenferro-benchmark` `41d25b1`) does not build
against this revision of `tenferro-rs`: its feature table still names
`tenferro-ad/native|blas` and its `src/lib.rs`/`src/tensornetwork.rs` still use
`EagerEinsumExt`, `EagerTensor::from_tensor_in` and `TraceContext::einsum_subscripts_with`,
all of which this revision removed. Both measured revisions therefore used one scratch
build of that worktree with those two obsolete wiring points neutralised — the feature
entries dropped and the tensornetwork support, which no `gpu/permutation` row touches,
excluded. The `benchmark_gpu_permutation` source itself is unmodified, and both revisions
were built from it with `--no-default-features --features cuda`. This is a harness-side
gap (the benchmark repo lags tenferro-rs main), not part of this change.

## Tests

- `cargo test -p tenferro-gpu --features cuda --lib`: 48 host tests pass.
- `TENFERRO_REQUIRE_GPU=1 … -- --ignored --test-threads=1`: 33 device tests pass.
- `cargo nextest archive --cargo-profile ci -p tenferro-gpu -p tenferro-ad -p tenferro-linalg
  -p tenferro-fft -p tenferro-einsum --features cuda,tenferro-einsum/autodiff` followed by
  `python3 scripts/ci/gpu_test_partition.py --kind cuda --lane host`: inventory validated
  and 1416 host tests passed (377 skipped to the device and compile-only lanes), which is
  the check that the partition rows above satisfy.
- New: `vendor_permutation_routing_takes_only_a_wide_supported_view` (host; every row of
  the routing rule, including the dtype, broadcast, negative-stride, alignment,
  shared-allocation and rank-cap refusals),
  `a_wide_reversal_reads_through_the_vendor_permutation` (rank 18, values against
  `host_compact`, vendor counter +1),
  `a_wide_reversal_is_written_into_a_reused_destination` (rank 23, destination reuse,
  sentinels past the span intact, vendor counter +1),
  `a_wide_cyclic_rotation_stays_on_the_tiled_transpose` (rank 18, tiled counter +1 and
  vendor counter unchanged), and a vendor-counter assertion in the existing
  rank-3 generic-path test.
- New, for the plan-build cleanup: `a_refused_vendor_plan_leaks_no_native_handle` injects a
  refusal after 1, 2, 3 and 4 native handles exist, and asserts that (a) the copy is still the
  `host_compact` reference, (b) no vendor permutation was counted, (c) exactly one fallback was
  counted per stage, (d) the plan cache did not grow, and (e) handles created equal handles
  destroyed across the four stages. With `PlanBuild::drop` disarmed the last assertion fails,
  so the test pins the fix rather than the code's shape.
  `a_refused_vendor_plan_keeps_the_reused_destination_copy` does the same on the
  destination-reuse arm and checks the sentinels past the span, so the caller's buffer is
  neither over- nor under-written when the vendor route falls away.

## Post-change re-measurement

The review fixes only change failure paths, but the routed call site was restructured, so
`reverse_18d_3` was measured again on the same host with the same command
(`taskset -c 8-15`, `--repeat 5`, one process): `tenferro-cuda-transpose` 6.762 ms (916.71
GB/s), `tenferro-cuda-to-contiguous` 6.778 ms, `tenferro-cuda-destination-reuse` 5.031 ms
(1232.12 GB/s), `cutensor` 5.024 ms. The campaign figures above (6.777 / 6.788 / 5.037)
are unchanged within run-to-run spread, and the routing tests assert the vendor counter
moves, so no row fell back to the catalogue's entry.

## Reviews

- Plan review: `ollama-cloud/glm-5.3-flash` (GLM-5.3-flash), read-only, on
  `IMPLEMENTATION-PLAN.md` before any edit. Verdict `CHANGES`; all three blockers fixed
  before implementing (predicate scope restricted to rank above the by-value limit; the
  aliasing claim corrected to a predicate refusal instead of an enforcement that does not
  exist; the acceptance bars split for reused and fresh destinations), plus the two
  importants (no `cudarc` feature path, cache-key collision) and the minors.
- Post-implementation review: `ollama-cloud/deepseek-v4.1-flash` (reviewer-flash-opencode-go),
  read-only, on the committed diff. Verdict `CHANGES`; every finding fixed in the same commit:
  - IMPORTANT: the `copy_read_into` vendor arm submitted a descriptor without the span proof
    every sibling route gets, so a view reaching past its allocation would have been read out of
    bounds by the library. The arm now validates and plans the view
    (`LayoutRef::new` + `plan_materialize_strided`, `structural.rs`) before the vendor call, which
    is also what makes `permute_into`'s documented precondition true for both callers.
  - MINOR: `VENDOR_PERMUTATION_MAX_RANK`'s doc block was spliced into `region_is_flat`'s; moved.
  - MINOR: the 4-byte alignment branch of the predicate is now pinned (F32 offset 4 accepted,
    offset 3 refused) next to the F64 rows.
  - MINOR: the permute plan's workspace *limit* is explained where it is passed — `cutensorPermute`
    takes no workspace pointer, so the estimate only bounds what the plan may plan with.
  - Verified and left as-is by that review: cache-key separation, eviction/Drop lifetimes,
    identity-mode permute descriptors, stream ordering, aliasing and offset soundness, and that no
    previously-served layout changed provider.

- Second post-implementation review: `ollama-cloud/deepseek-v4.1-flash`, read-only, on
  `6e6a3a5c8`. Verdict `CHANGES`; all five findings fixed in this commit:
  - BLOCKER: `build_permutation_plan` and `build_plan` created descriptors, an operation and a
    plan preference and then returned through `?`, so every failure after the first handle left
    the driver holding handles no code could reach. Both builds now own their handles through
    `cutensor.rs::PlanBuild`, whose `Drop` destroys whatever exists unless the complete bundle
    was handed to the cache (`into_cached`).
  - BLOCKER: `insert_plan` dropped a freshly built plan when the eviction's `synchronize()`
    failed — the plan was never stored and nothing had run against it. It now destroys that plan
    and restores the evicted one, so neither is lost.
  - IMPORTANT: no test proved the cleanup ran. `a_refused_vendor_plan_leaks_no_native_handle`
    injects a refusal after 1, 2, 3 and 4 handles (`arm_plan_build_failure`) and asserts
    created == destroyed across the four stages; disarm-the-guard mutation makes it fail, so the
    test is red without the fix.
  - IMPORTANT (design): a routed view whose vendor route could not be planned had no provider at
    all, so a host without a usable cuTENSOR lost a layout the catalogue had always served. The
    route now reports `PermuteFailure::Unusable` for setup that submitted nothing and the
    catalogue's wide entry finishes the operation, counted by `vendor_permutation_fallbacks`; a
    failed *call* (`PermuteFailure::Submitted`) stays an error because the destination may be
    partly written. `docs/design/gpu-backend-design.md` states why this is provider selection
    rather than the silent native downgrade the vendor contract forbids.
  - MINOR: the plan-build failpoint and the handle counters are `#[cfg(test)]`, so a release
    build carries no failpoint and no counter.

## Files

- `crates/tenferro-gpu/src/cuda/cutensor.rs`: `cutensorCreatePermutation`/`cutensorPermute`
  in the `Vtable`, `Permutation`, `PermuteFailure`, `Cutensor::permute`,
  `build_permutation_plan`, a shared `insert_plan` (eviction and counters),
  `permutation_signature_of`, and the failure-atomic `PlanBuild` guard.
- `crates/tenferro-gpu/src/cuda/structural.rs`: `vendor_permutation_applies`,
  `VENDOR_PERMUTATION_MAX_RANK`, `CudaEngine::permute_into` (which reports whether the vendor
  served the view), and the two route arms in `materialize_storage` and `copy_read_into`.
- `crates/tenferro-gpu/src/cuda/engine.rs`: the `vendor_permutations` and
  `vendor_permutation_fallbacks` counters.
- `crates/tenferro-gpu/src/cuda/storage.rs`: `CudaStorage::allocation_id`, so the predicate
  can tell a shared allocation from two.
- `docs/design/gpu-backend-design.md`: the vendor route in the materialization section.
- `scripts/ci/cuda_test_partition.tsv`: the six new tests, audited into the lanes they run
  in (five device tests to `gpu`, the routing predicate to `host`). The partition is
  exhaustive by design — `gpu_test_partition.py` refuses an unclassified test — and it
  caught this omission on the first CI run, before any paid GPU execution started.
