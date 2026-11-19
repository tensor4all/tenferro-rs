# GPU: `vdot_read` is one cuBLAS conjugate dot (#2624)

## Decision

`CudaEngine::vdot_read` no longer composes `conj_read` -> `mul_read` ->
`reduce_axes`. It reads both operands' compact spans and issues one BLAS-1 call
on the engine's stream: `cublasSdot`/`cublasDdot` for a real dtype and
`cublasCdotc`/`cublasZdotc` for a complex one, which conjugates the left operand
by definition, so no conjugation copy exists to skip or to keep. This is the
contract #1733 established for the CubeCL engine and #2208 dropped with it,
restored on the engine that remains.

What that removes from a standalone call on two compact f64 vectors: a full
conjugation copy that is the identity on a real dtype, a full product tensor, a
separate reduction (and its partial-vector split on long shapes), and two of the
three launches. The completion wait is no longer part of the difference: #2666
stopped every eager CUDA launch waiting, so the composed path does not wait per
launch either (see "Measured against current `main`" below).

Rejected alternatives:

- **Keep the composition and only skip the real-dtype conjugation copy.** The
  product tensor and the reduction stay, so two of the three costs remain.
- **A hand-written fused kernel that loads both operands and writes one scalar.**
  It competes with cuBLAS on the one shape where cuBLAS is already bandwidth
  bound, and it would need its own NVRTC source, artifact-cache key and a
  reduction tree. The vendor call is the same number of launches with no new
  kernel to maintain.
- **Skipping the destination's zero fill** (`alloc_uninit` + `assume_init`, the
  pattern `reduce_axes` uses for #2406) would remove one `cuMemsetD8Async` from
  the call, but the raw scope can only register an initialised `Buffer`
  (`RawResource::Buffer`); passing an uninitialised one needs a
  `RawResource::BufferUninit` in `t4a-cuda-runtime`, so the fill stays. Revisit
  if the engine's per-call overhead is attacked in general.
- **`norm_squared_read` and `axpby_read_into_accum`** still compose; they were
  not part of this request and their compositions are correct.

## Measurements

Host `primerose`: NVIDIA A100 80GB PCIe, driver 580.178.04, CUDA toolkit 12.6,
`nvidia-smi` reporting an idle device between runs (0 %), `torch 2.12.0+cu130`
available in the benchmark venv. Rust `1.100.0`, release profile.

The workload is `gpu/reuse_contraction/vdot_read_f64_n262144` (and its `c64`
sibling): two compact rank-1 operands of 262144 elements, one `vdot_read` per
iteration, steady state, host API plus one device synchronisation per call.

The numbers below come from a **diagnostic** (`vdot-probe`, kept outside the
repository) that issues the same call with the same sampling protocol as the
campaign's `benchmark_gpu_public_api`: one untimed priming call, 3 warmups, a
batch grown until it lasts 2 ms, then 15 samples of `batch_ms / ops_per_sample`,
outputs retained until the timer stops, median reported. Arm A is `aebe8cb3`
(the merge base) and arm B is `6081fea5`; both arms are the same binary source
built against each revision, run back to back in one window.

| case | A `aebe8cb3` (composition) | B `6081fea5` (one cuBLAS dot) | speedup |
| --- | --- | --- | --- |
| `vdot_read_f64_n262144` | 0.0629 / 0.0655 / 0.0661 / 0.0651 ms | 0.0214 / 0.0211 / 0.0208 / 0.0215 ms | 3.06x |
| `vdot_read_c64_n262144` | 0.0647 / 0.0662 / 0.0685 / 0.0688 ms | 0.0223 / 0.0220 / 0.0217 / 0.0231 ms | 3.03x |

Four interleaved A/B repetitions each; the table lists them in run order, and
the arm medians are 0.0652 ms -> 0.0213 ms (f64) and 0.0673 ms -> 0.0222 ms
(c64). Against `torch.vdot`'s 0.01017 ms on the same case, the fused path is
2.1x behind instead of 24.8x.

Two caveats the reader must have:

- The issue's baseline of 0.2518 ms is from pin `9a850b06`. On `main` at the
  time of that measurement the composition already cost 0.0652 ms, 3.9x faster
  than that baseline, so most of the 24.8x gap was closed by earlier reduction
  work (#2406, #2474) and this change closes the remainder of the *launch* gap,
  not of a 0.25 ms cost.
- The machine was not quiet: two other agents were building and running GPU
  tests throughout, and `/proc/loadavg` sat around 26-28 (64 cores) for the
  measurement window. No benchmark process was running and the device was idle
  at 0-16 % at the start and end of every pair, but the absolute values include
  that host contention; both arms share it, which is why the pairs are
  interleaved rather than compared across sessions. The one outlier sample
  (arm A, first run: IQR upper 1.91 ms) is contention, not the operation.
- **Those `aebe8cb3` numbers no longer describe `main`.** #2666 removed the
  per-launch completion wait that both arms paid there, which was most of the
  absolute cost on both sides. The table above is kept as the record of what
  was measured before that landed; the ratio that describes this change on
  current `main` is the one below.

## Measured against current `main` (after the rebase onto `51a3c2ee`)

The same diagnostic, re-run because #2666 changed the cost of *both* arms: with
no launch waiting, the composed path lost three waits and the fused path lost
one. Arm A is `origin/main` `51a3c2ee` and arm B is this branch rebased on it;
three interleaved pairs, one measurement at a time.

The figure is an **amortised per-op cost inside a batch**, not a synchronised
per-call latency: ops are enqueued back to back and the batch synchronises once
at its end, so the batch width differs between the arms (128 ops for A, 256 for
B, fixed by the 2 ms target) and a per-call host barrier is not part of either
number. That was already true of the earlier `aebe8cb3` run, but there the
engine itself took a barrier per launch, which is why those absolute numbers
were so much larger.

Discipline: the probe pinned to CPUs 0-7, one L3 domain
(`/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list`), the observer shell
on CPUs 8-15. Before every one of the six runs, after an 8 s drain:
`nvidia-smi` reported 0 % utilisation, 14 MiB used and an empty compute-process
list, and `/proc/loadavg` was 0.63-1.13. Raw log, outside the repository:
`~/work/tf-2624-plan/raw-ab-rebase.txt`.

| case | A `51a3c2ee` (composition) | B rebase head (one cuBLAS dot) | ratio |
| --- | --- | --- | --- |
| `vdot_read_f64_n262144` | 0.024242 / 0.024325 / 0.024820 ms | 0.011276 / 0.011298 / 0.011393 ms | 2.15x |
| `vdot_read_c64_n262144` | 0.026257 / 0.026194 / 0.026679 ms | 0.012257 / 0.012317 / 0.012274 ms | 2.14x |

The table lists the three repetitions in run order; the arm medians are
0.024325 ms -> 0.011298 ms (f64) and 0.026257 ms -> 0.012274 ms (c64). Two
earlier windows on this host, before the final worklog edit, measured the same
pair at 2.19x/2.18x and 2.20x/2.19x, so the ratio is stable at 2.1-2.2x.

So the change still owns a 2.1x ratio on current `main`; it is smaller than the
3.0x recorded before #2666 only because that change took the waits out of both
arms, and it is no longer mostly a wait story on either side. The 0.01017 ms
`torch.vdot` figure recorded for this case earlier is from a different session
and protocol and is therefore indicative only, but by that order the fused path
is now within roughly 10 % of it, against a 24.8x gap at the issue's baseline.

## Rebase onto `51a3c2ee`

One textual conflict and two semantic ones, all in the tree the rebase touched.

**Textual:** `docs/design/gpu-backend-design.md`, a directory listing.
`main` had renamed the `cutensor.rs` line to "cuTENSOR contraction and
permutation integration" after this branch forked, and the branch had rewritten
both that line and the `vendor.rs` line. Resolved by keeping `main`'s
`cutensor.rs` wording with the branch's `vendor.rs` wording: neither side loses
information and no claim changes.

**Semantic 1 — `structural.rs`, a symbol `main` deleted.** #2666 deleted
`CudaEngine::prove_launch` (and `launch_completion_syncs`) outright; git's
auto-merge kept the branch's `self.prove_launch(OP)?;` in `vdot_read`, which no
longer compiles. Resolved to `self.release_completed_staging();`, which is what
`main` does after the analogous vendor submission, the cuTENSOR `contract` in
the same file. The fused dot is a stream-ordered submission whose result is
device-resident and observed at the caller's own barrier, exactly like every
other eager CUDA operation on `main`; the cuBLAS handle is bound to the engine's
stream at creation, so no different completion call is needed.

**Semantic 2 — a test whose premise #2666 removed, replaced.** The branch's
`cuda_vdot_is_one_completion_wait_over_compact_operands` counted
`CudaEngine::launch_completion_syncs`, which #2666 deleted, and asserted the
fused path takes one wait where the composed one takes three. On current `main`
both take none, so that assertion cannot be repaired without re-adding the
counter #2666 deliberately removed as a policy change. It is replaced by
`cuda_vdot_over_compact_operands_admits_no_catalogue_launch`, which pins the
property the change actually owns — **launches**, not waits — through the
counter `main` kept: `runtime().stats().retirement.launches`, the same hook
`tenferro-ad`'s fused-softplus test uses. A compact `vdot_read` admits zero
catalogue launches (cuBLAS is not one) and zero stream waits; the test also
counts `conj_read` + `mul_read` over the same complex operands, which admit 2,
and the reduction the composition ended with is a third, so the assertion is
not vacuous and a re-composition fails it. Its
`scripts/ci/cuda_test_partition.tsv` row was renamed with it
(`gpu_test_partition.py`'s `validate_inventory` compares the partition to the
nextest inventory as an exact set and fails closed, so the pair moves
atomically). The test count is therefore unchanged at fifteen.

**Also on `main`, and blocking the gate:** `cuda_test_partition.tsv` on
`51a3c2ee` is missing two `tenferro-ad::integration` tests that landed with
#2656 (`eager_runtime_api::eager_filled_constants_hold_the_value_and_take_no_gradient`
and `fallible_api::traced_filled_constants_differentiate_as_constants_and_reject_seed_roles`,
`crates/tenferro-ad/tests/integration/{eager_runtime_api.rs:51,fallible_api.rs:233}`).
`gpu_test_partition.py`'s `validate_inventory` compares the partition to the
nextest archive as an exact set, so **every** PR's local GPU gate fails at the
CUDA host lane on `51a3c2ee` until they are classified; the first gate run at
this head failed exactly there. Both bodies run on `CpuEngine::serial()` /
`cpu_runtime()` and belong to the `host` lane, which is what the rows added here
say. This is a main-side inventory gap, not a rebase artifact; it is fixed here
because the gate cannot run otherwise.

## Verification

All of the following are at the rebased head, on `primerose`.

- `cargo test -p tenferro-gpu --features cuda --test blas1_cuda -- --ignored
  --test-threads=1` — 15 passed, 0 failed (1.12 s): the four real and complex
  widths on compact operands, F64 transposed stencil, offset view and
  owned-against-strided operands against the CPU oracle, rank-0 operands, the
  empty-input identity, and the launch/stream-wait count above.
- `cargo test -p tenferro-gpu --features cuda --test seam_vendor -- --ignored` —
  5 passed, 0 failed (0.53 s).
- `cargo test -p tenferro-gpu --features cuda --lib vendor_narrowing` — 5
  passed; the new case refuses a dot longer than `i32::MAX` elements by name
  and value.
- `python3 -m unittest scripts.ci.tests.test_gpu_test_partition` — 11 tests, OK:
  the partition still matches the archive inventory after the row rename and the
  two #2656 rows.
- `cargo clippy --workspace --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc` (the command the
  `clippy` CI job runs) — clean, and
  `cargo clippy -p tenferro-gpu --features cuda --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc` — clean;
  `cargo fmt --all --check` — clean. The workspace form is what caught two unused
  `TensorElementwise` imports in the new test (`conj_read`/`mul_read` are `Engine`
  methods, so no trait import is needed); the crate-scoped form alone had been run
  before that test existed.
- `python3 scripts/ci/local_gpu_gate.py run --jobs 12` — the repository's own GPU
  lane at the new head (this PR carries `gpu-validated-locally`, so the local
  record is what publishes the required `CI GPU gate`): passed, every step `ok` —
  CUDA archive, PJRT archive, CUDA tutorial build, CUDA host lane 1442 passed /
  391 skipped, PJRT host lane 54 passed / 6 skipped, CUDA device lane 389 passed
  / 1120 skipped, CUDA tutorial, PJRT device lane 3 passed / 37 skipped. The CUDA
  device lane is the one that runs `blas1_cuda`; its 389 includes the fifteen
  above.
- The CUDA environment for all of the above:
  `CUDA_PATH=/usr/local/cuda-12.6`,
  `LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12`,
  `TENFERRO_REQUIRE_GPU=1`, and for the gate
  `TENFERRO_LOCAL_PJRT_ROOT` pointing at the host's JAX/PJRT site-packages for
  the PJRT device lane.

## Review record

- Plan review, before implementing: `ollama-cloud/glm-5.3-flash:high`,
  **VERDICT: CHANGES** — adopted: a typed `Unavailable` when cuBLAS cannot be
  loaded (cudarc's `culib()` panics otherwise), the `i32`-length narrowing with
  a test and a CHANGELOG note, and the stale anchors. Declined: dropping
  `DotRequest` (the enum is what keeps the four entry points one call site).
- Diff review, after implementing: `openai/gpt-6.1-sol`, **VERDICT: CHANGES** —
  adopted: `Cublas::dot` is `pub(crate)` (it trusts its spans), a handle whose
  pointer mode cannot be proven is recreated instead of reused by the GEMM
  entries, the design note no longer claims `tenferro-gpu` reads
  `TENFERRO_CUBLAS_PATH` (it loads through cudarc's soname search), and the
  wait-count test is described as a completion-wait regression test rather than
  a launch count. No finding was left unfixed.
