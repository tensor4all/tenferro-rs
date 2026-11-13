# CUDA elementwise: stop zero-filling the destination (#2529)

## Decisions
- An elementwise launch fully overwrites its destination, so `alloc_tensor`'s
  `cuMemsetD8Async` over the whole destination was a second, wasted write of the same bytes.
  `elementwise_binary` (including its `division` helper), `elementwise_unary`,
  `elementwise_extreme`, `remainder`, `clamp_read`, `compare_read`, `select_read` and `pow` now
  allocate with `alloc_uninit_tensor_bytes` and hand the launch an `UninitBuffer`, converting it
  with `UninitBuffer::assume_init` after `prove_launch` — the same rule reductions (#2406) and
  permutation materialization (#2528) already follow.
- `Dest<'a>` (`Init(&Buffer)` / `Uninit(&UninitBuffer)`) is the destination parameter of the six
  dispatcher methods those entries use (`arithmetic`, `binary`, `unary`, `clamp_buffers`,
  `compare`, `select`), so a call site cannot pair an uninitialised buffer with the initialized
  argument variant, or the reverse. It is crate-internal; no public API, ABI or pinned revision
  changes.
- A caller-owned destination keeps `Dest::Init`: `elementwise_into_compact`, `scale_in_place`, and
  the plan-level tests in `src/cuda/dispatch/tests.rs`. `pow`'s checked-integer route keeps its
  status download before the conversion, so the reject case never converts the output.
- The empty-domain case stays correct without a fill: `Dispatcher::run` admits no launch for a
  zero grid, and an empty destination has no element to leave uninitialised.

## Verification conclusions and constraints
- A100 80GB PCIe, driver 580.178.04, CUDA 12.6, f64, n=1048576, one isolated call per sample,
  3 warm-ups and 15 timed runs, 5 interleaved processes per arm, `taskset -c 8-15`, GPU compute
  app count 0 before every process:

  | op | before (µs) | after (µs) | speed-up |
  | --- | --- | --- | --- |
  | neg | 23.47 | 18.11 | 1.30x |
  | add | 24.91 | 19.76 | 1.26x |
  | exp | 24.19 | 18.91 | 1.28x |
  | tanh | 25.49 | 20.19 | 1.26x |
  | clamp | 35.54 | 32.12 | 1.11x |

  f32 gains 1.13–1.22x at the same size. Every cell satisfies the paired acceptance rule
  `after max < before min` over the five processes, so the difference is larger than the spread.
- The saving is the removed pass, not a faster kernel: `nsys` on the candidate shows no
  `cuMemsetD8Async` and no `[CUDA memset]` at all, and the kernel medians are unchanged
  (`t4a_cuda_unary` 7.1 µs, `t4a_cuda_add` 8.2 µs, `t4a_cuda_compare` 7.2 µs, f64 at 8 MiB).
  Before the change the same profile carried `cuMemsetD8Async` at 3.4 µs of host time plus a
  5.6 µs `[CUDA memset]` per call.
- The fixed per-call cost at n=1024 barely moves (f64 15.6→14.3 µs), which is the expected shape:
  a fill of 8 KiB is nearly free, a fill of 8 MiB is not. Against PyTorch 2.12's ~11.5 µs fixed
  cost the isolated-call gap narrows from ~1.4x to ~1.2x rather than closing.
- Tests: `an_elementwise_launch_does_not_zero_its_destination` reads the runtime's own
  `Stats.retirement.zero_fills` counter, so it fails on the unfixed build and passes here;
  `the_zero_fill_counter_tracks_a_zeroed_allocation` proves that counter is live (the same
  assertion on `alloc_tensor` moves it). `an_elementwise_result_holds_no_stale_bytes` uploads its
  operands first and then drops a same-sized NaN-filled buffer, so the `neg` destination is that
  freed block and a short write leaves the pattern visible (the later calls in it assert the same
  exact values without claiming a reuse). `an_elementwise_launch_covers_every_length` covers `len` 0, 1, 7 and 1_000_003 over the f64
  entries and a `DType::Bool` comparison. `a_destination_is_passed_as_the_storage_it_holds` pins
  the argument variant of each `Dest`.
- Not covered here: #2528's `reverse_*` permutation rows (the merge-axes and destination-reorder
  work in #2551/#2581 already moved the cyclic, rotation and `tn_light` rows by 3.9–11.4x), and the
  remaining isolated-call gap against PyTorch, which is now mostly the per-call fixed cost rather
  than kernel geometry.
