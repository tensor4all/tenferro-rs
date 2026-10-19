# Deferred prepared completion in a caller-held session (#2087, P3-U8)

## Decisions

- Added the public `Runtime::begin_prepared_deferred` guard. Its
  `run_prepared_deferred` executes the prepared graph borrowed when the guard was
  opened, enqueues a run, and returns its index without the event-domain
  retirement barrier; `finish_prepared` drains all runs since the previous
  finish and returns their outputs in enqueue order. `Drop` retires pending work
  best-effort and cannot report the failure.
- `Runtime::run_prepared_in` is now the same deferred path with an immediate
  finish, so there is one execution implementation. Its externally visible
  contract is unchanged: device work is retired before it returns.
- The guard borrows the session and prepared graph and leases the executor state
  for its lifetime. It also owns each run's outputs until finish and ties input
  borrows to the guard lifetime, so pending device work cannot outlive an input
  or an output allocation. A device-side retirement failure is returned from
  `finish_prepared`. A host-side validation failure returns from the deferred
  call only after retiring work already enqueued in the batch.
- Deferred state is one reusable `ScheduledEventDomains` per batch. A coalesced
  single-domain schedule therefore records no per-node events across the batch,
  and one finish takes one dispatch fence and one stream barrier.
- Left out of scope: a queue-full policy for an unbounded detached submit loop,
  and completion-aware reuse outside the caller-held guard. Those remain the
  open stage-2 items in `docs/design/held-cuda-session-1945-u3.md` §8.

## Verification conclusions and constraints

- GPU tests on an A100 80GB PCIe, CUDA 12.6, debug test profile: eight deferred
  runs of the eight-matmul prepared chain followed by one finish match the
  synchronous run, record zero scheduler events, one fence and one retirement.
  A CubeCL -> vendor -> CubeCL chain with four distinct inputs returns each
  package in enqueue order and matches its own synchronous reference. Dropping
  an unfinished guard retires once and leaves the session usable.
- CPU integration tests cover three deferred identity runs for every scalar
  dtype (f32, f64, i32, i64, bool, c32, c64). A counting event-domain driver
  proves a rejected wrong-shaped input retires the earlier deferred run before
  returning, and the following finish still returns that run's output. Another
  runtime unit test has `finish_prepared` return an injected device-side
  retirement error while the deferred call succeeds.
- A100 release timing from `host_path_phases`, one-op batched-matmul vendor graph,
  16 x 32x32 f64, 400 samples. Three runs on the shared host: synchronous
  `run_prepared_in` median 36.6, 41.6 and 44.2 us/call; deferred in batches of 64
  with one finish 24.8, 28.4 and 30.9 us/call. The per-call whole-arm delta is
  therefore about -12 to -13 us. This is a whole-arm number, not a phase
  attribution.
- A local mirror of the external QR JVP 256 trace graph (`grad_sum_qr` JVP,
  n=256, matrix seed 26) through `Runtime::run_compiled` was built against the
  merge base `9aff3efc2` and this branch. Nine interleaved paired runs on the
  same A100 gave median-of-run-medians 1.952 ms on the base and 1.966 ms on this
  branch (+0.7%, inside the 1.918-2.033 ms spread), so no regression was
  detected on that case.
