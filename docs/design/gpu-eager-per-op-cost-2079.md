# Decode-sized eager op host cost (#2079): remove the per-op bookkeeping

Status: design. Implementation follows on the `develop` branch; #2112 item 7.

## The defect

For decode-sized tensors the host cost of one eager CUDA op is several times its
kernel time. Measured on the A100 (release, one thread, f32 `add` on a 1024x8
tensor, warm):

| arm | per op |
| --- | --- |
| raw-launch floor (same kernel, reused output) | **5.36 us** |
| `eager-session-streaming` (result dropped per op) | **9.11 us** |
| `eager-session` (harness retains every result) | 9.94 us |

The target is per-op within 1-2 us of the floor, i.e. about 6.4-7.4 us.

**Constraint, measured** (#2137 I2): one device-thread round trip - `Client::check`
with a single handle - costs 8.19 us, more than the whole target. Nothing on this
path may add a synchronisation point; the saving has to come from host-side work
that is done per op and buys nothing.

The `eager-session` arm retains every iteration's output in a `Vec`, so its
allocation reuse cannot help; the acceptance is the **streaming** arm, and the pool
must not be "fixed" to satisfy the retained number.

## What is done per op (inventory)

One binary `Add`, untracked, f64/f32, contiguous equal shapes:

`EagerSession::run_binary` (`eager.rs:2724`) -> `broadcast_binary_in_session`
(`eager_ops.rs:34`) -> `EagerTensor::nary_op_in_session` (`eager_ops.rs:225`) ->
`exec_standard_op_on_tensor_reads_in_session` (`eager_exec.rs:386`) ->
`CudaExecSession::add_read` -> `ops::add_read` (`ops.rs:13`) ->
`binary_read_native` (`cubecl/mod.rs:4703`) -> `launch_binary_views`
(`dispatch.rs:1136`) -> CubeCL `launch_unchecked`, then the result record
(`eager.rs:5470`).

Steady state: **22-24 heap allocations, 15-25 `Arc` clones and 2+ global `Mutex`
acquisitions per op.** The ones that buy nothing:

| item | site | cost |
| --- | --- | --- |
| `broadcast_shape` `Vec` even when the shapes are equal | `eager_ops.rs:42` | 1 alloc + a scan |
| `EagerTensor::clone()` for both operands in the equal-shape case | `eager_ops.rs:74` | 2 clones, each bumping 5-8 `Arc`s |
| `PromotedTensorRead::Borrowed(Box::new(input))` per operand, even when no promotion happens | `eager_exec.rs:165` | 2 `Box`es |
| `collect_input_reads` `Vec` (arity is static) | `eager_ops.rs:257` | 1 alloc |
| `read_views(&[slot])` `Vec` per pooled operand read | `eager.rs:936` -> `storage/group.rs:613` | 1 alloc per operand; `read_view` (`group.rs:631`) is the single-slot path and is unused here |
| one-element result `Vec<Tensor>` | `eager_exec.rs:399` | 1 alloc |
| `cubecl_shape_and_strides` per binding | `dispatch.rs:235` | 2 `Vec` per binding, 3 bindings per op = 6 allocs |

Genuine, and stated so they are not mistaken for candidates:

- `Box<dyn PreparedDeviceAccess>` from `prepare_device_read` (`types.rs:1862`,
  `3205`, `9580` -> `dispatch.rs:98`) crosses the tensor-provider API; removing it
  means changing that contract, not this path.
- One `handle.clone()` per binding retains the allocation (`Arc` increment).
- `client.empty` on a pool miss and the `take_output_handle`/`return_output_handle`
  mutex pair (`tenferro-gpu/src/lib.rs:194`, `:210`) are the pool's own cost; the
  device handle is already reused.
- The new value's identity - its record `Arc`s and the shape `Box<[usize]>`
  (`eager.rs:919-930`) - is what the value *is*, in the streaming arm too.

## Plan, in order, each step measured

1. **Delete the allocations that buy nothing** (no cache, no API change):
   *landed*: the equal-shape path in `broadcast_binary_in_session` now returns
   borrowed operands, and a pooled value read uses the single-slot `read_view`.
   Measured with the in-repo harness below: **10.15 us/op -> 8.96 us/op**.
   the equal-shape fast path in `broadcast_binary_in_session` (skip
   `broadcast_shape`, borrow instead of cloning), `PromotedTensorRead` borrowing
   instead of boxing, the single-slot `read_view` instead of `read_views(&[slot])`,
   a fixed-arity input-read array, and the one-element result array.
2. **Cache/Build the binding metadata without a `Vec`** (`cubecl_shape_and_strides`,
   `dispatch.rs:235`): *tried and measured worse, not landed.* CubeCL's
   `Shape`/`Strides` are `SmallVec<[usize; 5]>`-backed, so building them directly
   (instead of a `Vec` each that is converted at the binding) removes two
   allocations and two conversions per binding - six allocations per eager op -
   but it also makes `CubeclPreparedAccess`, which is boxed once per binding, about
   100 bytes larger and copies the metadata inline. Measured on the A100 with the
   harness below, five samples each:

   | metadata | streaming samples | best |
   | --- | --- | --- |
   | `Vec` per binding (kept) | 8.35, 8.33, 8.46, 8.88, 8.94 | **8.33 us/op** |
   | `SmallVec` built inline | 9.36, 9.28, 9.45, 9.51, 9.07 | 9.07 us/op |

   The allocation count is lower and the host cost is 8% higher; the measurement
   wins, so the `Vec` stays. An allocation-count argument is not a host-cost
   argument. A shape-keyed *cache* would have to return the `Shape`/`Strides`
   themselves to avoid the conversion, and the same struct-size effect would
   apply, so it is not obviously better than the status quo.
3. **Re-measure** after each step against the floor, on the streaming arm. Step 1
   is the only step that measured better; step 2 measured worse and was dropped.

Explicitly **not** in this design:

- Pooling the output's host descriptors (shape `Vec`, `StorageBuffer` `Box`, record
  allocation). Each returned value must own stable descriptors, so this needs a
  take/return discipline of its own; it is worth its own design if step 1-2 leave
  a gap.
- The `Box<dyn PreparedDeviceAccess>` provider-contract change.

## Measurement harness

`eager_op_f64_decode_add_host_cost_benchmark`
(`crates/tenferro-ad/tests/integration/gpu_ad_tests.rs`, lane `gpu` in
`scripts/ci/cuda_test_partition.tsv`: the partition reserves its single
`benchmark` entry for a reviewed exclusion) prints three arms for the decode shape
(1024x8, one session around the loop) and asserts nothing - the acceptance is a
reading taken with the same harness before and after a change:

| arm | before step 1 | after step 1 |
| --- | --- | --- |
| streaming (result dropped per op) | 10.15 us/op | **8.96 us/op** |
| retained (all results held) | 15.78 us/op | 16.17 us/op |
| `alloc_output` + drop alone | 0.73 us/op | - |

The retained arm is unchanged by design: holding every result defeats the pool, and
#2079 says the pool must not be "fixed" to satisfy that number. Re-read after
merging `main` (which carries #2149's compact-layout cache, the same per-op
neighbourhood) the streaming samples are 8.65-8.79 us, best **8.65 us/op** - the
same reading within this harness's noise, so #2149 neither helped nor hurt it. The harness is a
*different* harness from the benchmark suite's (one process, one session around the
loop, f64), so the absolute values are not the recorded 9.11 / 5.36 pair - what is
comparable is a before/after delta measured here, and the official arms remain the
authority.

## Acceptance

- The streaming arm of the decode-sized `add` measurement, on the A100, at 1-2 us
  above the same run's launch-only floor. **Not yet met**: step 1 moved it from
  10.15 to 8.96 us/op, and the remaining work is step 2 (the binding-metadata
  cache) plus the descriptor-pooling question this record leaves open.
- No regression in the device suite (218 tests) and no new per-op synchronisation:
  `docs/design/gpu-correctness-invariants-2137.md` I2's measured round-trip cost is
  the reason.
- The official arms (`eager_op_f32_add_decode_*` in `gpu/perf_issues`) live in
  `tensor4all/tenferro-benchmark`; the in-repo measurement reproduces the same
  shape and is what this design is accepted against.

## Why the trimming stops here: the floor is the wall

The numbers say more than "there is a little left". For the shape this issue
measures (8x1024 f32 `add`), the kernel runs in **1.2 us** and the raw-launch floor
is **5.36 us**, so the host pays at least 4.5x the kernel time for one op even
after every bookkeeping item in the inventory above is removed. Step 1 got the
streaming arm from 10.15 to 8.96 us; the floor is the wall it approaches, and the
wall is above the kernel.

That is the real finding: **an isolated decode-sized op has no GPU advantage to
win.** At the floor the GPU is idle for roughly four fifths of the op's wall time,
so shaving another 0.3 us of host bookkeeping moves a percentage that the launch
overhead already decided. The measurable local levers are exhausted:

- what remains in tenferro is spread across many small allocations and `Arc`
  clones (see the inventory), and the one candidate that removes six of them
  measured 8% *worse*;
- the larger part of the floor is CubeCL's own `launch_unchecked` argument
  marshalling and the launch itself, which this repository cannot change;
- one device-thread round trip (8.19 us, #2137 I2) is larger than the whole target,
  so nothing on the path may add a synchronisation point.

The remedy for the decode case therefore belongs to a different level, not to this
one:

1. **Amortise the launch** - capture/replay, batched submission, or fusion across
   ops, so one launch covers many decode steps. Sequence-level execution is owned
   by #1938 section 5 (and #1945); this issue's T1 scope explicitly excludes it.
2. **Do not dispatch such ops to the GPU at all** - a dispatch-policy decision for
   a tensor small enough that the CPU wins (a 1024x8 `add` is 8192 elements). That
   is a new question with its own trade-offs (placement of the operands, transfers,
   the user's expectation of where their data lives) and is not implied by this
   record.

Until one of those is decided, `#2079`'s acceptance ("per-op within 1-2 us of the
launch floor") is the wrong target: it asks for 0.x us of work whose ceiling is set
by a floor 4.5x the kernel time. #2112 item 7 records the replacement.

## What this record does not claim

- The split of the ~3.7 us between tenferro's host bookkeeping and CubeCL's own
  `launch_unchecked` argument marshalling is not measured; a profiler would be
  needed, and no saving is claimed for the latter.
- Whether a per-op session (rather than one session around a loop) is the harness's
  shape is unknown here; if it is, `lock_backend` and the entry guard belong in the
  same inventory.
