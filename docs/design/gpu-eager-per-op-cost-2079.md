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
2. **Cache the binding metadata** (`cubecl_shape_and_strides`): keyed on the
   shape's extents, held in the runtime, returning a shared
   `Arc<[usize]>`/inline array so a recurring shape costs no allocation. This is
   the "plan/analysis cache by shape/stride/dtype" half of the issue, in its
   smallest useful form: the elementwise path has *no* runtime plan to cache
   (the kernel is a static dtype match, `cubecl/mod.rs:4703`), so the metadata is
   what recurs.
3. **Re-measure** after each step against the floor, on the streaming arm.

Explicitly **not** in this design:

- Pooling the output's host descriptors (shape `Vec`, `StorageBuffer` `Box`, record
  allocation). Each returned value must own stable descriptors, so this needs a
  take/return discipline of its own; it is worth its own design if step 1-2 leave
  a gap.
- The `Box<dyn PreparedDeviceAccess>` provider-contract change.

## Measurement harness

`eager_op_f64_decode_add_host_cost_benchmark`
(`crates/tenferro-ad/tests/integration/gpu_ad_tests.rs`, lane `benchmark` in
`scripts/ci/cuda_test_partition.tsv`) prints three arms for the decode shape
(1024x8, one session around the loop) and asserts nothing - the acceptance is a
reading taken with the same harness before and after a change:

| arm | before step 1 | after step 1 |
| --- | --- | --- |
| streaming (result dropped per op) | 10.15 us/op | **8.96 us/op** |
| retained (all results held) | 15.78 us/op | 16.17 us/op |
| `alloc_output` + drop alone | 0.73 us/op | - |

The retained arm is unchanged by design: holding every result defeats the pool, and
#2079 says the pool must not be "fixed" to satisfy that number. The harness is a
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

## What this record does not claim

- The split of the ~3.7 us between tenferro's host bookkeeping and CubeCL's own
  `launch_unchecked` argument marshalling is not measured; a profiler would be
  needed, and no saving is claimed for the latter.
- Whether a per-op session (rather than one session around a loop) is the harness's
  shape is unknown here; if it is, `lock_backend` and the entry guard belong in the
  same inventory.
