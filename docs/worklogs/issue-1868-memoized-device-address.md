# Invalidating the memoized device address on a queued write (issue #1868)

Reported by [@Ryo-wtnb11](https://github.com/Ryo-wtnb11) from reading the
source, with no hardware reproduction. The investigation below was run on an
NVIDIA A100 80GB PCIe (driver 580.126.09, CUDA 12.6) during intake.

## What the hazard actually is

`typed_device_ptr` returns a memoized device address when the buffer was
created on the current stream, skipping `client().get_resource(..)`. The
existing comment justified the skip by CubeCL's cross-stream alignment pass
ignoring same-stream bindings. That is true and it is not the whole contract:
`get_resource` is `submit_blocking`, and that server round trip is also what
pushes queued CubeCL kernels onto the CUstream.

Confirmed structurally in both codebases:

- CubeCL kernel launch is `device.submit` (asynchronous, `SEND_NO_FLUSH`);
  `get_resource` is `device.submit_blocking`.
- Nothing on the vendor operand path flushes. `do_empty` resolves its layout
  host-side and submits asynchronously, `raw_cuda_stream` early-returns from a
  `OnceLock` after the first call, and `cross_stream_handles` is a pure filter.
  `blas1.rs` contains no `flush_cubecl` at all.

## What is and is not exposed

- **`dot_general` is safe, structurally.** Every call allocates a fresh output,
  and that output's `typed_device_ptr` has no memoized address, so it takes the
  `get_resource` path before the contraction is issued. A reproduction attempt
  found 0 violations in 200 trials at 256x256 and 20 trials at 2048x2048 (a
  32 MB fill), with the operand fast path asserted taken. The probe was not
  committed: a test that passes for an incidental reason is worse than none.
- **`axpby` is exposed.** Both operands are owned tensors and neither
  allocates, so once the addresses are memoized the call reaches the CUstream
  with no server round trip anywhere on the path.

## Decision

Invalidate the memoization when a CubeCL kernel is queued to write the buffer,
rather than flushing before vendor calls. The next raw-FFI access then resolves
through `get_resource`, which is the barrier that was always there.

- Rejected: `flush_cubecl` before the vendor call. Measured cost below: 4.5x to
  17x on the `axpby` enqueue path, which is exactly what the fast path exists
  to avoid, and it grows with the work already queued.
- Rejected: invalidating inside `typed_tensor_array_arg`. It also builds input
  arguments, so it would invalidate read-only operands and remove the fast path
  entirely.
- Rejected: documenting a caller-side flush obligation. It leaves the trap in
  place for every future call site.

`device_addr` moves from `OnceLock<u64>` to `AtomicU64`, with 0 meaning "not
memoized" because no CUDA allocation lives at the null address; `OnceLock`
cannot be cleared through a shared reference.

## Verification conclusions and constraints

- Measured on the A100 (median of 200 calls, release):

  | case | baseline | invalidate | flush per call |
  |---|---|---|---|
  | `axpby` len 1024 | 3.61 us | 3.23 us | 16.39 us |
  | `axpby` len 65536 | 3.17 us | 3.17 us | 16.04 us |
  | `axpby` len 1048576 | 3.07 us | 3.59 us | 52.09 us |
  | `dot_general` n=32 | 20.18 us | 20.34 us | - |
  | `dot_general` n=128 | 20.78 us | 21.61 us | - |
  | `dot_general` n=512 | 22.25 us | 21.32 us | - |

  The invalidation column is baseline within noise, because these loops issue
  no CubeCL write between vendor calls and so never trigger it. The cost is one
  `get_resource` after each write, on a path that already paid it.

- A hardware test asserts the memoization is dropped by a queued write and
  restored by the next resolve. A source-contract test asserts every
  `launch_*` helper taking `output: &TypedTensor` invalidates; removing one
  invalidation makes it fail and name the helper, which was checked.

- Constraint: the fix is only as complete as that signature audit. Four helpers
  currently write an existing tensor, and all four invalidate. A path that
  writes an owned buffer through some other shape would still be missed, which
  is why the rule is pinned on the source rather than on one behaviour.

- Constraint: the original hazard was never reproduced, so this change cannot
  be validated by observing a behaviour change. What is verified is structural:
  a queued write leaves no memoized address behind.
