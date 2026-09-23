# Issue #1875: mutable CubeCL write bindings invalidate the memoized address

## Summary

#1872 established that a queued CubeCL write must clear a buffer's memoized
device address. Its audit covered the `dispatch.rs` `launch_*` helpers whose
output parameter is `&TypedTensor`, and the accompanying source-contract test
keyed on exactly that name/signature pair.

`interop.rs` reaches the same hazard through a different shape: the in-place
scale bridge (`scale_typed_tensor_for_op`, `scale_typed_view`) binds an existing
destination via `dispatch::typed_view_mut_array_arg` and queues a scale kernel
without clearing the memoized address. The same binding helper is used by the
axpby path in `blas1.rs`, the written copy paths in `mod.rs`, and the strided
fill in `interop.rs`. `interop.rs` also contains no `launch_*` helper at all, so
the older scan could never see any of them.

## Decision

Clear the memoized device address where a mutable CubeCL binding is created, in
`dispatch::typed_tensor_mut_array_arg` and `dispatch::typed_view_mut_array_arg`,
using the same `cubecl_buffer` / `cubecl_view_mut_buffer` accessors the audited
launch helpers already use.

That places the invalidation at the single boundary every "bind an existing
buffer for a device write" path in the crate passes through, instead of
repeating it at each scale/copy/fill call site. The read-only binding helpers
and `prepared_*_mut_access` are deliberately unchanged: the former cannot queue
a write, and the latter prepares a raw provider access whose enqueue ordering is
already handled by the vendor enqueue path (`fill_zero_span`).

Clearing before the binding can fail is intentional. The cost of clearing is one
extra `get_resource` round trip on the next raw-FFI access; the cost of not
clearing is an out-of-order vendor call. The invariant is recorded at both call
sites so a later reader can re-verify it.

## Rejected alternatives

- Invalidate inside `scale_typed_tensor_for_op` / `scale_typed_view` only. That
  is the site the report names, but it leaves the same shape uncovered at the
  axpby, copy, and strided-fill bindings.
- Invalidate inside `prepare_cubecl_access` / the tensor-crate write preparation.
  Those layers are shared with the raw provider paths and with non-CubeCL
  backends; lowering the rule there would either miss CubeCL buffers or
  invalidate for reads.
- Extend the source-contract test by listing `interop.rs` helper names. The
  residual risk the #1868 worklog itself named is a write path with a different
  shape, so the new guard keys on the mutable CubeCL binding shape instead of on
  a file or a name prefix.

## Verification

- New source-contract test: every top-level function in `dispatch.rs` whose
  signature mentions `ArrayArg<CubeclCudaRuntime>` and whose body calls
  `prepare_device_write` must also call `invalidate_device_addr()`.
- New CUDA-gated behavioural test: memoize the address, run a real in-place
  scale through `scale_typed_tensor`, assert the memoized address is gone, and
  assert the scaled value on the host. The test is a direct check of the
  reported path rather than a reproduction of the interleaving, because the
  hazard needs an unflushed CubeCL-to-vendor transition that a test cannot force
  deterministically.
- The pre-existing `launch_*` contract test is unchanged and still passes.

Hardware execution of the new CUDA-gated test is outstanding on this machine
(no CUDA device); it is expected to run under the repository GPU gate.

## Residual risk

The invalidation is structural: it proves that no memoized address survives a
CubeCL write binding, not that every vendor call is ordered. Cross-stream
write-publication is the separate contract tracked by tensor4all/cubecl#16 and
is not addressed here.
