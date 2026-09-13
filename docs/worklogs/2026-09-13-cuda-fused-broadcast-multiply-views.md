# Run the fused broadcast multiply from borrowed views

## Decisions

- Accept a borrowed view in `CudaBackend::execute_broadcast_multiply` instead of
  returning `Ok(None)`. The eager einsum path prepares operands as views over
  already allocated device storage, so the previous owned-tensor-only guard sent
  every broadcast-multiply step to the caller's materializing fallback; the
  traced runtime hands owned operands and took the fused kernel, which is why
  the same workload was ~2.7x slower on the eager backend.
- Consume the view directly rather than materializing it. A compact zero-offset
  column-major view satisfies the kernel's existing contiguity contract (the
  same one `dispatch::typed_view_binding` enforces), so a copy would only
  reproduce the fallback this change removes.
- Keep `Ok(None)` for strided views and for mixed owned/borrowed operands. Both
  forms stay on the caller's existing fallback, which keeps behavior unchanged
  where the kernel cannot index the operand.
- `launch_binary_bindings` takes prepared operand bindings so the owned and the
  borrowed operand forms share one output allocation and launch path, instead of
  adding a parallel set of launchers per dtype.

## Verification conclusions and constraints

- `gpu/tensornetwork_permutation_optimized_f32` on an A100: eager
  209.2 ms -> 119.9 ms with verification passing, against 79.1 ms for the
  traced backend (ratio 2.75 -> 1.52). Over the same 13 tree executions the
  cuTENSOR `tensor_permute` launches fall from 32,335 to 5,412 and
  `cudaLaunchKernel` from 32,427 to 5,504; the fused
  `broadcast_multiply_float_e_f32` kernel replaces the
  `broadcast_in_dim` + `mul` pairs.
- The remaining gap to the traced backend comes from the per-step result
  reordering in `tenferro-einsum` (`transpose_to_labels`), not from this entry
  point. Removing the `duplicate()` in front of that transpose was measured
  neutral on this workload (121.2 ms vs 119.9 ms) and is not part of this
  change.
- The strided-view and mixed-operand fallbacks are unchanged and covered by the
  new CUDA-gated regression test, which also asserts that a strided view still
  returns `None`.
- Verification surface: `tenferro-gpu` lib and integration tests, the
  CUDA-gated `structural`, `reduction`, and fusion suites, and
  `tenferro-einsum` lib tests pass on an A100.
