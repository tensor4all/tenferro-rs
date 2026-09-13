# CUDA `_read` entry points accept borrowed views

## Decisions

- Implement the missing `_read` entry points in `tenferro-gpu` rather than
  relaxing the shared `tenferro-tensor` defaults. Those defaults reject
  borrowed views deliberately and are covered by
  `default_read_methods_delegate_owned_tensors_and_reject_views`; the CUDA
  backend is the side that does not honor the documented `_read` contract
  (`docs/design/tensor-prims.md`).
- Materialize a borrowed view once through the CUDA backend's own
  `to_contiguous_read`, then call the existing owned kernel. CUDA has no
  view-native kernel for the structural, reduction, or most analytic and
  elementwise operations (`ElementwiseReadOp` covers only add, sub, mul, neg,
  conj, div), so a native route would be a larger change with partial coverage
  and a second result path to verify. `TensorDot::dot_general_read` already
  uses this fallback, so the CUDA entry points now follow the same rule.
- Forward the same entry points from `CudaExecSession`. The traced runtime
  reaches them through the erased `BackendSession` surface, so overriding them
  on `CudaBackend` alone would not fix the reported failure.
- Materialization is bounded to the call: `CudaReadInput` owns the materialized
  tensor for exactly as long as the kernel needs it.
- The class is closed for the current trait surface, and
  `tests/integration/backend_read_contract.rs` re-derives the view-rejecting
  entry points from the `tenferro-tensor` source and fails when `CudaBackend`
  or `CudaExecSession` stops covering one. No repository audit rule was added:
  the guard covers the same root cause at the boundary where it can recur.
- For the six operations in `ElementwiseReadOp` the CUDA backend also has a
  view-native path (`elementwise_read_into`). Routing those entry points
  through it would avoid one materialization and stays a separate
  performance change; this fix keeps a single result path.

## Verification conclusions and constraints

- `cargo test -p tenferro-tensor --lib` (275 tests) and
  `cargo test --features cuda -p tenferro-gpu --lib` (119 tests) pass; the
  CUDA-gated `structural` (30) and `reduction` (11) suites pass on an A100.
- The new CUDA-gated regression test
  `test_cuda_read_entry_points_accept_borrowed_views` fails on the previous
  revision and passes with this change, and
  `tests/integration/backend_read_contract.rs` passes with it.
- `tensor4all/tenferro-benchmark` `gpu/linalg_jvp_vjp`, which drives traced
  JVP/VJP graphs, reported 30 `runtime failed` rows before the fix and 50 `ok`
  rows after it on an A100.
- Not covered: dtypes whose `to_contiguous_read` is itself unsupported (Bool)
  still return the typed unsupported error from materialization.
