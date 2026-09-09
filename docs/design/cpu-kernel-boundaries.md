# CPU kernel family boundaries

## Ownership

The CPU backend is assembled from three internal crates:

- `tenferro-cpu-basic` owns the persistent host `BufferPool`, the
  `PooledUninitOutput` full-overwrite guard, host-storage views, raw descriptor
  adapters, and shared dtype/complex-order helpers. It depends on
  `tenferro-tensor` and `strided-basic`, but not on an operation-family kernel.
- `tenferro-internal-cpu-kernels` owns ordinary dtype-dispatch elementwise
  kernels and the pool-aware one-shot `elementwise_read_into` implementation.
  It depends on `tenferro-cpu-basic` and the ordinary `strided-kernel` crate.
- `tenferro-cpu-fused` owns the runtime-DAG fused elementwise adapter. It
  depends on `tenferro-cpu-basic` and `strided-fused`, not on the ordinary
  internal kernel implementation.
- `tenferro-cpu` owns `CpuBackend`, provider selection, session entry,
  resource arbitration, and execution policy. It assembles both operation
  families and borrows the same context and pool for each call.

`tenferro-tensor` owns backend traits and tensor semantics, but does not
instantiate CPU strided kernels. Its `TensorElementwise::elementwise_read_into`
hook is required of concrete backends. This keeps the tensor crate independent
of the CPU kernel families and makes placement/context policy explicit at the
backend boundary.

## Execution contracts

CPU elementwise read-into enters through `CpuBackend` or `CpuExecSession` and
calls the internal ordinary kernel with the installed `CpuExecutionContext` and
persistent pool. Supported normal fusion enters the separate fused adapter from
the same closure, with the same context and pool. Unsupported fused plans return
the existing `None` result and retain the normal fallback behavior.

The one-shot read-into implementation validates arity, storage overlap, host
placement, dtype, shape and reachable layouts before constructing raw strided
descriptors or writing the caller-owned output. Its uninitialized output path
uses the existing full-overwrite guard and never forms an initialized view
before completion. The raw adapters in basic remain `unsafe` and are used only
by callers that hold the corresponding typed storage and layout proof.

CUDA implements the read-into hook through its existing device-native allocating
operations and copy boundary. WebGPU implements the hook as an explicit typed
unsupported operation because that provider has no elementwise implementation;
it does not silently transfer or fall back to CPU. CPU uses its pooled strided
replay and preserves automatic fusion.

The split does not add a dynamic callback or per-element indirection. Generic
callbacks and optimized multiply/comparison paths stay in their existing
owners. Existing validation, error translation, output initialization, pool
reuse, thread ownership and provider selection remain the responsibility of
their original owner.

## Dependency staging

The tenferro worktree temporarily pins all strided packages to the exact
strided-rs PR #254 head used for this integration candidate. After that PR is
merged, update the pin to its merged commit before merging or publishing the
tenferro change. Registry publication is not part of this implementation.
