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
- `tenferro-cpu` owns the caller-held `CpuEngine`, compile-time provider
  selection, and execution policy. The engine takes over a caller-built `Pool`
  (or runs serially without one) and owns persistent buffers and workspaces.
  Calls borrow it through `&mut`; there is no session entry or resource arbiter.

`tenferro-tensor` owns backend traits and tensor semantics, but does not
instantiate CPU strided kernels. Its `TensorElementwise::elementwise_read_into`
hook is required of concrete backends. This keeps the tensor crate independent
of the CPU kernel families and makes placement/context policy explicit at the
backend boundary.

## Execution contracts

CPU elementwise read-into enters through a caller-provided `&mut CpuEngine` and
calls the internal ordinary kernel with the operation's execution token and the
engine's persistent `BufferPool`. Supported normal fusion uses the separate
fused adapter with the same engine-owned resources and token policy; no
execution context is installed. Unsupported fused plans return the existing
`None` result and retain the normal fallback behavior.

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

The tenferro integration uses the strided-rs #254 merge commit
`fbd10fa5b70bb462b961cfd9e02faadb6bb95be0`. The split packages remain
unpublished; registry publication is not part of this implementation.
