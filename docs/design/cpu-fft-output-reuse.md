# CPU FFT output reuse and consuming eager execution

Accepted scope: #1765 and #1766; implementation is in progress. The maintainer
approved the host-reclamation/CPU-pool synchronization changes as part of the
same performance PR. See the linked [work log](../worklogs/1765-1766-cpu-performance.md)
for evidence and verification status.

## Ownership and reclamation

Reuse the existing CPU BufferPool and its retention limits, clearing and stats.
Separate its short pool-state lock from CPU execution admission and the engine
resource lock. Acquiring or returning a vector holds only the short state lock;
never hold that lock while invoking a numerical kernel, returning a tensor,
or destroying a tensor with a recycler. Do not introduce a second pool or a
queue of indefinitely retained buffers.

A completed pooled host allocation carries a weak typed reference to its
original recycler. The host allocation remains the sole owner of the vector.
Its final destruction transfers the vector back to the original pool if that
pool still exists, otherwise normal destruction frees it. The weak reference
must not prolong backend lifetime or create a cycle. AD containers, views and
capture keep the existing ownership/borrow semantics: recycling is a consequence
of final root destruction, not eager-handle count or a separate liveness table.

Explicit extraction to Vec transfers responsibility to the caller and disarms
the recycler. Explicit reclaim must not return the same allocation twice.
Partial uninitialized outputs remain MaybeUninit until successful full overwrite;
errors and unwind must not expose uninitialized scalar values. Automatic return
must not trigger the legacy loan's replacement allocation for a live output.
Pool bounds and clear/stat operations apply to the one shared retained state.
Changing this boundary requires tests for drop during execution, backend-first
drop, live aliases/AD records, explicit extraction, limits and clearing.

## FFT execution

The CPU FFT kernel writes directly into its final host output destination using
the canonical pooled uninitialized owner.

The maintainer approved retaining the managed output copy boundary for this PR,
including Apple CPU/Metal shared allocations. Inspection and the managed
regression test show that legacy BackendStorage::map_write returns a copy-only
HostWriteGuard, not a writable span. SharedTensorAllocationDomain::allocate also
permits uninitialized output, so treating its storage as an initialized mutable
slice is invalid. The current implementation preserves this provider boundary
with pooled staging and a single copy, while sharing the parallel lane kernel.
Removing that copy requires a separately specified provider-owned uninitialized
write/completion capability; the host recycler alone cannot supply it. Direct
managed writes, synchronization and Apple hardware validation are separate
future work, not prerequisites or performance claims of this PR.

Padding and c2r reconstruction initialize every scratch element that is read.
Plans remain in the existing extension cache; no global plan or scratch cache.

Lane jobs are executed inside the existing CPU session context and obey its
thread budget and nested policy. The one-thread path remains sequential.
Partition the lane domain, including noncontiguous lanes of the last axis;
prove disjoint output index sets before parallel execution. Reuse lane and
RustFFT scratch within each job rather than allocate per lane.

## Caller-owned FFT destinations (#2653)

The concrete FFT family declares overwrite-only `_into` operations. The final
output's dtype, shape, placement and supported layout are checked before any
transform runs. Accumulate and scaled destinations are unsupported. Typed
real-to-complex and complex-to-real calls use the result scalar's destination
type. The eager and traced layers keep their existing allocating contract.

`FftExecutor` retains prepared requests and reuses its caller-owned
`FftPlanCache`; reuse the CPU engine as well. For an unchanged request with
owned dtype-erased `Tensor` input and destination in plain host storage,
repeated `_into` calls make zero Rust heap allocations after the request, provider plans and engine working
buffers are warm and remain retained. This includes C2C, full/one-sided R2C,
C2R, both scalar widths, and multi-axis requests. Changing the request or
thread budget, clearing a cache/pool, or evicting its working set requires
warming again. Cache and pool limits remain effective; there is no unbounded
or global scratch cache. Noncompact inputs keep the documented same-placement
canonicalization boundary and are outside this allocation guarantee.

A non-leading axis of a compact column-major tensor is not a noncompact
operand. Its lanes have stride `product(shape[..axis])`; the CPU kernel gathers
one transform lane per job and scatters directly into the caller's output.
There is no full-input gather. Typed inputs and compact borrowed views also avoid allocations
while their metadata fits the inline rank capacity (8); adapters for higher
ranks may allocate metadata. A noncompact view still gathers its logical
input once, costing `product(input.shape()) * size_of(input scalar)` bytes of
compact storage. The allocation regression includes that view as a negative
control, so a warm pool cannot conceal the distinction.

CPU `_into` accepts host storage destinations; backend-backed destinations,
including managed providers, are unsupported. The allocating managed-output boundary above is unchanged.

CUDA destinations remain on the executing device. CUDA uses same-device
intermediate storage and copies the result into the destination on device;
no Rust allocation guarantee applies there, and neither
input nor output is implicitly uploaded or downloaded.

## Consuming eager in-place

Expose explicitly named consuming in-place operations for CPU C32/C64,
shape-preserving c2c forward/inverse transforms on compact column-major storage
only. Check actual input compactness before ownership extraction: into_value
preserves a descriptor's layout and must not be mistaken for canonicalization.
Use the existing descriptor-bounded host write guard after extraction, not a
whole-root mutable slice. This preserves the offset and extent of compact
slices while leaving elements outside the logical tensor untouched.

An extracted owner exports its own logical values; the whole root allocation and
the descriptor offset are a separate, physical view (`as_view`, `buffer`,
`prepare_device_*`, and the owning reinterpretations). Every typed accessor -
`host_data`/`host_data_mut`, `as_slice`, the host mapping guards, `duplicate`,
`get`/`get_mut`, the CPU adapters and `into_host_vec`/`into_parts` - reads and
writes the exported extent only, so no consumer can reach the root prefix or the
bytes outside it. A value (`TensorValue`) layout stays relative to the backing
root, so a value duplicate copies that root rather than the exported prefix.
A nonzero descriptor offset is accepted only for a plain host allocation: the
provider binding path reads the root handle without a descriptor offset, so a
device-backed or managed owner stays whole-root.
Existing borrowed fft/ifft
remain nondestructive. Validate dtype, axis and backend before taking ownership.
Reject active grad/capture participation; do not infer safety from no_grad alone.
Normal untracked leaves carry a separate semantic leaf value in the current AD
implementation; the presence of that metadata alone is not a mutation ban.
Saved physical values remain protected by structural ownership extraction.
Use existing into_value structural extraction to reject shared handles or
retained containers without implicit duplication. Obtain write access through
the extracted tensor's exclusive borrow. Do not form overlapping input/output
views for an out-of-place kernel. A rejected ownership acquisition must preserve
a usable original value (boxed only on the error path); subsequent execution
errors consume the acquired input without rollback. No r2c/c2r or resizing in-place in this contract.

## Binary operand preparation

Share borrowed/owned broadcast preparation between dynamic and typed runtime
surfaces. Same-shape and identity source-shape operands remain borrowed; only
actual reshape/broadcast results need owners. Keep validation, error categories,
dtype promotion and backend dispatch intact, including compare/select/clamp.
