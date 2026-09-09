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

## Consuming eager in-place

Expose explicitly named consuming in-place operations for CPU C32/C64,
shape-preserving c2c forward/inverse transforms on compact column-major storage
only. Check actual input compactness before ownership extraction: into_value
preserves a descriptor's layout and must not be mistaken for canonicalization.
Use the existing descriptor-bounded host write guard after extraction, not a
whole-root mutable slice. This preserves the offset and extent of compact
slices while leaving elements outside the logical tensor untouched.
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
