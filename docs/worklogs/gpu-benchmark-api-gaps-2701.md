# GPU benchmark public-API gaps (#2701)

## Decisions

- Output reuse is spelled in the #2552 grammar, not as a new suffix. `_into` with
  `Dest::Overwrite` is declared for `add`, `sub`, `mul`, `div`, `neg` and `conj` on the
  concrete and eager layers. The traced layer keeps the allocating forms. `_read`-style names
  and an eager `ElementwiseReadOp` entry were rejected: the registry forbids `_read` operation
  names and treats backend hooks as provider contracts.
- An eager `_into` destination is a caller-owned `Tensor`, not an eager value. While gradient
  recording is enabled, a tracked operand or an active capture is refused before any write,
  because the destination has no place in the reverse graph. The `_into` forms do not promote:
  every operand must have the destination's dtype. This is checked at the concrete boundary,
  since a backend's allocating kernel may promote a real scalar against a complex one.
- "Zero-copy" is measured, not asserted. An eager value reads as a descriptor view, and the CUDA
  elementwise path staged every view. A flat view (one compact span at its allocation's base)
  is now passed to the flat kernel where it lies. The CUDA test pins the contract with the
  runtime's allocation and launch counters: one launch and no allocation per eager `_into` on
  compact operands, measured for the f64 arithmetic forms and f64/c64 `conj`.
- No new name for the eager "borrowed strided read". An eager `transpose` already yields a view,
  but over a compact copy it makes, and the flat CUDA kernels gather permuted views once.
  Reading permuted views in place needs stride-aware kernel entries in `t4a-cuda-kernels`
  (only `StridedOp::{Mul, Axpby}` exist). Claiming the route without them would hide a copy.
- Scalar setup inside operations is removed where it is local. The Householder reflector
  application uses cuBLAS host pointer mode with 16-byte-aligned host scalars (the #1870 fault
  class) instead of three uploads per call. Kernel status words are zeroed by a device memset.
  The composite-scalar class (softmax, layer norm, gelu, reduce_mean, scale_*, inv/pinv
  identities) is deferred. It needs an engine-owned bounded device-scalar cache, and on the
  eager side it also needs compact eager values read in place by every CUDA `*_read` hook, not
  only by `_into`. Otherwise each cached scalar is still staged per use.

## Verification conclusions and constraints

- On an A100, the eager `_into` forms on compact eager operands (f64 arithmetic, f64/c64
  `conj`) perform 0 device allocations and 1 launch and are bit-identical to the owned-tensor
  hook. Complex and integer division are checked for values only: they allocate and zero a
  4-byte kernel status word per call (no result allocation, no operand copy). An integer
  division by zero is a NumericalFailure reported after the launch, so the destination contents
  are then unspecified.
- Allocating eager CUDA ops still stage compact eager operands (`exp` on an eager value: 2
  allocations and 2 launches against 1 and 1 on the owned hook). Fixing that needs a
  borrowed-allocation `Resolved` in the CUDA `device_read`.
- Remaining unsupported benchmark classes, and which are reference-API limitations
  (JAX/PyTorch), are listed in the PR description's ledger.

## Second round: in-place strided reads and hoisted composite setup

### Decisions

- Strided, offset, reversed and broadcast operands are read where they lie. The flat CUDA
  elementwise entries address `len` consecutive elements, so the fix lives in the kernel
  catalogue: `t4a-cuda-kernels` ABI 15 adds layout-taking unary, binary, compare and select
  entries that compute the flat entries' per-element formulas with the same device helpers. The
  engine resolves each operand once: a flat region goes to the flat entry at its own allocation, any
  other region of rank at most `MAX_RANK` goes to the layout-taking entry, and only a wider view is
  still staged. Reductions take a strided view's own layout in their first pass. This also removes
  the staging of compact eager values by allocating ops, the open finding of the first round.
- Composite constants stay host values until an operation consumes them. `Engine` gets three
  provided hooks (`binary_scalar_read`, `select_scalar_read`, `fill_scalar`) whose defaults upload
  per call, exactly as before. The CUDA engine keeps a bounded set of resident device scalars and
  lends them to the same formula (a zero-stride view plus the matching `*_read` entry), so its
  result equals the portable hook's bit for bit. A cached device scalar cannot be handed out as a
  value: `Tensor` and `TensorValue` have one move-only owner, so the hook is operation-level.
- An eager session reuses its rank-0 constants. An untracked eager op on a GPU reads a broadcast
  operand as a zero-stride view instead of recording an eager `broadcast_in_dim`, which duplicates
  its base by design (an eager output cannot borrow another value's allocation group). With
  gradient recording on, the recorded broadcast is unchanged.
- Eager output reuse for `dot_general`/einsum (and the accumulating `axpby`), eager immediate
  `vdot`/`norm_squared`, and an eager prepared einsum plan are not added here. They extend the
  #2552 registry grammar on the eager layer, which the first round already marked as a separate
  decision.

### Verification conclusions and constraints

- On the A100, every operation the benchmark measures on a transposed read costs what a compact
  operand costs (one allocation and one launch for elementwise, the compact reduction's counts
  for a reduction) on the direct and eager routes, bit-identical to the compact operand.
- In steady state every benchmarked composite (activations, softmax family, layer/RMS norm,
  reduce_mean, scale_real/complex) transfers nothing to the device and allocates only kernel
  results; eager costs what direct costs and returns the same bits. inv, pinv and lstsq transfer
  nothing in steady state. Softmax went from 15 allocations, 11 launches and 2 transfers to 7, 7
  and 0 on the direct route.
- The layout-taking entries index with 64-bit coordinates, so a strided read is slower per element
  than a flat one; compact operands keep the flat entries.
