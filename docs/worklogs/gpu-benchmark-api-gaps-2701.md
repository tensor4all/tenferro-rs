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
