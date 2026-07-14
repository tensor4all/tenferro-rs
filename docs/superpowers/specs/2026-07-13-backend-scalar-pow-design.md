# Backend Scalar `pow` Design

## Context

Issue #1371 concerns rank-0 operands passed directly to CPU and CUDA backend
`pow`. The public `TensorOpsExt`, typed, eager, and traced APIs already own
general NumPy-style broadcasting and emit explicit broadcast operations before
backend execution. Direct backend `pow` currently requires exact shape equality.

JAX, PyTorch, and NumPy establish scalar/tensor power as expected user-facing
semantics. This design adds the corresponding narrow backend capability to both
CPU and CUDA without moving general broadcasting into the backend layer.

## Contract

Exactly one same-dtype rank-0 operand may pair with a non-scalar tensor:

- tensor base, scalar exponent: `[N, ...] ** [] -> [N, ...]`
- scalar base, tensor exponent: `[] ** [N, ...] -> [N, ...]`

Two scalars continue through the existing equal-shape path and return shape
`[]`. Two unequal non-scalar shapes continue to return `ShapeMismatch`.

Supported dtypes are the existing backend `pow` dtypes: `F32`, `F64`, `I32`,
and `I64` on CUDA, plus the CPU dtypes it already supports. This change does not
add CUDA complex power or mixed-dtype power.

Integer exponent validation remains global over the logical exponent operand.
A negative scalar exponent fails before output production, and a tensor
exponent containing any negative value produces the same typed
`NegativeIntegerExponent` error as the equal-shape path.

## CPU Design

The CPU implementation gains a scalar-aware internal mapping path shared by
owned tensors and `TensorRead` views. It selects the non-scalar operand shape,
reads the rank-0 value once, allocates one output, and applies `PowElem` in the
original operand order. It must not allocate a dense broadcast of the scalar.

Integer validation runs against the original exponent view before scalar-aware
mapping. Exact-shape behavior and complex CPU behavior remain unchanged.

## CUDA Design

CUDA reuses the existing scalar-binary launch boundary introduced for `div`
and `rem`. New scalar-indexed float and checked-integer pow kernels map the
rank-0 operand to index zero and the tensor operand to the output index. The
checked integer kernel retains the device error flag and reports the existing
negative-exponent error after synchronization.

The ordinary equal-shape kernels remain unchanged. Source-contract tests must
forbid `broadcast_typed` and require the scalar launch only in the one-scalar
branch.

## Public, Traced, And AD Surfaces

No public, traced, or AD implementation changes are required. Those layers
already broadcast inputs explicitly and reduce broadcast cotangents to original
input shapes. Existing tests remain regression coverage; focused tests will
confirm that this change does not alter their graph representation.

## Error And Edge Semantics

- Dtype mismatch remains checked before shape dispatch.
- Unequal non-scalar shapes remain `ShapeMismatch` with original operand shapes.
- Empty non-scalar tensors produce an empty output after residency and shape
  validation, without launching a kernel.
- Floating results follow the existing CPU `pow` oracle for finite values,
  NaN, infinity, and signed zero.
- Device residency is validated before CUDA early returns.

## Verification

Testing follows RED/GREEN discipline:

1. Replace the CPU/CUDA scalar-pow rejection regression with expected parity
   and observe it fail on both direct backends.
2. Add CPU owned/read-view tests for both operand orders, dtypes, empty output,
   negative integer exponents, and incompatible non-scalar shapes.
3. Add CUDA A100 parity tests for `F32`, `F64`, `I32`, and `I64`, including
   floating exceptional classes and exact integer errors.
4. Update launch/source contracts and prove they fail before the CUDA dispatch
   change.
5. Run focused crates, CUDA no-run compilation, the ignored A100 suite, and the
   complete repository pre-PR checklist.

## Non-goals

- General backend-level broadcasting
- Dense scalar materialization
- New dtype promotion
- Complex CUDA `pow`
- Mixed-dtype `pow`
- New public APIs, dependencies, feature flags, or AD conventions
