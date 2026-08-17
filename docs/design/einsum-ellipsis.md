# Einsum Ellipsis Resolution

## Decision

`EinsumNotation` is the unresolved public notation. Its axes are `Label(u32)` or
`Ellipsis`; `EinsumSubscripts` remains the canonical rank-resolved integer-label
payload used by planning, execution, lowering, caching, and AD.

A single resolver expands one ellipsis per input/output after input ranks are
known. Ellipsis labels are allocated from unused `u32` values, align from the
right, and are shared by compatible batch axes. Input dimensions use
NumPy-style equal-or-one broadcasting (with zero plus one producing zero).
An input ellipsis may cover zero axes. If an input ellipsis covers axes, the
explicit output must contain an ellipsis. Parenthesized notation remains
unsupported in this first implementation.

Concrete and eager operations resolve from concrete input shapes. Traced
operations resolve ellipsis when dimensions are concrete; symbolic dimensions
continue to support the existing equality constraints, while unresolved
ellipsis broadcasting returns a typed planning/validation error rather than
introducing a backend-specific shape rule.

Singleton broadcast axes are expanded before contraction. The contraction
planner stores the broadcasted label extent, and the eager/standard-op
builders insert `BroadcastInDim` operations or zero-stride views as needed.
Semantic extension payloads carry only a private permission bit for the
shape-inference layer; the payload still contains resolved labels, and ordinary
integer-label operations retain their existing equality guards. This keeps
kernels and extension payloads ellipsis-free.

## Rejected alternatives

- A sentinel `u32` label was rejected because it would reserve a value in the
  existing canonical integer-label API and would make unresolved rank state
  visible to planners.
- Backend-specific ellipsis handling was rejected; all backends consume the
  existing resolved contraction tree.
- Implicit output notation and ellipsis reduction by omission are deferred.
- Parenthesized `NestedEinsum` ellipsis is deferred until its intermediate-rank
  contract is separately specified.
