# PyTorch-Like DynAdTensor Public Surface Design

## Goal

Reshape `tenferro-dyadtensor` so that its public tensor API looks like a
PyTorch-style dynamic tensor frontend:

- users work with `DynAdTensor`
- the AD engine uses `chainrules::Tape<DynTensor>` internally
- scalar values are rank-0 tensors
- mixed-dtype AD is supported through runtime-typed graph values

`DynTensor` remains necessary, but only as an internal primal payload and tape
value type.

## Why Change The Previous Direction

The earlier redesign direction still exposed both:

- `DynTensor` as dynamic primal tensor
- `DynAdTensor` as dynamic AD tensor

That keeps the engine clean, but it makes the public surface harder to learn:
users have to decide whether they should hold a primal tensor or an AD tensor.
For a PyTorch-like extension layer, that is the wrong trade-off.

PyTorch exposes one user-facing tensor object and stores autograd metadata
inside that object. tenferro should mirror that at the dyadtensor layer.

## Design Principles

- Keep the core numerical crates typed
- Make only `extension/tenferro-dyadtensor` dynamic and PyTorch-like
- Expose one public tensor object: `DynAdTensor`
- Keep `DynTensor` as the internal primal payload for `Tape<DynTensor>`
- Keep `Diag` as a small structured extension
- Keep linalg AD dense-only for now
- Avoid compatibility shims

## Public Model

### Public Tensor Type

`DynAdTensor` becomes the only public tensor object for dyadtensor users.

It must cover:

- primal values
- forward-mode values
- reverse-mode values

The public API should not require users to switch to a different tensor type
just because they want a primal-only value.

### Internal Primal Type

`DynTensor` remains internal and is used for:

- `Tape<DynTensor>`
- `TrackedValue<DynTensor>`
- `DualValue<DynTensor>`
- primal snapshots
- cast-back and promotion internals
- runtime/storage/FFI lowering

So the relationship becomes:

- public: `DynAdTensor`
- internal engine payload: `DynTensor`

## Typed API Removal

`AdTensor<T>`, `AdScalar<T>`, and `AdValue<T>` should leave the public surface.

If any typed helpers remain internally, they must be implementation detail only.
They must no longer shape:

- public docs
- public examples
- public result types
- public eager AD entry points

## Cast And Promotion Model

### Explicit Cast

Public explicit cast should be PyTorch-like:

- `DynAdTensor::to_scalar_type(...)`

This is a user-directed numeric conversion.

### Implicit Promotion

Operation-local promotion should stay internal:

- determine a result dtype
- cast operands to that dtype for execution
- cast pullback results back to each input dtype

This is the only way to support mixed-dtype reverse AD while keeping a
homogeneous tape.

## Structured Policy

`DynAdTensor` still carries structured payloads through internal `DynTensor`.

- dense tensors are supported
- `Diag` stays supported
- general `axis_classes` stay supported where already implemented

Structured AD support stays limited to operations with clear dense-reference
semantics:

- einsum
- reduction
- layout-preserving linear ops

Linalg AD stays dense-only.

## Result Types

Dynamic public API implies dynamic result types.

For linalg families, typed result wrappers such as `AdSvdResult<T>` should be
replaced by dynamic result wrappers whose fields are `DynAdTensor`.

This keeps the public surface coherent:

- one tensor object
- dynamic results built out of that tensor object

## Correctness Policy

- dense linalg AD: all supported oracle DB records replay
- structured einsum/reduction/layout ops: dense reference consistency
- structured linalg AD: explicit unsupported error
- cast/promotion: forward result dtype and reverse cast-back both tested

## Success Criteria

- `DynAdTensor` is the only public tensor object in `tenferro-dyadtensor`
- `DynTensor` is internal only
- dyadtensor graph payload is `Tape<DynTensor>`
- public docs and examples use `DynAdTensor`, not `AdTensor<T>`
- mixed-dtype reverse AD is expressible under the new cast/promotion model
