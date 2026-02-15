# tenferro Autodiff Design

Date: 2026-02-15

## Purpose

Define the formal architecture for automatic differentiation in tenferro-rs.
This document covers both the current API skeleton and planned implementation
work that is not yet complete in POC.

## Position in Workspace Architecture

`tenferro-autodiff` sits above `tenferro-einsum` and `tenferro-prims`.

- Forward computation uses `tenferro-einsum` / `TensorPrims<A>`.
- Derivatives use VJP/JVP rules expressed in terms of primitive ops.
- Algebra-specific behavior is dispatched via `HasAlgebra` and `TensorPrims<A>`.

## Scope

Current POC scope:

- Public API skeleton for reverse-mode and forward-mode
- `TrackedTensor`, `DualTensor`, `backward`, `einsum_vjp`, `einsum_jvp`
- Trait extension points: `VjpRule`, `JvpRule`

Planned scope (not yet implemented):

- Reverse-mode tape runtime and node execution
- Rule registry and decomposition rules for all primitives
- Higher-order APIs (HVP, forward-on-reverse)
- Device-aware execution paths for GPU backends

## API Layers

1. Local derivative kernels

- `einsum_vjp(subscripts, operands, cotangent) -> Vec<Tensor<T>>`
- `einsum_jvp(subscripts, primals, tangents) -> Tensor<T>`

These are useful for:

- FFI integrations (`custom_vjp` / `custom_jvp`)
- explicit gradient code without global tape

2. Reverse-mode user API

- `TrackedTensor<T>`
- `tracked_einsum(...)`
- `backward(loss)`

3. Forward-mode user API

- `DualTensor<T>`
- `dual_einsum(...)`

4. Rule extension API

- `VjpRule<T>`
- `JvpRule<T>`

## Algebra and Tropical Support

Autodiff must remain algebra-aware.

- Standard arithmetic: direct VJP/JVP formulas over `+/*`.
- Tropical algebra (`tenferro-tropical`): formulas may need algebra-specific
  state (for example, argmax path information for max-plus variants).
- API design keeps this extensible by relying on `HasAlgebra` and
  `TensorPrims<A>` rather than hard-coding only standard arithmetic.

In short: tropical algebra is part of the design target even though the
runtime implementation is not complete in current POC.

## Design Decisions

1. Separate reverse and forward wrappers (`TrackedTensor` vs `DualTensor`).
2. Keep local VJP/JVP callable without constructing a tape.
3. Keep public APIs backend-neutral; backend-specific execution stays in
   `tenferro-prims` / device layer.
4. Use doc examples on all public items to make API review possible before
   implementation.

## Out of Scope in This Phase

- Full runtime implementation of graph scheduling
- End-to-end optimized GPU backward kernels
- Full second-order differentiation API
