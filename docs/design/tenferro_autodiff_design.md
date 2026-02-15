# tenferro Autodiff Design

Date: 2026-02-15

## Purpose

Define the formal architecture for automatic differentiation in tenferro-rs.
This document covers both the current API skeleton and planned implementation
work that is not yet complete in POC.

## Position in Workspace Architecture

`tenferro-autodiff` is a **generic AD framework** (like Julia's ChainRulesCore.jl)
that does not depend on any tensor type. It defines the `Differentiable` trait
for tangent space operations, and generic wrapper types (`TrackedTensor<V>`,
`DualTensor<V>`) parameterized by any `V: Differentiable`.

- `tenferro-autodiff` depends only on `thiserror` (no tenferro crate deps).
- `tenferro-tensor` depends on `tenferro-autodiff` and implements
  `Differentiable for Tensor<T>`.
- Operation-specific AD rules live with their operations:
  - Einsum AD functions (`tracked_einsum`, `dual_einsum`, `einsum_rrule`,
    `einsum_frule`) are in `tenferro-einsum`.
  - Future operations (e.g., block-sparse matmul) define their own AD rules
    in their own crates.
- `tenferro-einsum` depends on `tenferro-autodiff` to use the AD framework.

```
tenferro-autodiff          ← Generic AD framework (Differentiable, no tensor deps)
    ↑
tenferro-tensor            ← impl Differentiable for Tensor<T>
    ↑
tenferro-einsum            ← Einsum + einsum AD rules
```

## Scope

Current POC scope (in `tenferro-autodiff`):

- Public API skeleton for reverse-mode and forward-mode
- `TrackedTensor`, `DualTensor`, `pullback`, `Gradients`, `PullbackPlan`
- Trait extension points: `ReverseRule`, `ForwardRule`
- Forward-over-reverse HVP: `HvpResult`, `hvp()`,
  `ReverseRule::pullback_with_tangents()`, `TrackedTensor::leaf_with_tangent()`

Current POC scope (in `tenferro-einsum`):

- `tracked_einsum`, `dual_einsum` — AD-aware einsum frontends
- `einsum_rrule`, `einsum_frule` — local derivative kernels for FFI/manual AD
- `einsum_hvp` — local HVP kernel for FFI/manual AD

Planned scope (not yet implemented):

- Reverse-mode tape runtime and node execution
- Rule registry and decomposition rules for all primitives
- Device-aware execution paths for GPU backends

## API Layers

1. AD framework (`tenferro-autodiff`)

- `Differentiable` — trait defining tangent space (zero_tangent, accumulate_tangent)
- `TrackedTensor<V>` — reverse-mode wrapper (with optional tangent for HVP)
- `DualTensor<V>` — forward-mode wrapper
- `pullback(loss)` — reverse-mode execution
- `hvp(loss)` — forward-over-reverse HVP execution
- `clear_tape<V>()` — tape management
- `Gradients<V>`, `PullbackPlan<V>`, `HvpResult<V>` — result and plan types
- `ReverseRule<V>`, `ForwardRule<V>` — rule extension traits
  (named after Julia's ChainRules.jl: rrule/frule)
  (`ReverseRule` includes `pullback_with_tangents` for HVP support)

All types are parameterized by `V: Differentiable` (not `T: ScalarBase`),
making the framework independent of any specific tensor or array type.

2. Operation-specific AD rules (in each operation's crate)

Einsum AD rules (in `tenferro-einsum`):

- `tracked_einsum(subscripts, operands)` — reverse-mode einsum
- `dual_einsum(subscripts, operands)` — forward-mode einsum
- `einsum_rrule(subscripts, operands, cotangent)` — local pullback for FFI/manual AD
- `einsum_frule(subscripts, primals, tangents)` — local pushforward for FFI/manual AD
- `einsum_hvp(subscripts, primals, tangents, cotangent, cotangent_tangent)` — local HVP for FFI/manual AD

Future operations define their own AD rules in their own crates using the
`ReverseRule` and `ForwardRule` traits.

## Algebra and Tropical Support

Autodiff must remain algebra-aware.

- Standard arithmetic: direct rrule/frule formulas over `+/*`.
- Tropical algebra (`tenferro-tropical`): formulas may need algebra-specific
  state (for example, argmax path information for max-plus variants).
- API design keeps this extensible by relying on `HasAlgebra` and
  `TensorPrims<A>` rather than hard-coding only standard arithmetic.

In short: tropical algebra is part of the design target even though the
runtime implementation is not complete in current POC.

## Design Decisions

1. Separate reverse and forward wrappers (`TrackedTensor` vs `DualTensor`).
2. Keep local rrule/frule callable without constructing a tape.
3. Keep public APIs backend-neutral; backend-specific execution stays in
   `tenferro-prims` / device layer.
4. Use doc examples on all public items to make API review possible before
   implementation.
5. **AD framework does not depend on operation crates.** Each operation
   crate owns its AD rules. This avoids circular dependencies and keeps
   the framework extensible to user-defined operations.

## Out of Scope in This Phase

- Full runtime implementation of graph scheduling
- End-to-end optimized GPU pullback kernels
- Full second-order differentiation API beyond HVP
  (HVP API skeleton is defined; full Hessian computation is deferred)
