# tenferro Autodiff Design

Date: 2026-02-15

## Purpose

Define the formal architecture for automatic differentiation in tenferro-rs.
This document covers both the current API skeleton and planned implementation
work that is not yet complete in POC.

## Position in Workspace Architecture

`tenferro-autodiff` is a **pure AD framework** that sits below operation
crates like `tenferro-einsum`. It provides the tape system, wrapper types,
and rule traits, but does **not** contain operation-specific AD rules.

- `tenferro-autodiff` depends on `tenferro-tensor`, `tenferro-algebra`,
  `tenferro-device` (no dependency on `tenferro-einsum` or `tenferro-prims`).
- Operation-specific AD rules live with their operations:
  - Einsum AD functions (`tracked_einsum`, `dual_einsum`, `einsum_vjp`,
    `einsum_jvp`) are in `tenferro-einsum`.
  - Future operations (e.g., block-sparse matmul) define their own AD rules
    in their own crates.
- `tenferro-einsum` depends on `tenferro-autodiff` to use the AD framework.

```
tenferro-autodiff          ← Pure AD framework (no op-specific knowledge)
    ↑
tenferro-einsum            ← Einsum + einsum AD rules
```

## Scope

Current POC scope (in `tenferro-autodiff`):

- Public API skeleton for reverse-mode and forward-mode
- `TrackedTensor`, `DualTensor`, `backward`, `Gradients`, `BackwardPlan`
- Trait extension points: `VjpRule`, `JvpRule`

Current POC scope (in `tenferro-einsum`):

- `tracked_einsum`, `dual_einsum` — AD-aware einsum frontends
- `einsum_vjp`, `einsum_jvp` — local derivative kernels for FFI/manual AD

Planned scope (not yet implemented):

- Reverse-mode tape runtime and node execution
- Rule registry and decomposition rules for all primitives
- Higher-order APIs (HVP, forward-on-reverse)
- Device-aware execution paths for GPU backends

## API Layers

1. AD framework (`tenferro-autodiff`)

- `TrackedTensor<T>` — reverse-mode wrapper
- `DualTensor<T>` — forward-mode wrapper
- `backward(loss)` — reverse-mode execution
- `clear_tape<T>()` — tape management
- `Gradients<T>`, `BackwardPlan<T>` — result and plan types
- `VjpRule<T>`, `JvpRule<T>` — rule extension traits

2. Operation-specific AD rules (in each operation's crate)

Einsum AD rules (in `tenferro-einsum`):

- `tracked_einsum(subscripts, operands)` — reverse-mode einsum
- `dual_einsum(subscripts, operands)` — forward-mode einsum
- `einsum_vjp(subscripts, operands, cotangent)` — local VJP for FFI/manual AD
- `einsum_jvp(subscripts, primals, tangents)` — local JVP for FFI/manual AD

Future operations define their own AD rules in their own crates using the
`VjpRule` and `JvpRule` traits.

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
5. **AD framework does not depend on operation crates.** Each operation
   crate owns its AD rules. This avoids circular dependencies and keeps
   the framework extensible to user-defined operations.

## Out of Scope in This Phase

- Full runtime implementation of graph scheduling
- End-to-end optimized GPU backward kernels
- Full second-order differentiation API
