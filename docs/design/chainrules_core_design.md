# chainrules-core Design

Date: 2026-02-15

## Purpose

Define the formal architecture for automatic differentiation in tenferro-rs.
This document covers both the current API skeleton and planned implementation
work that is not yet complete in POC.

## Position in Workspace Architecture

The AD system is split into two crates following Rust convention
(`foo-core` = traits, `foo` = full library):

- **`chainrules-core`** — Pure trait definitions (like Julia's ChainRulesCore.jl).
  Defines `Differentiable`, `ReverseRule<V>`, `ForwardRule<V>`, error types,
  `NodeId`, `SavePolicy`. Depends only on `thiserror`.
- **`chainrules`** — AD engine (like Zygote.jl). Provides `TrackedTensor<V>`,
  `DualTensor<V>`, `pullback()`, `hvp()`, `Gradients<V>`, `PullbackPlan<V>`.
  Depends only on `chainrules-core`. Re-exports all of `chainrules-core`.

Neither crate depends on any tensor or tenferro crate.

- `tenferro-tensor` depends on `chainrules-core` and implements
  `Differentiable for Tensor<T>`.
- Operation-specific AD rules live with their operations:
  - Einsum AD functions (`tracked_einsum`, `dual_einsum`, `einsum_rrule`,
    `einsum_frule`) are in `tenferro-einsum`.
  - Future operations define their own AD rules in their own crates.
- `tenferro-einsum` depends on `chainrules` (which re-exports core).

```
chainrules-core          ← Core AD traits (Differentiable, no tensor deps)
    ↑
chainrules               ← AD engine (TrackedTensor, pullback, hvp)
    ↑
tenferro-tensor            ← impl Differentiable for Tensor<T> (depends on chainrules-core)
    ↑
tenferro-einsum            ← Einsum + einsum AD rules (depends on chainrules)
```

## Scope

Current POC scope (in `chainrules-core`):

- `Differentiable` trait, `ReverseRule<V>`, `ForwardRule<V>` traits
- Error types (`AutodiffError`, `AdResult`), `NodeId`, `SavePolicy`

Current POC scope (in `chainrules`):

- `TrackedTensor<V>`, `DualTensor<V>`, `pullback`, `Gradients`, `PullbackPlan`
- Forward-over-reverse HVP: `HvpResult`, `hvp()`,
  `TrackedTensor::leaf_with_tangent()`

Current POC scope (in `tenferro-einsum`):

- `tracked_einsum`, `dual_einsum` — AD-aware einsum frontends
- `einsum_rrule`, `einsum_frule` — local derivative kernels for FFI/manual AD
- `einsum_hvp` — local HVP kernel for FFI/manual AD

Planned scope (not yet implemented):

- Reverse-mode tape runtime and node execution
- Rule registry and decomposition rules for all primitives
- Device-aware execution paths for GPU backends

## API Layers

1. Core AD traits (`chainrules-core`)

- `Differentiable` — trait defining tangent space (zero_tangent, accumulate_tangent)
- `ReverseRule<V>`, `ForwardRule<V>` — rule extension traits
  (named after Julia's ChainRules.jl: rrule/frule)
  (`ReverseRule` includes `pullback_with_tangents` for HVP support)
- `AutodiffError`, `AdResult`, `NodeId`, `SavePolicy`

2. AD engine (`chainrules`)

- `TrackedTensor<V>` — reverse-mode wrapper (with optional tangent for HVP)
- `DualTensor<V>` — forward-mode wrapper
- `pullback(loss)` — reverse-mode execution
- `hvp(loss)` — forward-over-reverse HVP execution
- `clear_tape<V>()` — tape management
- `Gradients<V>`, `PullbackPlan<V>`, `HvpResult<V>` — result and plan types

All types are parameterized by `V: Differentiable` (not `T: ScalarBase`),
making the framework independent of any specific tensor or array type.

3. Operation-specific AD rules (in each operation's crate)

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
6. **`Differentiable` does not require `Clone` on the primal type.**
   Only `Tangent: Clone` is required (for gradient accumulation at
   fan-out nodes). Large values like tensors may be expensive to clone;
   the AD engine avoids cloning primals by taking ownership
   (`detach(self)`, not `detach(&self)`). Implementations that need
   cheap duplication (e.g., for tape storage) should use internal
   reference counting (`Arc`) rather than relying on `Clone`.
7. **Tape lifetime: `NodeId` + runtime validation (POC).**
   `TrackedTensor` references the tape via `NodeId(usize)`, not a
   lifetime parameter or `Arc<Tape>`. This keeps the API simple.
   `NodeId` can become invalid after `clear_tape()` — detected at
   runtime, not compile time. Future migration path: store
   `Arc<Tape>` inside `TrackedTensor` (grabbed from thread-local at
   `leaf()` time). This is an internal change that does **not** alter
   public API signatures, because the tape is accessed through the
   `TrackedTensor` itself (e.g., `pullback(&loss)` reads `loss`'s
   internal tape reference).

## Out of Scope in This Phase

- Full runtime implementation of graph scheduling
- End-to-end optimized GPU pullback kernels
- Full second-order differentiation API beyond HVP
  (HVP API skeleton is defined; full Hessian computation is deferred)
