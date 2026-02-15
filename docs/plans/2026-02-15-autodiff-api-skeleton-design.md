# Autodiff API Skeleton Design for tenferro-rs

Date: 2026-02-15

## Goal

Define a reviewable public API for automatic differentiation in tenferro-rs without adding implementations.

- Reverse-mode: tape/graph based API (`TrackedTensor`, `pullback`)
- Forward-mode: tangent propagation API (`DualTensor`, frule)
- Rule extension traits (`ReverseRule`, `ForwardRule`) for backend-agnostic AD

All function bodies remain `todo!()` in POC phase.

## External Reference

`ndtensors-rs` was referenced for API shape and separation of concerns:

- split user-facing wrappers: `TrackedTensor` (reverse) and `DualTensor` (forward)
- expose local primitive derivatives (`contract_rrule`, `contract_frule`)
- keep op-level rrule/frule independent from high-level graph execution

Applied to tenferro as:

- `tracked_einsum` and `dual_einsum` as AD-aware frontends (in `tenferro-einsum`)
- `einsum_rrule` and `einsum_frule` as local derivative kernels (in `tenferro-einsum`)
- rule traits (`ReverseRule`, `ForwardRule`) for backend-agnostic extension (in `chainrules-core`)

## Architecture: AD Framework vs Operation AD Rules

The AD system is split into two crates following Rust convention:

- `chainrules-core` — traits: `Differentiable`, `ReverseRule`, `ForwardRule`
- `chainrules` — AD engine: `TrackedTensor`, `DualTensor`, `pullback()`, tape management
- `tenferro-einsum` — einsum AD rules: `tracked_einsum`, `dual_einsum`,
  `einsum_rrule`, `einsum_frule`

Dependency direction: `tenferro-einsum → chainrules → chainrules-core`.

This design enables user-defined operations to register their own AD rules
without modifying the AD framework.

## Crates

### chainrules-core

Dependencies: `thiserror` (only dependency — no tensor or algebra deps)

Public API:

- `Differentiable` — tangent space definition (`Tangent` type, `zero_tangent`,
  `accumulate_tangent`). Like Julia's ChainRulesCore.jl.
- `ReverseRule<V>`: `pullback(cotangent) -> [(NodeId, V::Tangent)]`,
  `pullback_with_tangents(cotangent, cotangent_tangent) -> [(NodeId, V::Tangent, V::Tangent)]`
- `ForwardRule<V>`: `pushforward(input_tangents) -> V::Tangent`
- `AutodiffError`, `AdResult<T>`, `NodeId`, `SavePolicy`

### chainrules

Dependencies: `chainrules-core`

Public API (re-exports all of `chainrules-core`):

- `TrackedTensor<V: Differentiable>` (with optional tangent for HVP)
- `DualTensor<V: Differentiable>`
- `Gradients<V: Differentiable>`
- `PullbackPlan<V: Differentiable>`
- `HvpResult<V: Differentiable>`
- `clear_tape<V>()`
- `pullback(loss: &TrackedTensor<V>) -> AdResult<Gradients<V>>`
- `hvp(loss: &TrackedTensor<V>) -> AdResult<HvpResult<V>>`

### tenferro-einsum (AD additions)

Einsum AD functions added to `tenferro-einsum`:

- `tracked_einsum(subscripts, operands)` — reverse-mode einsum
- `dual_einsum(subscripts, operands)` — forward-mode einsum
- `einsum_rrule(subscripts, operands, cotangent)` — local pullback for FFI/manual AD
- `einsum_frule(subscripts, primals, tangents)` — local pushforward for FFI/manual AD
- `einsum_hvp(subscripts, primals, tangents, cotangent, cotangent_tangent)` — local HVP for FFI/manual AD

## Design Decisions

1. Keep reverse and forward APIs separate.
2. Do not force a specific graph runtime strategy in the public interface.
3. Keep local rrule/frule callable without tape construction.
4. Reuse `tenferro_device::Result` for primitive derivative helpers where AD-specific errors are not required.
5. Require minimal but sufficient doc examples for each public item, matching repository doc policy.
6. AD framework does not depend on operation crates. Each operation crate owns its AD rules.

## Non-goals (Current POC)

- No runtime implementation of tape/graph execution.
- No decomposition AD rules yet for every `TensorPrims` operation.
- No full second-order API beyond HVP (HVP API skeleton is now defined).
