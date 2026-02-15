# Autodiff API Skeleton Design for tenferro-rs

Date: 2026-02-15

## Goal

Define a reviewable public API for automatic differentiation in tenferro-rs without adding implementations.

- Reverse-mode: tape/graph based API (`TrackedTensor`, `backward`)
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
- rule traits (`ReverseRule`, `ForwardRule`) for backend-agnostic extension (in `tenferro-autodiff`)

## Architecture: AD Framework vs Operation AD Rules

The AD framework (`tenferro-autodiff`) is a **pure framework** that does not
depend on any operation crate. Operation-specific AD rules live with their
operations:

- `tenferro-autodiff` — framework: `TrackedTensor`, `DualTensor`, `backward()`,
  `ReverseRule`, `ForwardRule`, tape management
- `tenferro-einsum` — einsum AD rules: `tracked_einsum`, `dual_einsum`,
  `einsum_rrule`, `einsum_frule`

Dependency direction: `tenferro-einsum → tenferro-autodiff` (not the reverse).

This design enables user-defined operations to register their own AD rules
without modifying the AD framework.

## Crates

### tenferro-autodiff

Dependencies:

- `tenferro-tensor`, `tenferro-algebra`, `tenferro-device`
- `strided-traits`
- `thiserror` for public error type

Public API:

Core types:

- `AutodiffError`, `AdResult<T>`
- `NodeId`
- `TrackedTensor<T>` (with optional tangent for HVP)
- `DualTensor<T>`
- `Gradients<T>`
- `BackwardPlan<T>`
- `HvpResult<T>`
- `SavePolicy`

Core traits:

- `ReverseRule<T>`: `pullback(cotangent) -> [(NodeId, Tensor<T>)]`,
  `pullback_with_tangents(cotangent, cotangent_tangent) -> [(NodeId, Tensor<T>, Tensor<T>)]`
- `ForwardRule<T>`: `pushforward(input_tangents) -> Tensor<T>`

Core functions:

- `clear_tape<T>()`
- `backward(loss: &TrackedTensor<T>) -> AdResult<Gradients<T>>`
- `hvp(loss: &TrackedTensor<T>) -> AdResult<HvpResult<T>>`

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
