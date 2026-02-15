# Autodiff API Skeleton Design for tenferro-rs

Date: 2026-02-15

## Goal

Define a reviewable public API for automatic differentiation in tenferro-rs without adding implementations.

- Reverse-mode: tape/graph based API (`TrackedTensor`, `backward`)
- Forward-mode: tangent propagation API (`DualTensor`, JVP)
- Primitive derivative APIs for interop (`einsum_vjp`, `einsum_jvp`)

All function bodies remain `todo!()` in POC phase.

## External Reference

`ndtensors-rs` was referenced for API shape and separation of concerns:

- split user-facing wrappers: `TrackedTensor` (reverse) and `DualTensor` (forward)
- expose local primitive derivatives (`contract_vjp`, `contract_jvp`)
- keep op-level backward/JVP rules independent from high-level graph execution

Applied to tenferro as:

- `tracked_einsum` and `dual_einsum` as AD-aware frontends for `tenferro-einsum`
- `einsum_vjp` and `einsum_jvp` as local derivative kernels usable from FFI/custom AD
- rule traits (`VjpRule`, `JvpRule`) for backend-agnostic extension

## Added Crate

New workspace member:

- `tenferro-autodiff`

Dependencies:

- `tenferro-tensor`, `tenferro-einsum`, `tenferro-algebra`, `tenferro-device`
- `strided-traits`
- `thiserror` for public error type

## Public API Summary

Core types:

- `AutodiffError`, `AdResult<T>`
- `NodeId`
- `TrackedTensor<T>`
- `DualTensor<T>`
- `Gradients<T>`
- `BackwardPlan<T>`
- `SavePolicy`

Core traits:

- `VjpRule<T>`: `backward(cotangent) -> [(NodeId, Tensor<T>)]`
- `JvpRule<T>`: `forward(input_tangents) -> Tensor<T>`

Core functions:

- `clear_tape<T>()`
- `backward(loss: &TrackedTensor<T>) -> AdResult<Gradients<T>>`
- `tracked_einsum(subscripts, operands)`
- `dual_einsum(subscripts, operands)`
- `einsum_vjp(subscripts, operands, cotangent)`
- `einsum_jvp(subscripts, primals, tangents)`

## Design Decisions

1. Keep reverse and forward APIs separate.
2. Do not force a specific graph runtime strategy in the public interface.
3. Keep local VJP/JVP callable without tape construction.
4. Reuse `tenferro_device::Result` for primitive derivative helpers where AD-specific errors are not required.
5. Require minimal but sufficient doc examples for each public item, matching repository doc policy.

## Non-goals (Current POC)

- No runtime implementation of tape/graph execution.
- No decomposition AD rules yet for every `TensorPrims` operation.
- No HVP/second-order API yet (can be layered on `DualTensor` + `backward` later).
