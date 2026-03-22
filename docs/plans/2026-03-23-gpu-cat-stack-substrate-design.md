# GPU Cat/Stack Substrate Design

**Context**

`tenferro-linalg` still has host-loop tensor ops such as `cross`, `householder_product`, and `vander` in [tensor_ops.rs](/home/shinaoka/tensor4all/tenferro-rs/.worktrees/complex-real-unary-substrate/tenferro-linalg/src/primal/tensor_ops.rs). Those paths currently depend on CPU slice extraction. In parallel, [combine.rs](/home/shinaoka/tensor4all/tenferro-rs/.worktrees/complex-real-unary-substrate/tenferro-tensor/src/tensor/combine.rs) still limits `Tensor::cat` and `Tensor::stack` to `LogicalMemorySpace::MainMemory`.

PyTorch does not solve these ops by adding ad hoc linalg-only kernels first. It relies on lower tensor substrate and layout/materialization helpers, then composes higher-level linalg behavior on top. tenferro should follow the same direction.

## Problem

We need a GPU-generic way to materialize packed outputs from multiple source tensors without bouncing payload data through host memory. Without that substrate, any attempt to GPU-enable the remaining host-loop tensor ops in `tenferro-linalg` will either:

- introduce linalg-specific CUDA kernels in the wrong layer, or
- reintroduce CPU fallback/materialization paths.

Both conflict with the current layering and the no-ad-hoc-host-transfer rule.

## Options Considered

### Option 1: Add op-specific CUDA kernels in `tenferro-linalg`

This would move `cross`, `vander`, or `householder_product` forward quickly, but it hardcodes tensor packing/materialization policy inside Layer 4. That duplicates logic and does not help any other tensor-native packing use case.

Rejected.

### Option 2: Implement GPU `stack` and `cat` directly inside `tenferro-tensor` with local CUDA ownership

This avoids touching linalg, but it risks reintroducing runtime ownership split between tensor and prims layers. We already aligned on using shared lower runtime substrate rather than making tensor own a separate CUDA world.

Rejected.

### Option 3: Add a low-level pack substrate below tensor combine, then lift `cat` and `stack`

Add a same-dtype, same-device, contiguous-output pack entrypoint in Layer 0/2, use it to remove the main-memory restriction from `Tensor::cat`, then express `Tensor::stack` as `unsqueeze + cat` or reuse the same pack path. After that, rewrite remaining host-loop linalg tensor ops against tensor-native combine behavior.

Chosen.

## Chosen Architecture

### Layer 0 / Layer 2 split

- `tenferro-device` provides the raw GPU pack/materialization kernel for multiple source views into one contiguous destination.
- `tenferro-tensor` owns argument validation, dimension math, and public `Tensor::cat` / `Tensor::stack` semantics.

`tenferro-linalg` does not gain new low-level CUDA ownership.

### Why `cat` first

`cat` is the primitive materialization operation. `stack` is structurally just:

1. `unsqueeze` each input along a new axis
2. `cat` those views along that axis

That means `GPU cat` is the core substrate, while `GPU stack` is mostly API shaping and validation on top.

### Initial scope limits

The first tranche should stay deliberately narrow:

- same dtype only
- same logical memory space / same device only
- contiguous output in column-major order
- no mixed-device dispatch
- no aliasing write-back; always allocate a fresh output tensor

This is enough to unblock the current tensor/linalg cleanup without overbuilding.

## Expected Follow-On Uses

Once `GPU cat` and `GPU stack` exist, the next cleanup targets become realistic:

- `vander`
- `cross`
- `householder_product`

At that point, remaining host-loop paths can be evaluated one by one against reusable tensor-native combine/materialization rather than ad hoc CUDA kernels.

## Testing Strategy

The first acceptance tests should be:

- GPU `Tensor::cat` parity with CPU for simple column-major inputs
- GPU `Tensor::cat` with nontrivial concat axis
- GPU `Tensor::stack` parity with CPU, implemented through the same substrate
- source-level regression that `combine.rs` no longer rejects non-main-memory tensors for supported paths

Only after those pass should `tenferro-linalg` start consuming the new substrate.

