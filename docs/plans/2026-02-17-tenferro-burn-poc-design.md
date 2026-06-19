# tenferro-burn POC Design

**Date**: 2026-02-17
**Status**: Approved

## Context

We need a bridge crate (`tenferro-burn`) enabling hybrid NN + Tensor Network
models where Burn handles NN layers and optimization, and tenferro-rs provides
tensor network operations (einsum, SVD, etc.) with AD-compatible gradients.

This POC validates that the crate structure compiles and the Burn Backend
Extension Trait pattern works for tenferro operations. All function bodies
are `todo!()` — no runtime behavior.

## Design

### Crate Location

```
extension/tenferro-burn/
├── Cargo.toml
└── src/
    ├── lib.rs         — TensorNetworkOps trait, public einsum API, re-exports
    ├── convert.rs     — burn_to_tenferro / tenferro_to_burn conversion functions
    ├── forward.rs     — TensorNetworkOps impl for concrete backends
    └── backward.rs    — TensorNetworkOps impl for Autodiff<B, C> + Backward trait impl
```

Added to workspace `Cargo.toml` members as `"extension/tenferro-burn"`.

### Dependencies

```toml
[dependencies]
burn = { version = "0.17", default-features = false, features = ["ndarray", "autodiff"] }
tenferro-tensor = { path = "../../tenferro-tensor" }
tenferro-einsum = { path = "../../tenferro-einsum" }
tenferro-device = { path = "../../tenferro-device" }
tenferro-algebra = { path = "../../tenferro-algebra" }
```

Burn version pinned to latest stable. `ndarray` feature for the reference
backend; `autodiff` for `Autodiff<B>` and `Backward` trait.

### Core Trait (lib.rs)

```rust
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// Extension trait adding tensor network operations to any Burn backend.
pub trait TensorNetworkOps: Backend {
    /// N-ary einsum contraction.
    fn tn_einsum(
        subscripts: &str,
        inputs: Vec<FloatTensor<Self>>,
    ) -> FloatTensor<Self>;
}

/// High-level einsum API on Burn tensors.
pub fn einsum<B: TensorNetworkOps, const D: usize>(
    subscripts: &str,
    inputs: Vec<burn::tensor::Tensor<B, D>>,
) -> burn::tensor::Tensor<B, D> {
    todo!()
}
```

### Forward Implementation (forward.rs)

Concrete impl for NdArray backend (f64). Forward delegates to tenferro's
einsum internally (via convert.rs):

```rust
use burn_ndarray::NdArray;

impl TensorNetworkOps for NdArray<f64> {
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        // burn_to_tenferro → tenferro_einsum::traced_tensor::einsum → tenferro_to_burn
        todo!()
    }
}
```

Note: blanket `impl<B: Backend> TensorNetworkOps for B` would violate orphan
rules. We provide concrete impls per backend instead.

### Backward Implementation (backward.rs)

Autodiff wrapper that records the operation on Burn's computation graph:

```rust
use burn::backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: TensorNetworkOps, C: CheckpointStrategy> TensorNetworkOps for Autodiff<B, C> {
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        // Define EinsumBackward struct implementing Backward<B, N>
        // In backward(): call tenferro's einsum rrule pullback via convert
        // Use OpsPrep pipeline (tracked vs untracked)
        todo!()
    }
}
```

The `Backward` trait impl will call tenferro's `einsum_rrule` pullback logic
(from tenferro-einsum) to compute input gradients, converting between Burn
and tenferro tensor formats at the boundary.

### Conversion Functions (convert.rs)

```rust
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// Convert a Burn float tensor to a tenferro tensor.
pub fn burn_to_tenferro<B: Backend>(
    tensor: FloatTensor<B>,
) -> tenferro_tensor::Tensor<f64> {
    // B::float_into_data → TensorData → as_slice → tenferro::Tensor::from_slice
    todo!()
}

/// Convert a tenferro tensor to a Burn float tensor.
pub fn tenferro_to_burn<B: Backend>(
    tensor: tenferro_tensor::Tensor<f64>,
    device: &B::Device,
) -> FloatTensor<B> {
    // tensor.buffer().as_slice() → TensorData → B::float_from_data
    todo!()
}
```

### f64 Support Confirmation

Burn fully supports f64 across all backends:
- `NdArray<f64>` works out of the box (`FloatNdArrayElement` implemented for f64)
- No feature flags restrict f64
- Default is f32 but f64 is a first-class element type

### Success Criteria

- `cargo build -p tenferro-burn` compiles without errors
- All function bodies are `todo!()`
- Trait hierarchy compiles: `TensorNetworkOps` for `NdArray<f64>` and `Autodiff<NdArray<f64>>`
- Workspace CI (`cargo test --workspace`) passes (no tests to run, just compilation)

### What This POC Does NOT Include

- Runtime behavior (all `todo!()`)
- SVD or other linalg operations (einsum only)
- GPU backends (NdArray only for concrete impl)
- f32 support (f64 only, matching tenferro's primary scalar type)
- Zero-copy optimization (copy-based conversion API only)
