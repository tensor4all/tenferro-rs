# Burn Framework Integration Design

## Motivation

A key use case for tenferro-rs is **hybrid NN + Tensor Network models**: neural network layers combined with tensor network operations (einsum, SVD, etc.), optimized end-to-end with gradient-based methods. [Burn](https://github.com/tracel-ai/burn) is a mature Rust deep learning framework with optimizers, module system, and multi-backend support. Rather than reimplementing NN infrastructure, tenferro-rs should integrate with Burn so that:

1. Tensor network operations appear as differentiable operations within Burn's computation graph
2. Burn's optimizers (Adam, SGD, etc.) can optimize parameters that flow through tensor network operations
3. Users define models mixing standard NN layers and tensor network contractions in a single `Module`

## Burn Architecture Summary

### Backend Trait System

Burn uses a trait-based backend abstraction. The core `Backend` trait composes many operation traits:

```rust
// burn-backend/src/backend/base.rs
pub trait Backend:
    FloatTensorOps<Self> + IntTensorOps<Self> + BoolTensorOps<Self>
    + ModuleOps<Self> + ActivationOps<Self> + ...
{
    type FloatTensorPrimitive: TensorMetadata + 'static;
    type FloatElem: Element;
    type Device: DeviceOps;
    // ...
}
```

Concrete backends (NdArray, CubeCL/Wgpu, Candle) implement these traits. Operations dispatch statically through the type system.

### Autodiff as a Decorator

Burn implements AD via a wrapper type `Autodiff<B, C>` that decorates any inner backend `B`:

```rust
// burn-autodiff/src/backend.rs
pub struct Autodiff<B, C = NoCheckpointing> { ... }

impl<B: Backend, C: CheckpointStrategy> Backend for Autodiff<B, C> {
    type FloatTensorPrimitive = AutodiffTensor<B>;  // wraps B's tensor + graph node
    type FloatElem = B::FloatElem;
    // Int/Bool tensors pass through unwrapped (no AD tracking)
}
```

Only float tensors are wrapped with AD tracking. The autodiff backend intercepts each float operation, records it on the computation graph, and delegates the actual computation to the inner backend.

### Custom Operations with Custom Gradients

Burn provides a `Backward<B, N>` trait (analogous to PyTorch's `autograd.Function`) for defining custom operations with custom backward passes:

```rust
// burn-autodiff/src/ops/backward.rs
pub trait Backward<B, const N: usize>: Send + Debug + 'static
where B: Backend
{
    type State: Clone + Send + Debug + 'static;

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    );
}
```

The pattern for adding a custom differentiable operation:

1. Define a custom `Backend` extension trait with the operation signature
2. Implement forward for concrete backends (NdArray, CubeCL, etc.)
3. Implement for `Autodiff<B, C>` with a `Backward` impl that computes gradients

A complete working example exists at `burn/examples/custom-cubecl-kernel/`:
- `src/lib.rs` — trait definition + high-level API
- `src/forward.rs` — forward kernel implementation
- `src/backward.rs` — `Backward<B, 3>` impl with gradient computation

## Tensor Data Interoperability

### Data Representations

| Aspect | Burn `TensorData` | tenferro `Tensor<T>` |
|--------|-------------------|----------------------|
| Storage | `Arc<Vec<u8>>` (bytes) | `Vec<T>` or external pointer (`DataBuffer<T>`) |
| Layout | Always contiguous, row-major | Arbitrary strides + offset |
| Strides | Not stored (implicit row-major) | Explicit `Vec<isize>` |
| Shape | `Vec<usize>` | `Vec<usize>` |
| GPU async | Minimal | `CompletionEvent` |

### Conversion Paths

**Burn → tenferro** (inside forward pass):
```
Burn FloatTensor<B>
  → B::float_into_data(tensor) → TensorData { bytes, shape, dtype }
  → bytes.as_slice::<f64>()
  → tenferro::Tensor::from_slice(slice, shape, MemoryOrder::RowMajor)
  → .into_contiguous(MemoryOrder::ColumnMajor)
```

**tenferro → Burn** (returning from forward/backward):
```
tenferro::Tensor<T>
  → .contiguous(RowMajor)       // ensure contiguous layout
  → .buffer().as_slice() → &[T]
  → TensorData::new(slice.to_vec(), shape)
  → B::float_from_data(data, device)
```

### Boundary Normalization

The canonical Burn bridge does **not** expose a zero-copy expert mode. Burn
data is treated as row-major at the boundary, normalized into tenferro's
column-major canonical tensors for computation, then materialized back to
row-major when exporting to Burn again.

This keeps the integration contract simple and avoids ambiguous reshape/order
behavior leaking across the bridge.

## Integration Architecture

### Black-Box rrule Bridge

tenferro operations are wrapped as opaque custom ops in Burn's computation graph. tenferro's `chainrules` rrule logic computes gradients; Burn's AD tape manages the overall graph.

```
Burn computation graph:
  ├── Linear(x)            ← Burn AD manages
  ├── ReLU(...)             ← Burn AD manages
  ├── einsum("ij,jk->ik")  ← Custom op: tenferro computes forward + backward
  │     forward:  Burn tensor → convert → tenferro einsum → convert back → Burn tensor
  │     backward: Burn grad → convert → tenferro rrule pullback → convert back → Burn grads
  ├── SVD(...)              ← Same pattern
  └── loss                  ← Burn AD manages
```

Key point: tenferro's internal `Tape` / `TrackedValue` are **not used**. Only the rrule pullback logic (the mathematical VJP formula) is called from within Burn's `Backward::backward()`.

### Crate Structure

The bridge crate `tenferro-burn`:

```
tenferro-burn/
├── Cargo.toml
└── src/
    ├── lib.rs          — TensorNetworkOps trait, public API, checked wrappers
    ├── convert.rs      — Burn tensor ↔ tenferro tensor conversion
    ├── forward.rs      — Forward implementations for concrete backends
    └── backward.rs     — Autodiff<B> implementations (Backward trait impls)
```

### Public API

```rust
use burn::tensor::Tensor;
use tenferro_burn::TensorNetworkOps;

// User-facing function
pub fn try_einsum<B: TensorNetworkOps, const D: usize>(
    subscripts: &str,
    inputs: Vec<Tensor<B, D>>,
) -> tenferro_burn::Result<Tensor<B, D>>;

// Used inside a Burn Module
struct HybridModel<B: TensorNetworkOps> {
    linear: burn::nn::Linear<B>,
    tn_core: Tensor<B, 3>,  // tensor network parameter, optimized by Burn
}

impl<B: TensorNetworkOps> HybridModel<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.linear.forward(x);
        tenferro_burn::einsum("ij,jkl,lm->im", vec![h, self.tn_core.clone(), ...])
    }
}
```

### Backend Extension Trait

```rust
pub trait TensorNetworkOps: burn::tensor::backend::Backend<FloatElem = f64> {
    /// N-ary einsum contraction.
    fn tn_einsum(
        subscripts: &str,
        inputs: Vec<FloatTensor<Self>>,
    ) -> FloatTensor<Self>;
}
```

### Backward Implementation Sketch

```rust
impl<B: TensorNetworkOps, C: CheckpointStrategy>
    TensorNetworkOps for Autodiff<B, C>
{
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct EinsumBackward { subscripts: String }

        impl<B: TensorNetworkOps> Backward<B, /* N */> for EinsumBackward {
            type State = (String, Vec<NodeId> /* checkpointed inputs */);

            fn backward(self, ops: Ops<Self::State, _>, grads: &mut Gradients, checkpointer: &mut Checkpointer) {
                let grad_output = grads.consume::<B>(&ops.node);

                // Convert Burn grad to tenferro tensor
                let tf_grad = burn_to_tenferro(grad_output);

                // Retrieve checkpointed inputs, convert to tenferro
                let tf_inputs: Vec<_> = /* ... */;

                // Call tenferro's rrule pullback (the VJP formula)
                let tf_input_grads = tenferro_einsum::einsum_rrule_pullback(
                    &self.subscripts, &tf_inputs, &tf_grad
                );

                // Convert back to Burn tensors and register
                for (parent, tf_grad) in ops.parents.iter().zip(tf_input_grads) {
                    if let Some(node) = parent {
                        grads.register::<B>(node.id, tenferro_to_burn(tf_grad));
                    }
                }
            }
        }

        // ... OpsPrep pipeline (tracked vs untracked) ...
    }
}
```

## Design Decisions

### Why not use tenferro's Tape inside Burn?

tenferro has its own AD engine (`tidu::Tape`), but using two separate AD tapes would create a discontinuity: Burn couldn't propagate gradients through the boundary. Instead, we extract only the **mathematical VJP rule** from tenferro's rrule implementations and call it within Burn's backward pass. This gives Burn full visibility of the gradient flow.

### Why a separate crate?

`tenferro-burn` depends on both Burn and tenferro-rs. Neither core library should depend on the other. The bridge crate is optional — users who don't need Burn don't pull in the dependency.

### Why start with NdArray backend?

NdArray is Burn's simplest CPU backend (pure Rust, no external dependencies). It's ideal for correctness testing. GPU backends (CubeCL) can be added later with the same trait pattern.

### Why checked helpers plus panic wrappers?

The bridge now keeps two layers on purpose:

- checked helpers (`try_einsum`, `try_burn_to_tenferro`, `try_tenferro_to_burn`)
  handle invalid subscripts, malformed nested einsum trees, and conversion
  failures explicitly
- convenience wrappers (`einsum`, `burn_to_tenferro`, `tenferro_to_burn`)
  panic only at the outer ergonomic boundary

This keeps library internals free of scattered `expect(...)` sites while
preserving the small POC-facing API.

## Implementation Phases

| Phase | Scope | Prerequisite |
|-------|-------|--------------|
| 0 | tenferro-rs POC implementation (einsum, einsum_rrule) | Current work |
| 1 | `convert.rs` — copy-based Burn ↔ tenferro tensor conversion | Phase 0 |
| 2 | `forward.rs` + `backward.rs` — einsum for NdArray backend | Phase 1 |
| 3 | SVD, QR, and other linalg operations | Phase 2 |
| 4 | CubeCL (GPU) backend support | Phase 2 |
| 5 | Zero-copy optimization (if profiling warrants) | Phase 2 |

## Verification

- **Gradient correctness**: Compare analytic gradients (Burn backward) against finite-difference numerical gradients for each operation
- **Round-trip data integrity**: Verify Burn → tenferro → Burn conversion preserves values exactly (bitwise for copy path)
- **End-to-end training**: Simple hybrid model (Linear + einsum) converges on a toy problem

## Related Issues

- [shinaoka/burn#1 — Add Complex Number Support for Burn](https://github.com/shinaoka/burn/issues/1):
  Complex tensor support is a prerequisite for scientific computing use cases
  (quantum mechanics, signal processing). The issue tracks a decorator-based
  `burn-complex` crate following upstream maintainer guidance. Complex number
  support in Burn is needed for tenferro interop with complex-valued tensor
  networks (e.g., `Tensor<Complex64>`).

## References

- Burn custom operation example: `burn/examples/custom-cubecl-kernel/`
- Burn autodiff internals: `burn/crates/burn-autodiff/src/ops/backward.rs`
- Burn backend trait: `burn/crates/burn-backend/src/backend/base.rs`
- tenferro chainrules-core: `tensor4all/chainrules-rs/crates/chainrules-core/src/lib.rs`
- tenferro tensor type: `tenferro-rs/tenferro-tensor/src/tensor/mod.rs`
