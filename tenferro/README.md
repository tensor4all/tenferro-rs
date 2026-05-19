# tenferro

User-facing tensor facade for the `tenferro-rs` v2 workspace.

`tenferro` is the main public crate for standard dense numeric computation.
It exposes eager execution through `EagerTensor` and `EagerContext`, including
scalar-loss reverse-mode accumulation via `backward()`, and it exposes
traced, transform-oriented AD through `TracedTensor` with `grad`, `vjp`,
`jvp`, and HVP composition. The crate also owns the execution engine
(`Engine<B>`), StableHLO-style lowering, execution-IR compilation, public
einsum helpers, and public multi-output linalg helpers.

## Public Surface

- `TracedTensor`
- `Engine`
- `EagerTensor`
- `EagerContext`
- `traced_tensor::einsum` and `traced_tensor::einsum_with`
- `eager_tensor::einsum`
- `tensor::einsum` and `tensor::einsum_owned`
- `typed_tensor::einsum`
- traced linalg helpers under `traced_tensor`, such as `svd`, `qr`, `eigh`,
  `solve`, `cholesky`, and `triangular_solve`
- `EagerTensor::backward`, `EagerTensor::clear_grad`, `EagerContext::clear_grads`
- `TracedTensor::grad`, `TracedTensor::jvp`, `TracedTensor::vjp`
- re-exported dense runtime types from `tenferro-tensor`:
  `Tensor`, `TypedTensor`, `DType`, `CpuBackend`

## Eager Example

```rust
use tenferro::{EagerTensor, Tensor};

let x = EagerTensor::requires_grad(Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]));
let loss = (&x * &x).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();

assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
```

## Traced Example

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn main() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));

    let mut engine = Engine::new(CpuBackend::new());
    let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
    let out = c.eval(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
}
```

## Notes

- `Tensor` is the concrete dense runtime value at graph boundaries.
- `TracedTensor` is the graph-aware lazy wrapper.
- CUDA GPU support is available through the feature-gated CubeCL backend.
- The crate no longer exposes the older runtime-installation and dynamic-carrier
  API family.
