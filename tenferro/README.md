# tenferro

User-facing tensor facade for the `tenferro-rs` v2 workspace.

`tenferro` is the main public crate for standard dense numeric computation.
It exposes eager execution through `EagerTensor` and `EagerRuntime`, including
scalar-loss reverse-mode accumulation via `backward()`, and it exposes
traced, transform-oriented AD through `TracedTensor` with `grad`, `vjp`,
`jvp`, and HVP composition. The crate also owns explicit graph compilation
through `GraphCompiler`, backend execution through `GraphExecutor<B>`,
StableHLO-style lowering, execution-IR compilation, public einsum helpers,
and public multi-output linalg helpers.

## Public Surface

- `TracedTensor`
- `GraphCompiler`
- `GraphExecutor`
- `EagerTensor`
- `EagerRuntime`
- `traced_tensor::einsum` and `traced_tensor::einsum_with`
- `eager_tensor::einsum`
- `tensor::einsum` and `tensor::einsum_owned`
- `typed_tensor::einsum`
- traced linalg helpers under `traced_tensor`, such as `svd`, `qr`, `eigh`,
  `solve`, `cholesky`, and `triangular_solve`
- `EagerTensor::backward`, `EagerTensor::clear_grad`, `EagerRuntime::clear_grads`
- `TracedTensor::grad`, `TracedTensor::jvp`, `TracedTensor::vjp`
- re-exported dense runtime types from `tenferro-tensor`:
  `Tensor`, `TypedTensor`, `DType`, `CpuBackend`

## Eager Example

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
let loss = (&x * &x).reduce_sum(&[0]).unwrap();
loss.backward().unwrap();

assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
```

## Traced Example

```rust
use tenferro::traced_tensor::einsum;
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
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

    let mut compiler = GraphCompiler::new();
    let c = einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
    let program = compiler.compile(&c).unwrap();
    let out = GraphExecutor::new(CpuBackend::new()).run(&program).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
}
```

## Notes

- `Tensor` is the concrete dense runtime value at graph boundaries.
- `TracedTensor` is the graph-aware lazy wrapper.
- CUDA GPU support is available through the feature-gated CubeCL backend.
- The crate no longer exposes the older runtime-installation and dynamic-carrier
  API family.
