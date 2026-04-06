# tenferro

Traced tensor frontend for the `tenferro-rs` v2 workspace.

`tenferro` is the main user-facing crate for standard dense numeric computation.
It owns the lazy graph surface (`TracedTensor`), the execution engine
(`Engine<B>`), StableHLO-style lowering, execution-IR compilation, public
einsum helpers, public multi-output linalg helpers, and first-order AD entry
points.

## Public Surface

- `TracedTensor`
- `Engine`
- `einsum::einsum` and `einsum::einsum_with`
- free linalg helpers such as `svd`, `qr`, `eigh`, `solve`, `cholesky`, and
  `triangular_solve`
- `TracedTensor::grad`, `TracedTensor::jvp`, `TracedTensor::vjp`
- re-exported dense runtime types from `tenferro-tensor`:
  `Tensor`, `TypedTensor`, `DType`, `CpuBackend`

## Example

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn main() {
    let a = TracedTensor::from_tensor(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_tensor(f64_tensor(
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
- GPU support is partial and experimental.
- The crate no longer exposes the older runtime-installation and dynamic-carrier
  API family.
