# tenferro

User-facing tensor facade for the `tenferro-rs` v2 workspace.

`tenferro` is the main public crate for tensor execution infrastructure.
It exposes eager execution through `EagerTensor` and `EagerRuntime`, including
scalar-loss reverse-mode accumulation via `backward()`, and it exposes
traced, transform-oriented AD through `TracedTensor` with `grad`, `vjp`,
`jvp`, and HVP composition. The crate also owns explicit graph construction
through `GraphCompiler`, backend execution through `GraphExecutor<B>`, and
selected dense runtime type reexports. Low-level extension runtime dispatch and
extension cache storage live in `tenferro-runtime` and are reexported here for
application ergonomics.

Operation families are separate extension crates. `tenferro` must not depend on
`tenferro-einsum`, `tenferro-linalg`, or `tenferro-fft`, and it must never add
facade paths such as `tenferro::einsum`, `tenferro::linalg`, or
`tenferro::fft`. Users import the extension crate directly and register its
runtime explicitly.

## Public Surface

- `TracedTensor`
- `GraphCompiler`
- `GraphExecutor`
- `EagerTensor`
- `EagerRuntime`
- `EagerTensor::backward`, `EagerTensor::clear_grad`, `EagerRuntime::clear_grads`
- `TracedTensor::grad`, `TracedTensor::jvp`, `TracedTensor::vjp`
- `extension::*` registration and application APIs
- re-exported extension runtime/cache types from `tenferro-runtime`
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
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};
use tenferro_einsum::einsum;

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
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_einsum::register_runtime).unwrap();
    let out = executor.run(&program).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
}
```

## Notes

- `Tensor` is the concrete dense runtime value at graph boundaries.
- `TracedTensor` is the graph-aware lazy wrapper.
- Einsum, linalg, and FFT are standard extension crates with direct APIs such
  as `tenferro_einsum::einsum` and `tenferro_linalg::svd`.
- `tenferro` does not depend on standard extensions. Register each extension
  runtime explicitly on the `GraphExecutor` that will execute it.
- The default feature set includes `autodiff` and `cpu-faer`. Primal-only builds
  use `default-features = false` plus an explicit CPU backend feature.
- CUDA GPU support is available through the feature-gated CubeCL backend.
