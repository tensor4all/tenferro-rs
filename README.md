# tenferro-rs

General-purpose dense tensor computation in Rust.

The current repository is the v2 graph-based implementation. The active
workspace is a small 6-crate core built around lazy traced tensors,
StableHLO-style lowering, and first-order AD for the standard dense numeric
path.

## Status

- `tenferro` is the main user-facing crate for lazy traced computation.
- `tenferro-tensor` owns concrete dense tensor values, execution backends, and
  CPU kernels.
- CPU execution is implemented and tested.
- GPU support is partial and experimental. CUDA symbols exist, but coverage is
  incomplete and ROCm remains mostly stubbed.
- Tensor storage is dense, contiguous, and column-major.

Legacy facade, internal, FFI, and extension crates were removed from the main
tree on April 6, 2026 to keep the repository aligned with the current 6-crate
workspace. Historical references may still exist in archived plan documents
under `docs/plans/`.

## Workspace Crates

| Crate | Role |
| --- | --- |
| `tenferro` | Traced frontend: `Engine`, `TracedTensor`, public einsum and linalg APIs, VJP/JVP |
| `tenferro-tensor` | Dense runtime tensors, backend traits, CPU backend, GPU backend stubs |
| `tenferro-einsum` | Subscripts, contraction trees, and fragment-building utilities |
| `tenferro-ops` | Graph op vocabulary (`StdTensorOp`, `SemiringOp`) and AD rule implementations |
| `tenferro-algebra` | Semiring/algebra traits |
| `tenferro-device` | Shared device and error infrastructure |

## Quick Start

Add the main crate from a local checkout:

```toml
[dependencies]
tenferro = { path = "../tenferro-rs/tenferro" }
```

Build a lazy einsum graph and evaluate it:

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

Compute a gradient through the traced graph:

```rust
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn main() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let loss = x.reduce_sum(&[0]);
    let mut grad = loss.grad(&x).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let gx = grad.eval(&mut engine).unwrap();

    assert_eq!(gx.shape(), &[4]);
}
```

## Design Notes

- `Tensor` is the concrete dense runtime value.
- `TracedTensor` is the lazy graph-aware wrapper used for standard-algebra
  computation and AD.
- Public einsum uses the free function `tenferro::einsum::einsum(...)`.
- Multi-output linalg operations are free functions such as `tenferro::svd(...)`
  and `tenferro::qr(...)`.

See [`docs/design/`](docs/design/) for local design notes and
`../tensor4all-meta/docs/design-v2/` for the current v2 planning documents.
