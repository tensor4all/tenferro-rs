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

Basic arithmetic on traced tensors (add, mul, neg):

```rust
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn main() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));

    // Elementwise operations use operator overloading on references
    let sum = &a + &b;
    let product = &a * &b;
    let negated = -&a;

    let mut engine = Engine::new(CpuBackend::new());
    let mut sum_val = sum;
    let mut prod_val = product;
    let mut neg_val = negated;
    assert_eq!(sum_val.eval(&mut engine).unwrap().shape(), &[3]);
    assert_eq!(prod_val.eval(&mut engine).unwrap().shape(), &[3]);
    assert_eq!(neg_val.eval(&mut engine).unwrap().shape(), &[3]);
}
```

N-ary einsum with automatic contraction tree optimization:

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
        vec![3, 4],
        (1..=12).map(|x| x as f64).collect(),
    ));
    let c = TracedTensor::from_tensor(f64_tensor(
        vec![4, 2],
        (1..=8).map(|x| x as f64).collect(),
    ));

    // Three-operand chain: A[ij] * B[jk] * C[kl] -> D[il]
    let mut engine = Engine::new(CpuBackend::new());
    let mut d = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();

    let result = d.eval(&mut engine).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
}
```

SVD decomposition via the linalg API:

```rust
use tenferro::engine::Engine;
use tenferro::traced::{eval_all, TracedTensor};
use tenferro::{svd, CpuBackend, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn main() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]));
    let (mut u, mut s, mut vt) = svd(&a);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut u, &mut s, &mut vt]).unwrap();

    assert_eq!(results[0].shape(), &[2, 2]); // U
    assert_eq!(results[1].shape(), &[2]);    // singular values
    assert_eq!(results[2].shape(), &[2, 2]); // V^T
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

## Documentation

- [Getting Started](docs/getting-started/) — tutorials and worked examples
- [Architecture](docs/architecture/) — design rationale for each subsystem
- [Specification](docs/spec/) — normative specs for ops, backends, and AD
- [Implementation Design](docs/design/) — implementation-focused design notes

## Design Notes

All operations on `TracedTensor` are lazy: they build a computation graph
instead of executing immediately. Evaluation happens only when `eval()` is
called, at which point the graph is compiled, optimized, and dispatched to
a backend (currently CPU via faer/BLAS).

The API naming largely follows PyTorch (`solve`, `triangular_solve`, `eigh`,
etc.), while the AD backend draws on ideas from JAX's trace-and-lower
architecture and ChainRules.jl's derivative contract. The key original contributions are:

- **Primitive-agnostic AD engine.** The stack is split into four
  independent layers — `computegraph` (graph IR), `chainrules-core` (AD
  trait contract), `tidu` (differentiate/transpose transforms), and
  `tenferro` (concrete tensor primitives). The AD engine knows nothing
  about specific operations; adding a new differentiable op only requires
  implementing a trait, with no changes to the AD infrastructure itself.
- **Semiring-generic graph execution.** The same graph infrastructure
  supports custom algebras beyond standard arithmetic. For example,
  tropical semiring einsum can be executed on the same graph IR without
  AD, while user-defined numeric types can opt into AD by implementing
  the `PrimitiveOp` trait.

Key types:

- `Tensor` — concrete dense runtime value (col-major storage).
- `TracedTensor` — lazy graph-aware wrapper for computation and AD.
- `Engine` — holds backend + compile cache; triggers evaluation.
- Public einsum: `tenferro::einsum::einsum(...)`.
- Multi-output linalg: free functions `tenferro::svd(...)`, `tenferro::qr(...)`, etc.
See [`docs/architecture/`](docs/architecture/) for design rationale,
[`docs/spec/`](docs/spec/) for normative specifications, and
[`docs/design/`](docs/design/) for implementation-focused design notes.
