# Core Concepts

tenferro separates tensor data, execution timing, automatic differentiation,
and device location. That separation is the main design point: users can stay
in typed tensor code for ordinary numeric work without autodiff, move to eager
execution for PyTorch-like forward-and-backward training loops, or move to
traced graphs for JAX-like `grad`, `vjp`, and `jvp` workflows plus compile/run
reuse.

## The Three Axes

| Axis | What it controls | User-facing choices |
| --- | --- | --- |
| Data layer | The value you pass around | `TypedTensor<T>`, `Tensor`, `EagerTensor`, `TracedTensor` |
| Execution model | When operations run | Direct, eager, traced compile/run |
| Backend/device | Where operations run | `CpuBackend` or `tenferro_gpu::CudaBackend` |

CUDA is not a separate tensor type. The same concrete, eager, and traced APIs
can run supported operations on CUDA tensors when data is explicitly uploaded
to a CUDA backend.

## Tensor Layers

| Layer | Role | Good fit |
| --- | --- | --- |
| `TypedTensor<T>` | Concrete tensor with compile-time scalar type | Most typed numeric code, typed host data, typed linalg/einsum |
| `Tensor` | Concrete tensor with runtime dtype | Dynamic dtype workflows, backend dispatch, CPU/CUDA values |
| `EagerTensor` | Concrete tensor in an eager runtime, with optional gradient tracking | Immediate forward execution; reverse-mode AD on scalar losses when tracked |
| `TracedTensor` | Graph-building tensor | `grad`, `vjp`, and `jvp` on traced graphs, graph optimization, repeated execution |

Use the smallest layer that matches the job. AD is optional, and many projects
should never leave the concrete tensor APIs.

## Memory Model

tenferro stores dense tensors as contiguous column-major buffers. The leftmost
dimension varies fastest in memory. This matches Fortran, Julia, MATLAB, and
LAPACK-oriented workflows, and makes trailing batch axes natural for batched
linear algebra and contractions.

<!-- snippet-source: crates/tenferro-runtime/examples/column_major_memory.rs -->
```rust
use tenferro_runtime::{Tensor, TypedTensor};

fn main() {
    let typed =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
    assert_eq!(typed.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let dynamic =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
    assert_eq!(
        dynamic.as_slice::<f64>().unwrap(),
        typed.as_slice().unwrap()
    );
}
```
<!-- end-snippet-source -->

Use `from_vec_col_major` for tensor construction. Data copied from PyTorch,
NumPy, JAX, or C-style examples must be explicitly reordered at the boundary
before entering tenferro.

## Direct Tensor Execution

`Tensor` operations run immediately through an explicit backend. Use this for
ordinary tensor computation without autodiff when runtime dtype is useful.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let c = tensor::matmul(&a, &b, &mut backend).unwrap();
assert_eq!(c.shape(), &[2, 2]);
```

`TypedTensor<T>` is the compile-time scalar type layer. It is useful when the
project already knows it is working with `f64`, `f32`, complex values, or
another supported scalar type.

Einsum is provided by the `tenferro-einsum` standard extension. Traced code
uses `tenferro_einsum::traced_tensor::einsum` and registers
`tenferro_einsum::register_runtime` on the executor.

## Eager Execution And Backward

`EagerTensor` wraps concrete values in an `EagerRuntime`. Each operation
computes or submits immediately and returns a concrete tensor handle. CPU
values are host-readable; CUDA-resident values require explicit download before
host inspection. If a tensor is a tracked variable, eager operations also
record reverse-mode state so a scalar loss can call `backward()` and accumulate
gradients.

This is not the forward-mode AD/JVP API. Use `TracedTensor` for `grad`, `vjp`,
`jvp`, and HVP via composition on traced graphs.

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap()).unwrap();
let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
loss.backward().unwrap();

assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);
```

## Traced Graph Execution

`TracedTensor` operations are lazy. They build a graph. A `GraphCompiler`
lowers that graph into a reusable program, and a `GraphExecutor<B>` runs the
program on a backend.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
let sum = (&a + &b).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&sum).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let result = executor.run(&program).unwrap();

assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
```

Traced mode is the right API for `grad`, `vjp`, `jvp`, and HVP via composition
on traced graphs, symbolic inputs, graph optimization, and repeated execution.
Core primitive AD rules are available by default. Extension operation families
that provide AD rules, such as `tenferro-linalg`, require enabling that crate's
`autodiff` feature and registering the extension rule set with
`with_extension_rules`.
