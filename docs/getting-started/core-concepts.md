# Core Concepts

tenferro separates tensor data, execution timing, automatic differentiation,
and device location. That separation is the main design point: users can stay
in typed tensor code for ordinary numeric work without autodiff, move to eager
execution for PyTorch-like forward-and-backward training loops or functional
`grad`, `vjp`, and `jvp` transforms, or move to traced graphs for compiled
transform workflows plus compile/run reuse.

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
| `EagerTensor` | Concrete tensor in an eager runtime, with optional gradient tracking | Immediate forward execution; `backward()` on scalar losses and `EagerRuntime` functional transforms when tracked |
| `TracedTensor` | Graph-building tensor | Compiled `grad`, `vjp`, and `jvp` on traced graphs, graph optimization, repeated execution |

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

Read values back with `as_slice::<T>()` on `Tensor`, or `as_slice()` on
`TypedTensor<T>`. Both return `Result<&[T]>` and yield `Err` on a dtype
mismatch, not `Option`.

## Direct Tensor Execution

`Tensor` operations run immediately through an explicit backend. Use this for
ordinary tensor computation without autodiff when runtime dtype is useful.

<!-- snippet-source: crates/tenferro-runtime/examples/direct_tensor_execution.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    let c = a.matmul(&b, &mut backend)?;
    assert_eq!(c.shape(), &[2, 2]);

    Ok(())
}
```
<!-- end-snippet-source -->

`TypedTensor<T>` is the compile-time scalar type layer. It is useful when the
project already knows it is working with `f64`, `f32`, complex values, or
another supported scalar type.

Einsum is provided by the `tenferro-einsum` standard extension. Traced code
uses `TraceContextEinsumExt` and installs `tenferro_einsum::extension_module`
on the runtime.

## Eager Execution And Autodiff

`EagerTensor` wraps concrete values in an `EagerRuntime`. Each operation
computes or submits immediately and returns a concrete tensor handle. CPU
values are host-readable; CUDA-resident values require explicit download before
host inspection. If a tensor is a tracked variable, eager operations also
record reverse-mode state so a scalar loss can call `backward()` and accumulate
gradients.

When the derivative itself should be returned as an eager tensor instead of
accumulated into a gradient slot, call `EagerRuntime::grad`,
`EagerRuntime::vjp`, or `EagerRuntime::jvp`. These functional eager transforms
can be composed for HVP-style workflows. Use `TracedTensor` when the derivative
workflow should be compiled, optimized as a graph, or reused across runs.

<!-- snippet-source: crates/tenferro-ad/examples/eager_backward.rs -->
```rust
use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = EagerRuntime::new().unwrap();
    let x = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let loss = x.mul(&x)?.reduce_sum(Some(&[0]))?;
    loss.backward()?;

    assert_eq!(x.grad()?.unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

Here `Tensor` is `tenferro_ad`'s re-export of `tenferro_runtime::Tensor` — the
same type as the concrete tensor used in the sections above, so values move
between the eager and direct APIs without conversion.

## Traced Graph Execution

`TracedTensor` operations are lazy. They build a graph. A `GraphCompiler`
lowers that graph into a reusable program, and `Runtime::run_compiled` runs the
program on registered runtime engines.

<!-- snippet-source: crates/tenferro-runtime/examples/traced_graph_execution.rs -->
```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime() -> Result<Runtime, Box<dyn std::error::Error>> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(
        &CpuBackend::new(),
    )?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0])?;
    let sum = (&a + &b)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&sum)?;
    let runtime = cpu_runtime()?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.len(), 1);
    let result = outputs.pop().unwrap();

    assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 6.0]);

    Ok(())
}
```
<!-- end-snippet-source -->

### Fallible operator chaining

Tensor operator overloads are deliberately fallible: each `+` produces a
`Result<TracedTensor, Error>`. Consequently, `a + b + c` parses as
`(a + b) + c`, and the second `+` would receive the first `Result` rather
than a tensor. The chained expression therefore does not compose for dynamic
tensors.

Use `?` at each step, or use the explicit fallible methods when a longer
operation sequence is clearer:

```rust
use tenferro_runtime::{Error, TracedTensor};

fn add_three(
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
) -> Result<TracedTensor, Error> {
    let ab = (a + b)?;
    let sum = (&ab + c)?;

    let method_chain = a.add(b)?.add(c)?;
    let _ = method_chain;
    Ok(sum)
}
```

Tenferro prioritizes robust error handling over the conciseness of chained
operator notation. The explicit methods are the canonical form for code that
needs to make fallible control flow and deferred graph errors easy to review.

Traced mode is the right API when `grad`, `vjp`, `jvp`, or HVP-style
composition should run on traced graphs with symbolic inputs, graph
optimization, and repeated execution.
Core primitive AD rules are available by default. Extension operation families
that provide AD rules, such as `tenferro-linalg`, require enabling that crate's
`autodiff` feature and registering the extension rule set with
`with_semantic_extension_rules`.
