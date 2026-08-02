# Tensor Operations

This guide covers everyday tensor operations: elementwise math, shape changes,
broadcasting, reductions, and concrete backend execution. These operations are
available through different tensor APIs depending on whether you need
computation without autodiff, eager forward execution with optional
`backward()` or functional `grad`/`vjp`/`jvp`, or traced graph execution.

## Setup

```toml
[dependencies]
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-tensor = { path = "../crates/tenferro-tensor" }
tenferro-ad = { path = "../crates/tenferro-ad" }
```

## Layer Coverage

Choose the tensor API first, then choose the operation entry point.

| Layer | Start here when | Operation entry point | AD |
| --- | --- | --- | --- |
| `TypedTensor<T, R>` | No autodiff, scalar type known at compile time | direct typed accessors; `TypedTensorOpsExt` backend-explicit methods for dynamic-rank `TypedTensor<T>` | No |
| `Tensor` | No autodiff, dtype selected at runtime or passed through backend dispatch | `TensorOpsExt` backend-explicit methods | No |
| `EagerTensor` | Immediate execution in an `EagerRuntime`, optionally with `backward()` or functional `grad`/`vjp`/`jvp` | `EagerTensor` methods and associated functions | Yes, for tracked values |
| `TracedTensor` | Graph transforms, compilation, `grad`, `vjp`, `jvp`, or graph reuse | `TracedTensor` methods and associated functions | Yes, through graph transforms |

For most code without autodiff, start with `TypedTensor<T, R>` when the scalar type is
known in Rust, and use `Tensor` when the dtype must remain dynamic. `Tensor`
is the concrete runtime-dtype value type underneath eager and traced execution,
but AD workflows should normally enter through `EagerTensor` or
`TracedTensor`, not by using `Tensor` directly.

CUDA is a backend/device choice for supported operations on `Tensor`,
`EagerTensor`, and `TracedTensor`; it is not a separate tensor API.
Owned runtime tensors are compact column-major. Arbitrary strides live on
views. Operations that require compact storage may copy a view into compact
storage on the same device, without silently transferring CPU/GPU data.

## Common Concepts And Differences

All tensor APIs use the same basic tensor vocabulary: shape, rank, dtype or
scalar type, column-major dense storage for owned runtime tensors, explicit
backend execution, explicit CPU/GPU transfers, and NumPy-style broadcasting
where the operation supports it. The difference is which facts are represented
in Rust's type system and when computation happens.

| Capability | `TypedTensor<T, R>` | `Tensor` | `EagerTensor` | `TracedTensor` |
| --- | --- | --- | --- | --- |
| Scalar type in Rust type | Yes, `T` | No, runtime dtype enum | No, wraps `Tensor` | No, graph metadata |
| Rank in Rust type | Optional, `R = DynRank` or `Rank<N>` | Dynamic | Dynamic | Dynamic or symbolic metadata |
| Host typed slice access | Direct `&[T]` on host tensors | Fallible `as_slice::<T>()` | Through concrete data | Only after graph execution |
| Host iteration | Direct `iter()` and `iter_mut()` on host tensors | Fallible `iter::<T>()` and `iter_mut::<T>()` | Through concrete data | Not a concrete-data API |
| Backend math | Selected typed wrappers | Broad concrete backend API | Eager runtime API | Graph-building API |
| AD | No | No | Stateful reverse mode plus functional transforms for tracked values | Transform AD and graph reuse |

Use this distinction when reading operation examples:

- direct typed accessors are `TypedTensor`-specific conveniences;
- backend elementwise, structural, reduction, and dot operations are concrete
  execution concepts shared by the non-autodiff, eager, and traced APIs;
- AD-only operations such as `backward`, `grad`, `vjp`, and `jvp` live on the
  eager or traced layers.

## TypedTensor For Fixed Scalar Types

`TypedTensor<T, R = DynRank>` is the fixed-scalar-type runtime tensor. Use it
when ordinary Rust code knows the element type and you do not need AD.
`R` defaults to dynamic rank; use `Rank<N>` when the rank itself should be
validated and carried in the Rust type.

```rust
use tenferro_tensor::{Rank, TypedTensor};

let mut x = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
)
.unwrap();
assert_eq!(x.shape(), &[2, 2]);
assert_eq!(*x.get(&[1, 0]).unwrap(), 2.0);

*x.get_mut(&[0, 1]).unwrap() = 5.0;
assert_eq!(*x.get(&[1, 1]).unwrap(), 4.0);

let sum: f64 = x.host_data().unwrap().iter().copied().sum();
assert_eq!(sum, 12.0);

let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major(
    [2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
)
.unwrap();
assert_eq!(static_rank.rank(), 2);
```

The flat slice and host-buffer APIs expose the physical column-major host buffer.
They are useful for host-side inspection, small manual edits, and
interoperability with code that expects slices. They are not backend kernels
and they do not configure CPU parallelism.

## TypedTensor Backend Operations

For common typed math without autodiff, `TypedTensorOpsExt` provides selected
backend-explicit methods that accept dynamic-rank `TypedTensor<T>` values and
return typed results.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorMaskOpsExt, TypedTensorOpsExt};

let mut backend = CpuBackend::new();
let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
let y = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]).unwrap();

let sum = x.add(&y, &mut backend).unwrap();
let product = x.mul(&y, &mut backend).unwrap();
let total = product.reduce_sum(&[0], &mut backend).unwrap();
let mask = sum.compare(&product, CompareDir::Lt, &mut backend).unwrap();
let selected = mask.where_select(&sum, &product, &mut backend).unwrap();

assert_eq!(sum.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(total.as_slice().unwrap(), &[32.0]);
assert_eq!(mask.as_slice().unwrap(), &[false, true, true]);
assert_eq!(selected.as_slice().unwrap(), &[4.0, 7.0, 9.0]);
```

The current typed wrapper set covers:

- elementwise arithmetic and analytic operations: `add`, `sub`, `mul`, `div`,
  `pow`, `maximum`, `minimum`, `neg`, `abs`, `sign`, `conj`, `exp`, `log`,
  `sin`, `cos`, `tanh`, `sqrt`, `rsqrt`, `expm1`, and `log1p`;
- boolean-producing and selection operations: `compare`, `where_select`, and
  `clamp`;
- reduction and structural operations that preserve scalar type: `reduce_sum`,
  `reshape`, `transpose`, and `broadcast_in_dim`;
- rank-2 matrix multiplication through `matmul`.

These wrappers are a convenience layer over concrete tensor backend execution.
For backend-resident CUDA tensors or operation families not covered by the
typed wrappers, use the runtime-dtype `Tensor` path or the eager/traced layer
that matches the workflow. Prefer backend-aware typed wrappers for tensor
reductions and shape operations; reserve `host_data()` and `as_slice()` for
host-side inspection, small assertions, or interoperability with ordinary Rust
slice code.

## Map, Iteration, And Parallelism

`TypedTensor` exposes explicit host-buffer access for slice-style iteration:

```rust
use tenferro_tensor::TypedTensor;

let mut x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
for value in x.host_data_mut().unwrap() {
    *value *= 2.0;
}
assert_eq!(x.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
```

There is no public closure-style `TypedTensor::map` or `mapv` method in the
current public API. For host-only transformations, use `host_data`,
`host_data_mut`, `as_slice`, or `as_slice_mut`. For tensor math,
reductions, or shape operations that should use backend execution, use typed
wrappers or the runtime-dtype `Tensor`, `EagerTensor`, or `TracedTensor`
operation API.

Host-buffer iteration uses ordinary Rust slice iterators. Backend CPU parallelism is
controlled by the backend execution context, not by host slice iteration. See
[Parallelism and Caching](parallelism-and-caching.md) for CPU thread-count
controls.

## Scalar Types

`TypedTensor<T, R>` is generic enough to hold host data for many Rust element
types when you only need construction, shape/layout metadata, and host access.
The tenferro operation and dtype system is intentionally narrower. Backend
operations and runtime-dtype `Tensor` conversion require `T: TensorScalar`,
which is the supported scalar set:

- `f32` and `f64`;
- `i32` and `i64`;
- `bool`;
- `num_complex::Complex32` and `num_complex::Complex64`.

`bool` is a supported dtype for masks, comparisons, and selection. It does
not mean every numeric or analytic operation is valid for boolean tensors.
Arbitrary non-numeric Rust structs can be useful as host-side typed storage,
but they are not part of backend math, CUDA execution, AD, or the runtime-dtype
`Tensor` operation API.

Runtime-dtype `convert` is checked against tenferro's dtype-promotion lattice.
It accepts promotion-compatible conversions such as real-to-wider-real,
real-to-complex, integer-to-promoted numeric dtype, and `bool` to numeric dtype.
It returns a typed error for lossy projections such as float or complex to
integer, complex to real, integer to `bool`, and precision narrowing. Use
explicit `cast` when that lossy projection is intended.

## Runtime-DType Tensor Example

Use `Tensor` with a backend when you want direct computation without autodiff but the dtype
should remain dynamic.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = a.add(&b, &mut backend).unwrap();
let product = a.mul(&b, &mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Eager Forward And Autodiff Example

Use `EagerTensor` when the same immediate computation should stay in an
`EagerRuntime`. Create tracked variables when a scalar loss should accumulate
gradients or when a derivative transform should return another eager tensor.

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new().unwrap();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap()).unwrap();
let y = (&x * &x).reduce_sum(Some(&[0])).unwrap();

y.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Traced Tensor Example

Use `TracedTensor` when operations should build a graph first and execute later.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
let b = TracedTensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
let sum = (&a + &b).unwrap();
let product = (&a * &b).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&sum, &product]).unwrap();
let mut builder = Runtime::builder();
builder
    .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
    .unwrap();
let runtime = builder.build().unwrap();
let outputs = runtime.run_compiled(&program, &[]).unwrap();

assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

### Fallible operator chains

The overloaded operators are fallible: `a + b` returns a
`Result<TracedTensor, Error>`. Therefore `a + b + c` parses as `(a + b) + c`,
and the second operator receives a `Result`, not a tensor. It is not a
composable chain for dynamic tensors.

Unwrap each step with `?`, or use the explicit fallible method chain:

```rust
use tenferro_runtime::{Error, TracedTensor};

fn add_three(
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
) -> Result<TracedTensor, Error> {
    let ab = (a + b)?;
    let sum = (&ab + c)?;
    let canonical = a.add(b)?.add(c)?;
    let _ = canonical;
    Ok(sum)
}
```

Tenferro prioritizes robust error handling over concise chained operator
notation, so explicit fallible methods are the canonical choice for longer
sequences and code that needs deferred graph errors to remain visible.

## Elementwise Math Functions

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new().unwrap();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0]).unwrap()).unwrap();
let y = x.exp().unwrap();

let data = y.materialized().unwrap().as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
```

## Reshape And Transpose

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let reshaped = a.reshape(&[6], &mut backend).unwrap();
let transposed = a.transpose(&[1, 0], &mut backend).unwrap();

assert_eq!(reshaped.shape(), &[6]);
assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(transposed.shape(), &[3, 2]);
assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

## Explicit Broadcast

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new().unwrap();
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap()).unwrap();
let repeated = v.broadcast_in_dim(&[3, 2], &[0]).unwrap();

assert_eq!(repeated.shape(), &[3, 2]);
assert_eq!(repeated.materialized().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
```

## Reduce Over Axes

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
// Logical matrix:
// [[1.0, 3.0, 5.0],
//  [2.0, 4.0, 6.0]]
let row_sums = a.reduce_sum(&[1], &mut backend).unwrap();
let total = a.reduce_sum(&[0, 1], &mut backend).unwrap();

assert_eq!(row_sums.shape(), &[2]);
assert_eq!(row_sums.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(total.shape(), &[] as &[usize]);
// Rank-0 tensors hold one scalar element; as_slice() returns a length-1 slice.
assert_eq!(total.as_slice::<f64>().unwrap(), &[21.0]);
```
