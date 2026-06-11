# Tensor Operations

This guide covers everyday tensor operations: elementwise math, shape changes,
broadcasting, reductions, and concrete backend execution. These operations are
available through different tensor APIs depending on whether you need
computation without autodiff, eager forward execution with optional
`backward()` on scalar losses, or traced graph execution.

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
| `TypedTensor<T, R>` | No autodiff, scalar type known at compile time | direct typed accessors; selected `tenferro_runtime::typed_tensor` wrappers for dynamic-rank `TypedTensor<T>` | No |
| `Tensor` | No autodiff, dtype selected at runtime or passed through backend dispatch | `tenferro_runtime::tensor` functions with an explicit backend | No |
| `EagerTensor` | Immediate execution in an `EagerRuntime`, optionally with `backward()` on scalar losses | `tenferro_ad::eager_tensor` functions and methods | Yes, for tracked values |
| `TracedTensor` | Graph transforms, compilation, `grad`, `vjp`, `jvp`, or graph reuse | `tenferro_runtime::traced_tensor` functions and methods | Yes, through graph transforms |

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
| AD | No | No | Optional reverse-mode for tracked values | Transform AD and graph reuse |

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
);
assert_eq!(x.shape(), &[2, 2]);
assert_eq!(x.get(&[1, 0]), &2.0);

*x.get_mut(&[0, 1]) = 5.0;
assert_eq!(x.try_get(&[1, 1]), Some(&4.0));

let sum: f64 = x.iter().copied().sum();
assert_eq!(sum, 12.0);

let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major(
    [2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
);
assert_eq!(static_rank.rank(), 2);
```

The flat slice and iterator APIs expose the physical column-major host buffer.
They are useful for host-side inspection, small manual edits, and
interoperability with code that expects slices. They are not backend kernels
and they do not configure CPU parallelism.

## TypedTensor Backend Operations

For common typed math without autodiff, `tenferro_runtime::typed_tensor` provides selected
wrappers that accept dynamic-rank `TypedTensor<T>` values and return typed
results.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{typed_tensor, CompareDir, TypedTensor};

let mut backend = CpuBackend::new();
let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]);
let y = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]);

let sum = typed_tensor::add(&x, &y, &mut backend).unwrap();
let product = typed_tensor::mul(&x, &y, &mut backend).unwrap();
let total = typed_tensor::reduce_sum(&product, &[0], &mut backend).unwrap();
let mask = typed_tensor::compare(&sum, &product, CompareDir::Lt, &mut backend).unwrap();
let selected = typed_tensor::where_select(&mask, &sum, &product, &mut backend).unwrap();

assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice(), &[4.0, 10.0, 18.0]);
assert_eq!(total.as_slice(), &[32.0]);
assert_eq!(mask.as_slice(), &[false, true, true]);
assert_eq!(selected.as_slice(), &[4.0, 7.0, 9.0]);
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
reductions and shape operations; reserve `iter()` and `as_slice()` for
host-side inspection, small assertions, or interoperability with ordinary Rust
slice code.

## Map, Iteration, And Parallelism

`TypedTensor` exposes slice-style host iteration:

```rust
use tenferro_tensor::TypedTensor;

let mut x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]);
for value in x.iter_mut() {
    *value *= 2.0;
}
assert_eq!(x.as_slice(), &[2.0, 4.0, 6.0]);
```

There is no public closure-style `TypedTensor::map` or `mapv` method in the
current public API. For host-only transformations, use `iter`, `iter_mut`,
`as_slice`, `host_data_mut`, or `as_physical_slice_mut`. For tensor math,
reductions, or shape operations that should use backend execution, use typed
wrappers or the runtime-dtype `Tensor`, `EagerTensor`, or `TracedTensor`
operation API.

Host iterators are ordinary Rust slice iterators. Backend CPU parallelism is
controlled by the backend execution context, not by `iter()` itself. See
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

## Runtime-DType Tensor Example

Use `Tensor` with a backend when you want direct computation without autodiff but the dtype
should remain dynamic.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);

let sum = tensor::add(&a, &b, &mut backend).unwrap();
let product = tensor::mul(&a, &b, &mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Eager Forward And Backward Example

Use `EagerTensor` when the same immediate computation should stay in an
`EagerRuntime`. Create tracked variables when a scalar loss should accumulate
gradients.

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
let y = (&x * &x).reduce_sum(&[0]).unwrap();

y.backward().unwrap();
assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
```

## Traced Tensor Example

Use `TracedTensor` when operations should build a graph first and execute later.

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{traced_tensor, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
let b = TracedTensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);
let sum = traced_tensor::add(&a, &b);
let product = traced_tensor::mul(&a, &b);

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&sum, &product]).unwrap();
let mut executor = GraphExecutor::new(CpuBackend::new());
let outputs = executor.run_many(&program).unwrap();

assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
```

## Elementwise Math Functions

```rust
use tenferro_ad::{eager_tensor, EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0]));
let y = eager_tensor::exp(&x).unwrap();

let data = y.data().as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
```

## Reshape And Transpose

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let reshaped = tensor::reshape(&a, &[6], &mut backend).unwrap();
let transposed = tensor::transpose(&a, &[1, 0], &mut backend).unwrap();

assert_eq!(reshaped.shape(), &[6]);
assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(transposed.shape(), &[3, 2]);
assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
```

## Explicit Broadcast

```rust
use tenferro_ad::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]));
let repeated = v.broadcast_in_dim(&[3, 2], &[0]).unwrap();

assert_eq!(repeated.data().shape(), &[3, 2]);
assert_eq!(repeated.data().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
```

## Reduce Over Axes

```rust
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{tensor, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
// Logical matrix:
// [[1.0, 3.0, 5.0],
//  [2.0, 4.0, 6.0]]
let row_sums = tensor::reduce_sum(&a, &[1], &mut backend).unwrap();
let total = tensor::reduce_sum(&a, &[0, 1], &mut backend).unwrap();

assert_eq!(row_sums.shape(), &[2]);
assert_eq!(row_sums.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(total.shape(), &[] as &[usize]);
// Rank-0 tensors hold one scalar element; as_slice() returns a length-1 slice.
assert_eq!(total.as_slice::<f64>().unwrap(), &[21.0]);
```
