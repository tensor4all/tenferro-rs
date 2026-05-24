# Einsum

`einsum` is a standard extension, not part of `tenferro` core. Add the
`tenferro-einsum` crate and call it explicitly as `tenferro_einsum::einsum`.
Compiled execution also requires explicit runtime registration.

```toml
[dependencies]
tenferro = { path = "../tenferro" }
tenferro-einsum = { path = "../tenferro-einsum" }
```

## Traced Matrix Multiply

Use the traced route when einsum should be part of a graph compiled by
`GraphCompiler` and executed by `GraphExecutor`.

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};

let a = TracedTensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);
let b = TracedTensor::from_vec_col_major(
    vec![3, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

let mut compiler = GraphCompiler::new();
let c = tenferro_einsum::einsum(&mut compiler, &[&a, &b], "ij,jk->ik").unwrap();
let program = compiler.compile(&c).unwrap();

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_einsum::register_runtime).unwrap();
let result = executor.run(&program).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## EagerTensor

With the `autodiff` feature, `tenferro-einsum` also exposes immediate
`EagerTensor` execution.

```rust
use tenferro::{EagerRuntime, Tensor};

let ctx = EagerRuntime::new();
let u = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]));
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]));

let outer = tenferro_einsum::eager_tensor::einsum(&[&u, &v], "i,j->ij").unwrap();
let diag = tenferro_einsum::eager_tensor::einsum(&[&v], "i->ii").unwrap();

assert_eq!(outer.data().shape(), &[2, 3]);
assert_eq!(diag.data().shape(), &[3, 3]);
```

## Optimization Controls

The default policy chooses an N-ary contraction order automatically. Advanced
users can pass an explicit strategy through `tenferro_einsum::einsum_with`.
The public optimizer surface is limited to the types needed to express that
choice: `EinsumOptimize`, `ContractionTree`, `ContractionOptimizerOptions`,
`Subscripts`, `NestedEinsum`, and `EinsumSubscripts`.

```rust
use tenferro::{GraphCompiler, TracedTensor};
use tenferro_einsum::EinsumOptimize;

let a = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
let b = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);

let mut compiler = GraphCompiler::new();
let c = tenferro_einsum::einsum_with(
    &mut compiler,
    &[&a, &b],
    "ij,jk->ik",
    EinsumOptimize::False,
).unwrap();

assert_eq!(c.shape, vec![2, 2]);
```

## Cache Management

Einsum uses the shared extension cache infrastructure from
`tenferro-runtime`, re-exported by `tenferro`. Compile-time extension caches
live on `GraphCompiler`; runtime contraction-plan caches live on
`GraphExecutor` and `EagerRuntime`.

Use `tenferro_einsum::EINSUM_EXTENSION_FAMILY_ID` with
`ExtensionCacheSelector` when you need to inspect or clear only einsum cache
entries.
