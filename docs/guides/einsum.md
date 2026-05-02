# Einsum

If you already use `torch.einsum(...)` or `jnp.einsum(...)`, the tenferro version should feel familiar:

- PyTorch: `torch.einsum("ij,jk->ik", a, b)`
- JAX: `jnp.einsum("ij,jk->ik", a, b)`
- tenferro: `einsum(&mut engine, &[&a, &b], "ij,jk->ik")`

tenferro keeps the same subscript language, but evaluation stays lazy until you call `.eval(&mut engine)`.

## Matrix multiply

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let b = TracedTensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let result = c.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
```

## Trace and diagonal

These match the usual repeated-label einsum idioms from NumPy, PyTorch, and JAX.
When a label appears more than once in one input, tenferro first selects the
diagonal for those axes. If that label is absent from the output, the diagonal
is reduced. If a label appears more than once in the output, tenferro embeds the
input on that diagonal.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let matrix = TracedTensor::from_vec(
    vec![3, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
);

let mut engine = Engine::new(CpuBackend::new());
let mut trace = einsum(&mut engine, &[&matrix], "ii->").unwrap();
let mut diagonal = einsum(&mut engine, &[&matrix], "ii->i").unwrap();

let trace_result = trace.eval(&mut engine).unwrap();
let diagonal_result = diagonal.eval(&mut engine).unwrap();

assert_eq!(trace_result.as_slice::<f64>().unwrap(), &[15.0]);
assert_eq!(diagonal_result.as_slice::<f64>().unwrap(), &[1.0, 5.0, 9.0]);
```

Higher-rank repeated labels use the same rule. For `"iij->ij"`, the first two
axes are diagonalized and the trailing `j` axis is preserved.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let x = TracedTensor::from_vec(
    vec![2, 2, 3],
    vec![
        1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ],
);

let mut engine = Engine::new(CpuBackend::new());
let mut y = einsum(&mut engine, &[&x], "iij->ij").unwrap();
let result = y.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 4.0, 5.0, 8.0, 9.0, 12.0]);
```

## Outer product

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let u = TracedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
let v = TracedTensor::from_vec(vec![3], vec![3.0_f64, 4.0, 5.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut outer = einsum(&mut engine, &[&u, &v], "i,j->ij").unwrap();
let result = outer.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]);
```

## Diagonal embedding

The same compact syntax can build a diagonal matrix from a vector.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let v = TracedTensor::from_vec(vec![3], vec![2.0_f64, 3.0, 5.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut diag = einsum(&mut engine, &[&v], "i->ii").unwrap();
let result = diag.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[3, 3]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]);
```

## N-ary contraction

Like `torch.einsum` and `jnp.einsum`, tenferro accepts more than two inputs. tenferro chooses a contraction order automatically and caches it on the engine.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
let b = TracedTensor::from_vec(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
let c = TracedTensor::from_vec(vec![2, 2], vec![9.0_f64, 10.0, 11.0, 12.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut out = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
let result = out.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 2]);
```

## Static vs symbolic shapes

A `TracedTensor` carries its shape in one of two modes, chosen at construction time:

| Mode | Constructor | When to use |
| --- | --- | --- |
| **Concrete (static)** | `from_vec`, `from_tensor_concrete_shape`, `input_concrete_shape(dtype, shape)` | Shape fixed at graph-build time. Enables build-time contraction path optimization, per-shape specialization |
| **Symbolic** | `from_tensor_symbolic_shape`, `input_symbolic_shape(dtype, rank)` | Shape only known at eval time. Use for dynamic batch sizes, polymorphic graphs, or to defer path optimization |

For N-ary einsum the mode propagates:

- **All inputs have concrete shapes** → the contraction path is optimized at graph-build time and the einsum is decomposed into binary `DotGeneral` ops.
- **Any input is symbolic** → the einsum is kept as a single `NaryEinsum` op; the contraction path is resolved at eval time from the actual input shapes.

The engine caches optimized contraction trees keyed by `(subscripts, input shapes)`, so the eval-time cost is amortized across repeated calls with the same shapes.

```rust
use tenferro::{einsum::einsum, CpuBackend, DType, Engine, Tensor, TracedTensor};

// One symbolic-shape input + one concrete-shape input.
let a = TracedTensor::input_symbolic_shape(DType::F64, 2);
let b = TracedTensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

let mut engine = Engine::new(CpuBackend::new());
let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();

// Bind a concrete tensor to the symbolic leg at eval time.
let a_concrete = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
let result = c.eval_with_inputs(&mut engine, &[(&a, &a_concrete)]).unwrap();

assert_eq!(result.shape(), &[2, 2]);
```

## Batched matrix multiply

PyTorch and JAX users often put the batch axis first. In tenferro, trailing batch axes line up naturally with column-major storage, so this example keeps the batch dimension on the right.

```rust
use tenferro::{einsum::einsum, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(
    vec![2, 2, 2],
    vec![1.0_f64, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
);
let b = TracedTensor::from_vec(
    vec![2, 2, 2],
    vec![5.0_f64, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
);

let mut engine = Engine::new(CpuBackend::new());
let mut c = einsum(&mut engine, &[&a, &b], "ijk,jlk->ilk").unwrap();
let result = c.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 2, 2]);
```
