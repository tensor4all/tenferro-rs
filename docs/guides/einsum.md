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

These match the usual einsum idioms from NumPy, PyTorch, and JAX.

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
