# Linear Algebra

tenferro exposes linear algebra as free functions on `TracedTensor`, much like `torch.linalg.*` or `jnp.linalg.*`. Multi-output decompositions return tuples of traced tensors, and each output can be evaluated with the same engine.

## Singular value decomposition

PyTorch: `torch.linalg.svd(a)`  
JAX: `jnp.linalg.svd(a)`

```rust
use tenferro::{svd, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
let (mut u, mut s, mut vt) = svd(&a);

let mut engine = Engine::new(CpuBackend::new());
let u_result = u.eval(&mut engine).unwrap();
let s_result = s.eval(&mut engine).unwrap();
let vt_result = vt.eval(&mut engine).unwrap();

assert_eq!(u_result.shape(), &[2, 2]);
assert_eq!(vt_result.shape(), &[2, 2]);

let mut singular_values = s_result.as_slice::<f64>().unwrap().to_vec();
singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(singular_values, vec![1.0, 2.0]);
```

## QR decomposition

PyTorch: `torch.linalg.qr(a)`  
JAX: `jnp.linalg.qr(a)`

```rust
use tenferro::{qr, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]);
let (mut q, mut r) = qr(&a);

let mut engine = Engine::new(CpuBackend::new());
let q_result = q.eval(&mut engine).unwrap();
let r_result = r.eval(&mut engine).unwrap();

assert_eq!(q_result.shape(), &[2, 2]);
assert_eq!(r_result.shape(), &[2, 2]);
assert_eq!(q_result.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
assert_eq!(r_result.as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
```

## Hermitian eigenvalue decomposition

PyTorch: `torch.linalg.eigh(a)`  
JAX: `jnp.linalg.eigh(a)`

```rust
use tenferro::{eigh, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]);
let (mut values, mut vectors) = eigh(&a);

let mut engine = Engine::new(CpuBackend::new());
let values_result = values.eval(&mut engine).unwrap();
let vectors_result = vectors.eval(&mut engine).unwrap();

assert_eq!(values_result.shape(), &[2]);
assert_eq!(vectors_result.shape(), &[2, 2]);

let mut eigenvalues = values_result.as_slice::<f64>().unwrap().to_vec();
eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
assert_eq!(eigenvalues, vec![1.0, 3.0]);
```

## Cholesky factorization

PyTorch: `torch.linalg.cholesky(a)`  
JAX: `jnp.linalg.cholesky(a)`

```rust
use tenferro::{cholesky, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let mut factor = cholesky(&a);

let mut engine = Engine::new(CpuBackend::new());
let result = factor.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 2]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);
```

## Solve a linear system

PyTorch: `torch.linalg.solve(a, b)`  
JAX: `jnp.linalg.solve(a, b)`

```rust
use tenferro::{solve, CpuBackend, Engine, TracedTensor};

let a = TracedTensor::from_vec(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 9.0]);
let b = TracedTensor::from_vec(vec![2, 1], vec![8.0_f64, 27.0]);
let mut x = solve(&a, &b);

let mut engine = Engine::new(CpuBackend::new());
let result = x.eval(&mut engine).unwrap();

assert_eq!(result.shape(), &[2, 1]);
assert_eq!(result.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
```
