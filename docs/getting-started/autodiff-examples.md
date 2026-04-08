# Autodiff Examples

tenferro provides first-order automatic differentiation through three
operations on `TracedTensor`:

- `grad(&wrt)` -- reverse-mode gradient of a scalar w.r.t. an input
- `vjp(&wrt, &cotangent)` -- vector-Jacobian product (reverse mode)
- `jvp(&wrt, &tangent)` -- Jacobian-vector product (forward mode)

Higher-order derivatives (e.g., Hessian-vector products) are computed by
composing these primitives.

## Imports

```rust,ignore
use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{CpuBackend, Tensor, TypedTensor};
// For linalg:
use tenferro::{svd, qr, eigh, cholesky, solve};
use tenferro::traced::eval_all;
```

## Simple gradient

`grad` computes the gradient of a **scalar** traced tensor with respect to
an input. The loss must be rank 0; for non-scalar outputs, reduce first.

```rust,ignore
let x = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![0.5, 1.0, -0.3]))
);
// f(x) = sum(exp(x))
let loss = x.exp().reduce_sum(&[0]);

let grad_x = loss.grad(&x).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let mut grad_x = grad_x;
let result = grad_x.eval(&mut engine).unwrap();
// grad = exp(x) = [exp(0.5), exp(1.0), exp(-0.3)]
match result {
    Tensor::F64(inner) => {
        let data = inner.host_data();
        assert!((data[0] - 0.5_f64.exp()).abs() < 1e-10);
    }
    _ => panic!("unexpected dtype"),
}
```

## Gradient through einsum

Gradients flow through einsum contractions. For non-scalar einsum outputs,
apply `reduce_sum` to get a scalar loss:

```rust,ignore
let a_data = vec![1.0, -2.0, 0.5, 3.0, 1.25, -0.75];
let b_data = vec![2.0, 0.25, -1.5, 4.0, 0.75, -0.5];

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], a_data))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3, 2], b_data))
);
let mut engine = Engine::new(CpuBackend::new());

// Y = A @ B, loss = sum(Y)
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let loss = y.reduce_sum(&[0, 1]);
let grad_a = loss.grad(&a).unwrap();

let mut grad_a = grad_a;
let result = grad_a.eval(&mut engine).unwrap();
// grad_a shape matches a: [2, 3]
assert_eq!(result.shape(), &[2, 3]);
```

### Three-way einsum gradient

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, -0.5, 2.0, 0.75]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![0.5, 1.5, -1.0, 0.25]))
);
let c = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![2.0, -1.5, 0.75, 1.25]))
);
let mut engine = Engine::new(CpuBackend::new());

let y = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
let loss = y.reduce_sum(&[0, 1]);
let grad_a = loss.grad(&a).unwrap();
```

### Multi-input gradients

Compute gradients of the same loss with respect to multiple inputs using
`eval_all`:

```rust,ignore
// Compute gradients w.r.t. multiple inputs
let mut grad_a = loss.grad(&a).unwrap();
let mut grad_b = loss.grad(&b).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let results = eval_all(&mut engine, &mut [&mut grad_a, &mut grad_b]).unwrap();
```

## VJP (vector-Jacobian product)

`vjp` computes the reverse-mode derivative for non-scalar outputs,
given a cotangent vector of the same shape as the output:

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3, 2], vec![0.5, -1.0, 2.0, 1.5, -0.25, 3.0]))
);
let cotangent = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, -0.5, 0.25, 2.0]))
);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();

// Cotangent for output Y, propagated back to input A
let ct_a = y.vjp(&a, &cotangent);

let mut ct_a = ct_a;
let result = ct_a.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

## JVP (Jacobian-vector product)

`jvp` computes the forward-mode derivative, given a tangent vector of the
same shape as the input:

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3, 2], vec![0.5, -1.0, 2.0, 1.5, -0.25, 3.0]))
);
let da = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, -0.5, 0.25, 0.0, 2.0, -1.0]))
);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();

// Tangent for input A, pushed forward to output Y
let dy = y.jvp(&a, &da);

let mut dy = dy;
let result = dy.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

## Hessian-vector product (HVP)

HVP is computed by composing forward and reverse mode. The standard
approach is **forward-over-reverse** (FoR): `jvp(grad(f, x), x, v)`.

### Scalar function

```rust,ignore
// f(x) = x^3, f''(x) = 6x
let x_val = 2.5_f64;
let x = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![], vec![x_val]))
);

let y = &(&x * &x) * &x;          // x^3
let g = y.grad(&x).unwrap();       // reverse: 3x^2

let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![], vec![1.0]))
);
let hv = g.jvp(&x, &v);           // forward-of-reverse: 6x

let mut engine = Engine::new(CpuBackend::new());
let mut hv = hv;
let result = hv.eval(&mut engine).unwrap();
match result {
    Tensor::F64(inner) => {
        let val = inner.host_data()[0];
        assert!((val - 6.0 * x_val).abs() < 1e-5);
    }
    _ => panic!("unexpected dtype"),
}
```

### Vector function: f(x) = sum(exp(x))

The Hessian is `diag(exp(x))`, so `Hv = exp(x) * v`:

```rust,ignore
let x_data = vec![0.5, 1.0, -0.3];
let v_data = vec![1.0, 2.0, 3.0];
let n = x_data.len();

let x = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![n], x_data.clone()))
);
let y = x.exp().reduce_sum(&[0]);

let g = y.grad(&x).unwrap();       // exp(x)
let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![n], v_data.clone()))
);
let hv = g.jvp(&x, &v);           // diag(exp(x)) * v

let mut engine = Engine::new(CpuBackend::new());
let mut hv = hv;
let result = hv.eval(&mut engine).unwrap();
// Hv[i] = exp(x[i]) * v[i]
```

### Quadratic form: f(x) = x^T A x

```rust,ignore
let a_data = vec![2.0, 0.5, 1.5, 3.0]; // col-major 2x2
let x_data = vec![1.0, -1.0];
let v_data = vec![0.5, 2.0];

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], a_data))
);
let x = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2], x_data))
);
let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2], v_data))
);

let mut engine = Engine::new(CpuBackend::new());
let y = einsum(&mut engine, &[&x, &a, &x], "i,ij,j->").unwrap();

let g = y.grad(&x).unwrap();
let hv = g.jvp(&x, &v);

let mut hv = hv;
let result = hv.eval(&mut engine).unwrap();
// H = A + A^T = [[4, 2], [2, 6]], Hv = [6.0, 13.0]
```

## Linalg gradients

Gradients flow through SVD, QR, eigendecomposition, Cholesky, and solve.

### SVD gradient

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3, 2], vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4]))
);

let (_u, s, _vt) = svd(&a);
let loss = (&s * &s).reduce_sum(&[0]);  // sum of squared singular values
let grad_a = loss.grad(&a).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let mut grad_a = grad_a;
let result = grad_a.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[3, 2]);
```

### QR gradient

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(
        vec![3, 3],
        vec![4.0, 0.1, 0.2, 0.1, 3.0, 0.3, 0.2, 0.3, 5.0],
    ))
);

let (_q, r) = qr(&a);
let diag_r = r.extract_diag(0, 1);
let loss = (&diag_r * &diag_r).reduce_sum(&[0]);
let grad_a = loss.grad(&a).unwrap();

let mut engine = Engine::new(CpuBackend::new());
let mut grad_a = grad_a;
let result = grad_a.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[3, 3]);
```

### Linalg operations overview

Available free functions (imported from `tenferro`):

| Function | Outputs | Description |
|----------|---------|-------------|
| `svd(&a)` | `(U, S, Vt)` | Singular value decomposition |
| `qr(&a)` | `(Q, R)` | QR decomposition |
| `eigh(&a)` | `(values, vectors)` | Symmetric eigendecomposition |
| `eig(&a)` | `(values, vectors)` | General eigendecomposition (complex output) |
| `cholesky(&a)` | `L` | Cholesky factorization |
| `solve(&a, &b)` | `X` | Solve AX = B |
| `inv(&a)` | `A_inv` | Matrix inverse |
| `det(&a)` | `scalar` | Determinant |
| `slogdet(&a)` | `(sign, log_abs_det)` | Sign and log-absolute-determinant |
| `lu(&a)` | `(P, L, U, parity)` | LU decomposition |
| `norm(&a, ord, dim, keepdim)` | `scalar` | Matrix/vector norm (`ord: Option<f64>`, `dim: Option<&[usize]>`) |
| `pinv(&a)` | `A_pinv` | Moore-Penrose pseudoinverse |

Multi-output functions return tuples of `TracedTensor`. Use `eval_all` to
evaluate them efficiently in a single pass:

```rust,ignore
use tenferro::traced::eval_all;

let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]))
);
let (mut u, mut s, mut vt) = svd(&a);

let mut engine = Engine::new(CpuBackend::new());
let results = eval_all(&mut engine, &mut [&mut u, &mut s, &mut vt]).unwrap();
// results[0] = U [2,2], results[1] = S [2], results[2] = Vt [2,2]
```
