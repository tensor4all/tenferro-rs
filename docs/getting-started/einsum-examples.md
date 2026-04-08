# Einsum Examples

tenferro provides N-ary einsum with automatic contraction path optimization.
The `einsum` function builds a lazy graph; call `.eval(&mut engine)` to execute.

## Imports

```rust,ignore
use tenferro::einsum::{einsum, einsum_with, EinsumOptimize};
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::CpuBackend;
use tenferro_tensor::{Tensor, TypedTensor};

// For nested notation and contraction trees:
use tenferro_einsum::{ContractionTree, NestedEinsum, Subscripts};
```

All data below is in **column-major** order.

## Unary einsum

Unary einsum operates on a single tensor. It can transpose, reduce, extract
diagonals, or embed diagonals.

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let mut engine = Engine::new(CpuBackend::new());

// Identity: "ij->ij"
let mut id = einsum(&mut engine, &[&a], "ij->ij").unwrap();
let r = id.eval(&mut engine).unwrap();
assert_eq!(r.shape(), &[2, 3]);

// Transpose: "ij->ji"
let mut t = einsum(&mut engine, &[&a], "ij->ji").unwrap();
let r = t.eval(&mut engine).unwrap();
assert_eq!(r.shape(), &[3, 2]);

// Row sum (reduce over columns): "ij->i"
let mut row_sum = einsum(&mut engine, &[&a], "ij->i").unwrap();
// result: [1+3+5, 2+4+6] = [9, 12]

// Full contraction (sum all elements): "ij->"
let mut total = einsum(&mut engine, &[&a], "ij->").unwrap();
// result: scalar 21.0
```

### Trace and diagonal

```rust,ignore
let m = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
);
let mut engine = Engine::new(CpuBackend::new());

// Trace: "ii->" -> sum of diagonal = 1 + 4 = 5
let mut tr = einsum(&mut engine, &[&m], "ii->").unwrap();

// Diagonal extraction: "ii->i" -> [1.0, 5.0, 9.0] for a 3x3 matrix
let m3 = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ))
);
let mut diag = einsum(&mut engine, &[&m3], "ii->i").unwrap();

// Diagonal embedding: "i->ii" -> 3x3 diagonal matrix from a vector
let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![2.0, 3.0, 5.0]))
);
let mut embed = einsum(&mut engine, &[&v], "i->ii").unwrap();
let r = embed.eval(&mut engine).unwrap();
assert_eq!(r.shape(), &[3, 3]);
```

## Binary einsum

### Matrix multiply

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(
        vec![3, 4],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ))
);
let mut engine = Engine::new(CpuBackend::new());
let mut c = einsum(&mut engine, &[&a, &b], "ij,jk->ik").unwrap();
let result = c.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 4]);
```

### Inner product (dot product)

```rust,ignore
let u = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]))
);
let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![4.0, 5.0, 6.0]))
);
let mut engine = Engine::new(CpuBackend::new());
let mut dot = einsum(&mut engine, &[&u, &v], "i,i->").unwrap();
// result: 1*4 + 2*5 + 3*6 = 32
```

### Outer product

```rust,ignore
let u = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]))
);
let v = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]))
);
let mut engine = Engine::new(CpuBackend::new());
let mut outer = einsum(&mut engine, &[&u, &v], "i,j->ij").unwrap();
let result = outer.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 3]);
```

### Hadamard product (elementwise)

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]))
);
let mut engine = Engine::new(CpuBackend::new());
let mut h = einsum(&mut engine, &[&a, &b], "ij,ij->ij").unwrap();
// result: [1*5, 2*6, 3*7, 4*8] = [5, 12, 21, 32]
```

### Matrix-vector product

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
let x = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]))
);
let mut engine = Engine::new(CpuBackend::new());
let mut y = einsum(&mut engine, &[&a, &x], "ij,j->i").unwrap();
let result = y.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2]);
```

### Batched matrix multiply

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0],
    ))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(
        vec![2, 2, 2],
        vec![5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0],
    ))
);
let mut engine = Engine::new(CpuBackend::new());
// Batch dimension is the last (k): A[i,j,k] * B[j,l,k] -> C[i,l,k]
let mut c = einsum(&mut engine, &[&a, &b], "ijk,jlk->ilk").unwrap();
let result = c.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 2, 2]);
```

## N-ary einsum

Chain-multiply three or more tensors. The optimizer automatically chooses
a good contraction order:

```rust,ignore
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]))
);
let b = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]))
);
let c = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]))
);
let mut engine = Engine::new(CpuBackend::new());
let mut d = einsum(&mut engine, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
let result = d.eval(&mut engine).unwrap();
assert_eq!(result.shape(), &[2, 2]);
```

## Contraction path control

For N-ary einsum, `einsum_with` lets you specify the contraction strategy
via `EinsumOptimize`.

### No optimization (left-to-right)

```rust,ignore
let mut result = einsum_with(
    &mut engine,
    &[&a, &b, &c],
    "ij,jk,kl->il",
    EinsumOptimize::False,
).unwrap();
```

### Explicit path (JAX-compatible)

Each pair specifies positions in a shrinking operand list. After each step,
the two contracted operands are removed and the result is appended.

```rust,ignore
// 3 operands: A(0), B(1), C(2)
// Step 1: contract B and C (positions 1,2) -> T. List becomes [A, T]
// Step 2: contract A and T (positions 0,1) -> result
let mut result = einsum_with(
    &mut engine,
    &[&a, &b, &c],
    "ij,jk,kl->il",
    EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
).unwrap();
```

### Nested notation

The most human-readable way to control contraction order. Parentheses
specify which operands to contract first:

```rust,ignore
// Contract A*B first, then multiply with C
let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
let mut result = einsum_with(
    &mut engine,
    &[&a, &b, &c],
    "ij,jk,kl->il",
    EinsumOptimize::Nested(nested),
).unwrap();
```

### Pre-computed contraction tree

Build a `ContractionTree` from external optimization and reuse it:

```rust,ignore
let subs = Subscripts::new(&[&[0u32, 1], &[1, 2], &[2, 3]], &[0u32, 3]);
let tree = ContractionTree::from_pairs(
    &subs,
    &[&[2, 2], &[2, 2], &[2, 2]],
    &[(1, 2), (0, 3)],
).unwrap();

let mut result = einsum_with(
    &mut engine,
    &[&a, &b, &c],
    "ij,jk,kl->il",
    EinsumOptimize::Tree(tree),
).unwrap();
```

## Contraction path caching

`Engine` caches contraction paths. When you call `einsum` with the same
subscript string and compatible shapes, the cached path is reused:

```rust,ignore
let mut engine = Engine::new(CpuBackend::new());

// First call computes and caches the path
let c1 = einsum(&mut engine, &[&a1, &b1], "ij,jk->ik").unwrap();

// Second call with same subscripts reuses the cached path
let c2 = einsum(&mut engine, &[&a2, &b2], "ij,jk->ik").unwrap();
```
