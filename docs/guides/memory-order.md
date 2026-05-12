# Memory Order

tenferro stores dense tensors in column-major order by default. In a 2D tensor,
the leftmost dimension varies fastest in memory.

```rust
use tenferro::{MemoryOrder, Tensor};

let col = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(col.order(), MemoryOrder::ColMajor);
```

For PyTorch, NumPy, and JAX-style row-major flat data, import with
`Tensor::from_vec_row_major`:

```rust
use tenferro::{MemoryOrder, Tensor};

let row = Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(row.order(), MemoryOrder::RowMajor);

let col = row.to_col_major().unwrap();
assert_eq!(col.order(), MemoryOrder::ColMajor);
assert_eq!(col.as_slice::<f64>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
```

Use `to_col_major()` or `to_row_major()` when you need a specific owned memory
order. These methods preserve the logical tensor values and reorder the owned
buffer when needed.

## Owned Export

Owned export is zero-copy only when the tensor already has the requested
memory order:

```rust
use tenferro::{MemoryOrder, Tensor};

let row = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
let (shape, data) = row.try_into_vec_row_major::<f64>().unwrap();
assert_eq!(shape, vec![2, 2]);
assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);

let col = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
let (_shape, data) = col.try_into_vec_with_order::<f64>(MemoryOrder::ColMajor).unwrap();
assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
```

If the order does not match, convert first with `to_col_major()` or
`to_row_major()`, then export with `try_into_vec_col_major()`,
`try_into_vec_row_major()`, or `try_into_vec_with_order::<T>(...)`.
