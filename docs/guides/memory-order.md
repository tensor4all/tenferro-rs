# Memory Order

tenferro dense tensors use column-major flat buffers. The leftmost dimension
varies fastest in memory.

```rust
use tenferro::Tensor;

let tensor = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]);
assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
```

The logical matrix is:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

## Importing Row-Major Data

PyTorch, NumPy, and JAX examples often show row-major flat buffers. Convert
those buffers to column-major before constructing a tenferro tensor.

```rust
use tenferro::Tensor;

fn row_major_to_column_major<T: Copy>(shape: &[usize], data: &[T]) -> Vec<T> {
    assert_eq!(shape.len(), 2);
    let rows = shape[0];
    let cols = shape[1];
    assert_eq!(data.len(), rows * cols);

    let mut out = Vec::with_capacity(data.len());
    for col in 0..cols {
        for row in 0..rows {
            out.push(data[row * cols + col]);
        }
    }
    out
}

let shape = vec![2, 3];
let row_major = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let column_major = row_major_to_column_major(&shape, &row_major);
let tensor = Tensor::from_vec(shape, column_major);

assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
```

## Owned Export

Owned export returns the column-major host buffer:

```rust
use tenferro::Tensor;

let tensor = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
let (shape, data) = tensor.try_into_vec::<f64>().unwrap();

assert_eq!(shape, vec![2, 2]);
assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
```

Convert the exported buffer in your application if a consumer expects
row-major data.
