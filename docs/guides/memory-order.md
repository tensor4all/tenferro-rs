# Memory Order

tenferro dense tensors use contiguous column-major storage. The leftmost
dimension varies fastest in memory.
Owned runtime tensors are compact column-major only. Arbitrary strides,
offsets, transposes, slices, and reversals live on typed views or layout
metadata until an operation needs compact storage.

This is intentional. tenferro is designed to feel natural for Fortran, Julia,
MATLAB, Eigen/LAPACK-oriented, and scientific computing workflows where
column-major arrays are common. It also gives a clear convention for batched
linear algebra: keep compute dimensions on the left and batch dimensions on
the right.

## Shape And Flat Buffer

For a logical `[2, 3]` matrix:

```text
[[1, 2, 3],
 [4, 5, 6]]
```

the column-major flat buffer is:

```text
[1, 4, 2, 5, 3, 6]
```

```rust
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
);

assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
```

## Importing Row-Major Data

PyTorch, NumPy, JAX, and many C-style examples present flat buffers in
row-major order. Use `from_vec_row_major` at the boundary when the input buffer
is written row by row.

```rust
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_row_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
);

assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
```

Use `from_vec_col_major` when the buffer already follows tenferro's physical
order. Use `from_vec_row_major` when the buffer follows the order shown in most
PyTorch, NumPy, and JAX snippets.

## Batch Axes

Because tenferro is column-major, trailing batch axes are natural for batched
matrix operations and einsum. In a shape like `[m, k, batch]`, each batch slice
contains a contiguous `[m, k]` matrix, so the backend can operate on each matrix
without treating batch as the fastest-varying dimension.

This differs from many PyTorch examples that place batch first. When porting
code, prefer moving batch axes to the right if the tensor will feed repeated
linear algebra or contraction kernels.

## Owned Export

Owned export returns the column-major host buffer:

```rust
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
let (shape, data) = tensor.try_into_vec_col_major::<f64>().unwrap();

assert_eq!(shape, vec![2, 2]);
assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
```

Convert the exported buffer in your application if a consumer expects
row-major data.

## Views And Placement

Metadata-only operations such as transpose and slice can create strided views
without changing the backing buffer. Compact-only operations may copy a host
view into a host compact tensor, or a GPU view into a GPU compact tensor, when
the backend requires compact storage. That copy stays on the same device.
tenferro never silently uploads CPU tensors or downloads GPU tensors as part of
a tensor operation.
