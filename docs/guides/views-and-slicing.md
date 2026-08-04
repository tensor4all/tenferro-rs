# Views, slicing, and explicit copies

A tensor view is a borrowed descriptor, not another owner. Use views when an
operation can consume shape/stride metadata directly; use `duplicate()` when a
fresh physical allocation is required.

## Read-only views

`as_view()` is an O(1) reborrow. It preserves static rank, and the dynamic
`TensorView` family preserves dtype and layout metadata without a transfer.
The typed forms are `TypedTensorView<T, R>` and `TypedTensorViewMut<T, R>`.

```rust
use tenferro_tensor::{Rank, TypedTensor};

let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 3], vec![1.0; 6])?;
let view = tensor.as_view();
assert_eq!(view.shape(), &[2, 3]);
assert_eq!(view.strides(), &[1, 2]);
assert_eq!(view.get(&[1, 2]), Some(&1.0));
# Ok::<(), tenferro_tensor::Error>(())
```

The view can be transformed without copying:

```rust
let transposed = view.transpose_view([1, 0])?;
let reversed = transposed.try_slice(&[
    tenferro_tensor::StridedSliceSpec::all(),
    tenferro_tensor::StridedSliceSpec::reverse(),
])?;
assert_eq!(reversed.shape(), &[3, 2]);
# Ok::<(), tenferro_tensor::Error>(())
```

These transformations preserve the same root allocation. They do not make a
host slice available for a backend buffer and they do not perform a transfer.

## Mutable views and disjointness

`as_view_mut()` requires an exclusive borrow of the owner. Use one mutable view
at a time unless the checked disjoint-view API proves that physical regions do
not overlap. Mutation is visible through later views of the same owner:

```rust
use tenferro_tensor::{Rank, TypedTensor};

let mut tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![0.0; 4])?;
{
    let mut view = tensor.as_view_mut();
    *view.get_mut(&[1, 0]).ok_or("missing element")? = 4.0;
}
assert_eq!(tensor.as_slice()?, &[0.0, 4.0, 0.0, 0.0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

A mutable view is not `Clone`, and there is no public mutable owner projection.
The borrow checker and descriptor validation are the alias-safety boundary.

## Explicit duplication

Call `duplicate()` when the next API needs an owned compact tensor. The
operation reads the view once and allocates a fresh owner; it never silently
uploads, downloads, or canonicalizes a backend view.

```rust
use tenferro_tensor::{Rank, TypedTensor};

let tensor = TypedTensor::<f64, Rank<1>>::from_vec_col_major([3], vec![1.0, 2.0, 3.0])?;
let view = tensor.as_view();
let duplicate = view.duplicate()?;
assert_eq!(duplicate.as_slice()?, &[1.0, 2.0, 3.0]);
assert_ne!(view.as_slice()?.as_ptr(), duplicate.as_slice()?.as_ptr());
# Ok::<(), tenferro_tensor::Error>(())
```

For a provider-resident view, use the provider's explicit same-device
materialization or `download_to_host` operation. A CPU backend that cannot
transfer a borrowed/noncompact layout returns a typed unsupported error; call
`duplicate()` first if compact host ownership is what the operation requires.

## Prepared element access

Prepared access is the boundary between validation and traversal. A prepared
contiguous read uses a typed slice. A prepared strided read initializes one
checked cursor and then advances its offset/carry state. No inner loop repeats
provider lookup, allocation, synchronization, descriptor resolution, or full
layout validation.

This is why the following two operations have different names and contracts:

| Operation | Physical effect |
| --- | --- |
| `transpose_view`, `try_slice`, reshape, reinterpretation | Descriptor-only view |
| `duplicate` | Fresh same-placement allocation and copy |
| `upload_tensor` | Fresh provider allocation and host-to-provider transfer |
| `download_tensor` | Fresh host allocation and provider-to-host transfer |
| `map`/host guard, synchronize | Existing-allocation visibility/order operation |

See [storage ownership](../storage-ownership.md) for detached/scoped execution,
AD/checkpoint retention, provider namespaces, and Apple shared endpoint rules.
