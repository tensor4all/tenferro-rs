# Owned Contiguous Memory Order Design

## Scope

tenferro remains a dense tensor library whose runtime tensors own contiguous
storage. The first row-major interoperability step is limited to CPU host
buffers passed by ownership transfer. It does not add borrowed tensor views,
strided tensor views, offset metadata, GPU placement changes, or a direct
`ndarray` dependency.

The goal is to let callers move an owned row-major `Vec<T>` into tenferro
without copying, and move an owned tenferro buffer back out when the requested
memory order already matches the stored order.

## Tensor Metadata

Add a physical contiguous memory-order tag:

```rust
pub enum MemoryOrder {
    ColMajor,
    RowMajor,
}
```

`TypedTensor<T>` gains only this field:

```rust
pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,
    pub shape: Vec<usize>,
    pub placement: Placement,
    pub order: MemoryOrder,
}
```

`shape` remains the logical tensor shape. Because tensors are contiguous, the
logical strides can be derived from `shape` and `order` whenever a backend
needs them. No `strides` or `offset` fields are added.

`Tensor`, `EagerTensor`, and `EagerContext` do not need structural changes.
`Tensor` continues to wrap dtype-specialized `TypedTensor<T>` values.
`EagerTensor` continues to hold `Arc<Tensor>`, so memory order is available
through its concrete data.

## Constructors and Export

Existing constructors keep their current column-major meaning:

```rust
TypedTensor::from_vec(shape, data)
Tensor::from_vec(shape, data)
```

Add explicit owned constructors:

```rust
TypedTensor::from_vec_col_major(shape, data)
TypedTensor::from_vec_row_major(shape, data)
Tensor::from_vec_col_major(shape, data)
Tensor::from_vec_row_major(shape, data)
```

Owned export should distinguish zero-copy extraction from layout conversion:

```rust
tensor.try_into_vec_col_major()
tensor.try_into_vec_row_major()
tensor.to_col_major()
tensor.to_row_major()
```

The `try_into_vec_*` methods return the owned buffer without copying only when
the stored order matches the requested order and the tensor owns host storage.
If the order differs, they return an error. The `to_*` methods may allocate and
reorder explicitly.

## Computation Graph Boundary

Graph semantics do not change. `TensorMeta` remains dtype plus logical shape,
and graph operations continue to use logical axis numbering. Row-major input
does not reverse graph axes.

Concrete input tensors carry `MemoryOrder` at evaluation time:

```rust
TracedTensor::from_tensor_concrete_shape(tensor)
TracedTensor::from_tensor_symbolic_shape(tensor)
TracedTensor::eval_with_inputs(engine, bindings)
Engine::eval_exec_ir(program, inputs)
```

Compilation, shape inference, AD rules, and operation descriptors remain based
on logical shape and dtype. Backends decide how to handle physical memory order
when executing concrete tensors. The minimal CPU implementation may materialize
row-major inputs into column-major buffers at backend boundaries for operations
that assume column-major layout.

Backend outputs should initially stay column-major unless a specific operation
has a deliberate reason to preserve or produce row-major output. Users who need
row-major output call an explicit conversion or zero-copy export method.

## ndarray Interop Without Dependency

The core crate does not depend on `ndarray`. Users or a future adapter crate can
bridge through owned contiguous buffers:

```rust
let shape = array.shape().to_vec();
let (data, offset) = array.into_raw_vec_and_offset();
assert_eq!(offset, Some(0));
let tensor = Tensor::from_vec_row_major(shape, data)?;
```

Non-contiguous ndarray views are not zero-copy inputs for this design. They must
be made contiguous before ownership transfer.

## Non-Goals

- No `ndarray` dependency in core crates.
- No borrowed host tensor views.
- No runtime `strides` or `offset` fields.
- No GPU or multi-device changes.
- No graph-level layout metadata.
- No implicit layout conversion during zero-copy export.
