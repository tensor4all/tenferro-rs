# Tensor

`tenferro-tensor` now treats layout as first-class runtime metadata. The
public dense tensor types are:

- `TypedTensor<T>` for a concrete scalar type
- `Tensor` as the dynamic dtype enum wrapper

The current host representation is:

```rust
pub enum Buffer<T> {
    Host(Arc<Vec<T>>),
    Backend(BufferHandle<T>),
}

pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
    pub placement: Placement,
}
```

## Core Rules

- Dense tensors are stride-aware. Column-major layout is the default for new
  host allocations, not a global invariant.
- Host-backed tensor clones are shallow. `Buffer::Host` uses `Arc<Vec<T>>`, so
  metadata-only views share storage.
- Mutable host access is copy-on-write through `Arc::make_mut`.
- Logical indexing always uses `shape + strides + offset`.
- Kernels must not reconstruct layout from shape when stride metadata exists.

## Layout Semantics

`shape`, `strides`, and `offset` define the logical tensor view over the
underlying storage.

- `col_major_strides(shape)` gives the default column-major layout.
- `row_major_strides(shape)` gives the dense row-major layout.
- `is_contiguous_col_major()` and `is_contiguous_row_major()` are predicates
  over the current metadata.
- `to_contiguous(LayoutOrder)` materializes a dense host tensor in the
  requested order.

`host_data()` exposes the raw host storage buffer. It does not promise logical
iteration order for non-contiguous tensors. Use `get()` for indexed access or
`to_contiguous(...)` before handing storage to code that expects dense layout.

## Metadata-Only Operations

The following operations are intended to stay metadata-only whenever possible:

- `transpose` / `permute`
- `broadcast_in_dim`
- `reshape` when the source layout is already exactly dense row-major or dense
  column-major for the requested shape

For broadcasted views, broadcast axes carry stride `0`.

## Materialization Boundaries

Materialization is explicit and delayed to the last responsible boundary.

Current materialization boundaries include:

- faer-backed linalg entry points that require dense column-major slices
- CPU GEMM fallback canonicalization when direct strided GEMM lowering is not
  possible
- FFI / C-API export boundaries

The contract is:

- keep values as views while graph/runtime metadata is sufficient
- materialize only at a kernel or external boundary that genuinely requires a
  specific dense layout

## Copy-On-Write Host Storage

Copy-on-write is part of the current public behavior.

- `clone()` on a host tensor keeps sharing the same `Arc<Vec<T>>`
- `host_data_mut()` and `get_mut()` detach only when the storage is shared
- metadata-only structural ops can stay cheap without exposing interior
  mutability to users

This gives eager mode the same basic freedom as the compiled path: views can
survive until a mutation or dense-kernel boundary forces a copy.
