# Storage ownership and access

This guide is the user-facing contract for tensor storage. It is deliberately
small: tensor values have one physical allocation owner, while views, runtime
handles, and provider metadata borrow or retain that owner without creating a
second writable owner.

## The capability triad

- `TypedTensor<T, R>` or `Tensor` is an owned value.
- `TypedTensorView<T, R>` or `TensorView` is an immutable, lifetime-bounded
  view. It preserves shape, strides, offset, placement, and static rank where
  the type carries one.
- `TypedTensorViewMut<T, R>` or `TensorViewMut` is an exclusive mutable view.
  Two mutable views are only allowed when their checked physical regions are
  provably disjoint.

`as_view()` and `as_view_mut()` are metadata-only reborrows. They do not copy,
allocate, synchronize, upload, download, or erase a static rank. A view does
not outlive the owner from which it was borrowed.

```rust
use tenferro_runtime::TypedTensor;

let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0; 4])?;
let view = tensor.as_view();
assert_eq!(view.shape(), &[2, 2]);
assert_eq!(view.strides(), &[1, 2]);
let mut view_mut = tensor.as_view_mut();
*view_mut.get_mut(&[1, 0]).ok_or("missing element")? = 3.0;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Copies are named

A view transformation such as slicing, reshaping, transposing, or
reinterpretation changes descriptors only. It does not copy bytes. To create
a fresh owner, call the explicit `duplicate()` operation on the view or owner.
A noncompact or provider-resident view is not silently downloaded or
canonicalized by this host duplicate boundary; request the provider's explicit
materialization or transfer operation instead.

```rust
use tenferro_runtime::TypedTensor;

let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0])?;
let view = tensor.as_view();
let duplicate = view.duplicate()?;
assert_eq!(duplicate.as_slice()?, &[1.0, 2.0]);
assert_ne!(view.as_slice()?.as_ptr(), duplicate.as_slice()?.as_ptr());
# Ok::<(), tenferro_tensor::Error>(())
```

Numeric casts are computations that produce a new output. Reinterpretation is
a descriptor operation and is only valid when dtype, alignment, byte range,
and layout proofs allow it. Neither operation is a disguised transfer.

## Explicit device movement

The provider namespaces are deliberate:

- CUDA: `tenferro_gpu::cuda::{CudaBackend, upload_tensor, download_tensor}`;
- WebGPU: `tenferro_gpu::webgpu::{WebGpuBackend, upload_webgpu_tensor,
  download_webgpu_tensor}`;
- Apple shared CPU/Metal: `tenferro_gpu::apple::AppleContext`.

Use `upload_host_tensor(TensorRead::...)` or the provider-specific upload
function for a host-to-device allocation. Use `download_to_host(...)` or the
provider-specific download function for a device-to-host allocation. Every
successful transfer has a new destination allocation. Unsupported layouts and
dtypes return typed errors; there is no hidden CPU staging or fallback.

Mapping a host-visible allocation and synchronizing a provider queue are not
transfers. They make an existing allocation observable at another endpoint.
In particular, Apple CPU-to-Metal and Metal-to-CPU endpoint changes preserve
one allocation identity and count synchronization/mapping rather than tensor
bytes copied.

## Prepared access and loops

A provider validates the descriptor, range, layout, dtype, placement, and
provider identity before constructing prepared access. The prepared value then
retains opaque provider state and the lifetime lease needed by the launch. A
contiguous loop uses a typed slice; a strided loop uses one checked cursor and
incremental carry state. Element bodies do not resolve storage, look up a
provider, validate a range, allocate, synchronize, decode full-rank
coordinates, or clone owner metadata.

This separation is observable in the storage tests and in the
[views and slicing guide](guides/views-and-slicing.md). `TensorRead<'_>` is a
borrowed dispatch input, not an implicit copy request.

## Detached and scoped execution

Detached execution consumes owned groups and publishes a completed bundle only
after provider retirement is proven. `ScopedReadInputs` is synchronous and
lifetime-bounded: it can borrow caller inputs, but no work or output escapes
that scope. Asynchronous CUDA, WebGPU, and Metal providers reject a borrowed
scoped submission before admission with a typed error. A completion-unproven
outcome retains the private group and exposes diagnostics only; it does not
provide owner recovery, retry, cancellation, or a hidden copy.

AD records, checkpoints, retained inputs, and gradient bundles retain group
metadata/lifetimes. Cloning a handle does not clone physical storage. Call an
explicit duplicate/extraction operation when a standalone owner is required.

## What to remember

1. Borrow views for read access and use mutable views only under exclusive
   access.
2. Name every physical copy with `duplicate`, `upload`, `download`, or a
   computation that produces an output.
3. Treat mapping and synchronization as endpoint/lifetime operations, not
   transfers.
4. Expect validation before prepared access and constant setup work across
   element counts.
5. Use provider namespaces; do not rely on removed flat aliases or raw
   provider handles.
