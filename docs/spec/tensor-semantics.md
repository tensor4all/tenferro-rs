# Tensor Semantics

**Date:** 2026-05-28
**Parent:** `../index.md`
**Related:** `../architecture/tenferro-crates.md`, `backend-contract.md`,
`primitive-catalog.md`

---

## I. Purpose

This document specifies the current dense tensor data model split between
`tenferro-tensor-core` and `tenferro-tensor`.

The split is intentional:

- `tenferro-tensor-core` is a lightweight rank/layout metadata and host-only
  adapter layer.
- `tenferro-tensor` adds runtime tensor storage, placement metadata, typed
  views, and backend traits.
- `tenferro-cpu` owns CPU backend implementations, CPU kernels, provider
  selection, and CPU execution resources.

`tenferro-tensor-core` must not require computation backends, GPU runtimes,
provider selection, graph execution, or AD. Crates that need only dtype tags,
host tensor data, scalar traits, shape/stride metadata, or metadata-only views
should depend on `tenferro-tensor-core`.

---

## II. `tenferro-tensor-core`

`tenferro-tensor-core` owns backend-independent host tensor metadata and
contiguous host storage.

Current public concepts:

- `DType`: runtime dtype tags for `F32`, `F64`, `I32`, `I64`, `Bool`, `C32`,
  and `C64`.
- `TensorScalar`: sealed scalar trait for supported scalar types.
- `HostTensor<T>`: owned typed host tensor with contiguous column-major data.
- `Tensor`: dynamic host tensor enum over the supported scalar types.
- `HostTensorView<'a, T>` and `TensorView<'a>`: borrowed metadata-only views.
- `TensorRef<'a>`: borrowed dynamic tensor reference.
- `ShapeVec` and `StrideVec`: compact shape and signed-stride vectors.
- `SliceSpec`: explicit slice descriptor. A zero step is invalid.

Core tensors are host-resident and backend-independent. They have no device
placement, no backend-owned buffers, no GPU handles, and no execution methods.

### Metadata-only views

Core views describe shape, signed strides, and an offset into borrowed host
storage. The view operations are metadata-only:

- `reshape_view`
- `transpose_view`
- `slice_view`

Views may be non-contiguous. `as_slice()` succeeds only when the view is
slice-contiguous for the borrowed storage. `TensorLayout` metadata slicing
supports signed strides and negative steps when reachable-range validation
proves every logical element maps inside the backing allocation. Zero step
remains invalid.

The current `tenferro-tensor-core` host adapters
`HostTensorView::slice_view` and `TensorView::slice_view` are a narrower
positive-step compatibility surface. Runtime views in `tenferro-tensor`
(`TypedTensorView` and `TypedTensorViewMut`) use the general reachable-range
contract for negative-step metadata views.

---

## III. `tenferro-tensor`

`tenferro-tensor` is the runtime dense tensor crate. It reuses the core dtype
and scalar model, then adds runtime storage and backend placement.

The current typed runtime tensor shape is:

```rust
pub struct TypedTensor<T, R = DynRank> {
    pub buffer: Buffer<T>,
    layout: TensorLayout<R>,
    pub placement: Placement,
}

pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(Arc<dyn BackendBuffer<T>>),
}
```

`Tensor` is the dynamic runtime enum over the supported scalar types:

- `F32`
- `F64`
- `I32`
- `I64`
- `Bool`
- `C32`
- `C64`

Runtime placement is explicit metadata:

```rust
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Managed,
    Other(String),
}

pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

pub enum GpuBackendKind {
    Cuda,
    Rocm,
    Other(String),
}

pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
}
```

Owned runtime tensors are compact column-major tensors. Arbitrary strides,
offsets, transposes, slices, and reverse layouts live on `TypedTensorView`,
`TypedTensorViewMut`, or `TensorLayout` metadata until an explicit
same-placement canonicalization boundary. Backend buffers are opaque to the
runtime tensor layer; the backend that owns the concrete handle is responsible
for downcasting and execution.

`tenferro-tensor` owns:

- runtime dense tensor types, including `TypedTensor<T, R = DynRank>` and
  dynamic-rank `Tensor`
- backend traits
- host/runtime views used by kernels

### DType conversion

Runtime dtype conversion has two public meanings:

- `convert(dtype)` is checked. It accepts conversions that are valid according
  to tenferro's dtype-promotion lattice and returns a typed error for lossy
  conversions such as float or complex to integer, complex to real, integer to
  boolean, or precision narrowing.
- `cast(dtype)` is explicit. It may perform lossy dtype projection and is the
  API callers use when they intentionally want truncation, precision narrowing,
  complex projection, or boolean truthiness.

The internal primitive and execution IR may continue to use the legacy
`Convert` operation name for dtype projection, including AD cotangent
projection, but public APIs must keep checked `convert` separate from explicit
lossy `cast`.

CPU backend implementations, CPU kernels, and CPU resource pools belong in
`tenferro-cpu`. GPU backend implementations and GPU transfer helpers belong in
`tenferro-gpu`.

---

## IV. Data Model vs. Execution

The tensor data model does not own graph compilation, AD, or extension
registration.

- `tenferro-runtime` owns concrete tensor helpers, traced tensors, graph
  compilation/execution, extension runtime registration, and extension cache
  storage.
- `tenferro-ad` owns eager AD runtime surfaces and traced AD extension traits.
- `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft` own their public
  APIs, traced/eager helpers, extension runtimes, and optional AD rules.

Computation should be exposed as free functions, backend dispatch, runtime
execution, or extension runtimes. The tensor types should remain data and
metadata carriers.

There is no implicit CPU<->GPU transfer for user-visible backend operations.
Tensors must already be placed on the correct device for the backend call,
except for explicit upload/download helpers and internal execution conveniences
documented in [`backend-contract.md`](backend-contract.md).

---

## V. Dense Tensor Boundary

`tenferro_tensor::Tensor` is a dense runtime tensor. It does not carry structural
metadata such as diagonal, symmetric, block-diagonal, or sparse layout tags.

This is a deliberate boundary:

- structural variants cause a combinatorial expansion of operation cases
- core graph and execution IR remain easier to reason about when runtime
  tensors are logically dense
- extension crates can add structured algorithms without changing the base
  tensor enum

Structured values can be represented by external crates or higher-level
wrappers that store dense tensor leaves and call tenferro operations.
tenferro's runtime tensor remains the dense leaf type.

---

## VI. Einsum, Diagonal, and Repeated Labels

Trace, diagonal extraction, diagonal embedding, and tensor-network hyper-edge
patterns should be expressed through `tenferro-einsum` rather than by adding
structured tensor variants to the runtime tensor type.

Examples:

```text
einsum("ii->", A)              # trace
einsum("ii->i", A)             # diagonal extraction
einsum("i->ii", v)             # diagonal embedding
einsum("ik,k,kj->ij", U, s, V) # SVD-like reconstruction without dense diag
```

The operation semantics and contraction planning belong to `tenferro-einsum`.
The runtime tensor model only provides dense tensor operands and results.

---

## VII. Linalg Batch Convention

Linalg ops follow trailing-batch convention: core matrix dims are leftmost and
batch dims are rightmost. Shape `[M, N, B1, B2, ...]` means `B1*B2*...`
independent `M x N` matrices. Each batch slice is contiguous in column-major
memory, enabling zero-copy slicing.

This differs from JAX, NumPy, and PyTorch leading-batch convention
`[B, M, N]`. The choice matches tenferro's column-major storage: rightmost
dims have the largest stride, so trailing batch dims make each `[M, N]` slice a
contiguous block.

When `shape.len() == core_rank`, the op is a plain 2D call with zero overhead.

| Op | Input shape | Output shape(s) |
|---|---|---|
| `cholesky` | `[N, N, B...]` | `[N, N, B...]` |
| `svd` | `[M, N, B...]` | U `[M, K, B...]`, S `[K, B...]`, Vt `[K, N, B...]` |
| `qr` | `[M, N, B...]` | Q `[M, K, B...]`, R `[K, N, B...]` |
| `eigh` | `[N, N, B...]` | vals `[N, B...]`, vecs `[N, N, B...]` |
| `solve` | A `[N, N, B...]`, b `[N, M, B...]` | `[N, M, B...]` |

The trailing-batch convention also applies to `DotGeneral` / `BatchedGemm`
(documented in `AGENTS.md` under column-major dimension ordering).

---

## VIII. Source of Truth

Current implementation ownership:

- `crates/tenferro-tensor-core/src/lib.rs` for the host-only data model and
  metadata-only views
- `crates/tenferro-tensor/src/types.rs` for runtime dense tensor storage and
  placement metadata
- `crates/tenferro-tensor/src/backend.rs` for backend traits
- `crates/tenferro-runtime/src/*` for graph execution and extension runtime dispatch

If this document conflicts with those files, the implementation wins and this
document should be updated.
