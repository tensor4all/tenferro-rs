# Typed Tensor Rank, View, And Stride Boundary Design

## Context

Issue #934 integrates rank metadata, borrowed views, mutable views, arbitrary
strides, materialization boundaries, and public tensor vocabulary into one
data-model migration.

The important naming decision is to keep tenferro's tensor vocabulary
consistent. This design does not introduce a public `Array` type family.
Instead, the existing `TypedTensor` family becomes the canonical typed tensor
surface and gains rank, view, and layout metadata support.

## Crate Boundary

`tenferro-tensor-core` owns storage-neutral metadata:

- sealed rank markers and rank conversion helpers
- `DynRank` and `Rank<const N: usize>`
- shape, stride, and offset metadata
- layout validation, bounds checks, and no-overlap checks
- metadata-only layout transforms
- optional host-only tensor data adapters, if a lightweight owned host value is
  still useful after the migration

`tenferro-tensor-core` must not own `Buffer`, `Placement`, backend handles,
GPU behavior, kernels, materialization loops, or execution dispatch.

`tenferro-tensor` owns execution-capable tensors:

- `TypedTensor<T, R = DynRank>`
- `TypedTensorView<'a, T, R = DynRank>`
- `TypedTensorViewMut<'a, T, R = DynRank>`
- dtype-erased `Tensor`
- host and backend buffers
- placement metadata
- same-placement canonicalization, scratch allocation, and copy-back
- backend operation dispatch

`TypedTensor<T, R>` remains capable of pointing at host or backend memory. Rank
genericity must not turn it into a host-only array type.

## Existing Type Migration

There must be only one public `TypedTensor` family after this migration:

- `tenferro-tensor::TypedTensor<T, R = DynRank>` is the canonical typed tensor
  type and may hold host or backend memory.
- `tenferro-tensor::TypedTensorView<'a, T, R = DynRank>` and
  `tenferro-tensor::TypedTensorViewMut<'a, T, R = DynRank>` are the canonical
  typed borrowed tensor views and may point at host or backend memory.
- The current host-only `tenferro-tensor-core::TypedTensor<T>` and
  `tenferro-tensor-core::TypedTensorView<'a, T>` names must not survive as
  public names because they would collide semantically with the execution
  tensor surface.

If a public host-only owned type remains necessary in `tenferro-tensor-core`, it
should be renamed to a host-specific name such as `HostTensor<T, R>`,
`HostTensorView<'a, T, R>`, and `HostTensorViewMut<'a, T, R>`. Those types are
strictly host-only wrappers around `Vec<T>` or borrowed slices plus
`TensorLayout<R>`. They must not be re-exported as `TypedTensor`.

## Rank Model

The rank model is sealed initially:

```rust
pub trait TensorRank: private::Sealed + Clone + Copy + Debug + Eq + Send + Sync + 'static {
    const RANK: Option<usize>;
    type Shape: Clone + Debug + PartialEq + Eq + AsRef<[usize]>;
    type Strides: Clone + Debug + PartialEq + Eq + AsRef<[isize]>;

    fn shape_from_vec(shape: ShapeVec) -> Result<Self::Shape>;
    fn shape_into_vec(shape: Self::Shape) -> ShapeVec;
    fn strides_from_vec(strides: StrideVec) -> Result<Self::Strides>;
    fn strides_into_vec(strides: Self::Strides) -> StrideVec;
}

pub struct DynRank;
pub struct Rank<const N: usize>;
```

`DynRank` stores dynamic vectors. `Rank<N>` stores fixed-size arrays. External
rank implementations are not supported in this phase.

## Layout Metadata

Core should provide a storage-neutral layout type:

```rust
pub struct TensorLayout<R: TensorRank = DynRank> {
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
}
```

`TensorLayout` owns metadata-only transforms and validation:

- `transpose_view`
- `slice_view`
- `reshape_view_as<R2>`
- `broadcast_in_dim_view<R2>`
- reachable physical range validation
- mutable no-overlap validation

Strides are measured in elements, not bytes. Negative strides are allowed when
the reachable range remains within the backing allocation. Read-only views may
alias logical elements. Mutable views must reject overlapping layouts,
including zero-stride broadcast layouts.

The migration must update the existing repository rule and implementation paths
that currently reject negative strides. In particular, bounds validation should
move from "all strides are non-negative" to "the minimum and maximum reachable
physical offsets are inside the borrowed allocation." Slice, reverse, reshape,
and offset helpers must be audited for assumptions that strides are
non-negative.

Owned tensors remain compact column-major. `TensorLayout` can represent
arbitrary strides, but constructors for owned `TypedTensor<T, R>` must create
compact layouts whose strides are derived from shape. Arbitrary strides are
introduced by typed view construction and metadata-only view transforms, not by
owned tensor constructors.

Mutable no-overlap validation should be correctness-first and conservative:

- reject zero-stride mutable axes unless a future API provides a uniqueness
  proof
- accept known injective layouts produced from compact tensors by transpose,
  slicing with non-zero steps, rank conversion, and compatible reshape
- for direct mutable-view construction, first use cheap sufficient checks such
  as sorted absolute-stride span validation
- for small layouts, exact physical-offset enumeration is acceptable
- for large layouts whose injectivity cannot be proven cheaply, reject the
  mutable view rather than accepting a potentially overlapping layout

This policy can be loosened later with better no-overlap proofs, but the first
implementation must not accept mutable aliases it cannot prove safe.

## Public Tensor Surface

`tenferro-tensor` should expose the rank-generic typed surfaces:

```rust
pub enum TensorBufferRef<'a, T> {
    Host(&'a [T]),
    Backend(Arc<dyn BackendBuffer<T>>),
}

pub enum TensorBufferRefMut<'a, T> {
    Host(&'a mut [T]),
    Backend {
        buffer: Arc<dyn BackendBuffer<T>>,
        borrow: PhantomData<&'a mut T>,
    },
}

pub struct TypedTensor<T, R: TensorRank = DynRank> {
    pub buffer: Buffer<T>,
    pub layout: TensorLayout<R>,
    pub placement: Placement,
}

pub struct TypedTensorView<'a, T, R: TensorRank = DynRank> {
    pub buffer: TensorBufferRef<'a, T>,
    pub layout: TensorLayout<R>,
    pub placement: Placement,
}

pub struct TypedTensorViewMut<'a, T, R: TensorRank = DynRank> {
    pub buffer: TensorBufferRefMut<'a, T>,
    pub layout: TensorLayout<R>,
    pub placement: Placement,
}
```

The exact names and field visibility can be narrower than shown here. Public
fields should remain public only when they are deliberate user-facing
contracts. The important contract is that host views borrow slices, while
backend views retain the backend allocation handle without transferring data.
Mutable backend views must carry a lifetime tied to the mutable borrow that
created the view, even if the backend allocation itself is held through `Arc`.

The dtype-erased `Tensor` remains dynamic-rank in this phase. It should offer
typed rank conversion helpers where useful, but it should not become
rank-generic.

The dtype-erased `Tensor` should store dynamic-rank typed variants whose typed
payloads use `TypedTensor<T, DynRank>`. It should not expose rank-generic enum
variants in this phase.

## View Traits

Small sealed traits are acceptable when they reduce duplication without making
execution APIs layout-polymorphic by default:

```rust
pub trait AsTensorView<T, R: TensorRank = DynRank>: private::Sealed {
    fn as_tensor_view(&self) -> TypedTensorView<'_, T, R>;
}

pub trait AsTensorViewMut<T, R: TensorRank = DynRank>: AsTensorView<T, R> {
    fn as_tensor_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R>;
}
```

These traits should only express borrowing as a tensor view. They must not own
materialization, computation, or backend dispatch. Linear algebra APIs such as
SVD should not become broad public `ArrayLike` or `TensorLike` generic APIs.

## Backend Layout Contract

Public tensor operations may accept arbitrary-stride readable views.

If a backend operation supports the incoming strided layout directly, it may
execute directly. If an operation requires compact column-major input, it must
canonicalize the input within the same placement before execution:

- CPU host view to CPU compact host buffer
- GPU/backend view to compact backend buffer through same-device copy or kernel

Hidden CPU-GPU transfers and silent CPU fallback are forbidden.

For mutable output views, unsupported strided writes use the same-placement
scratch-and-copy-back pattern:

1. allocate compact scratch in the same placement
2. run the compact-only operation into scratch
3. scatter or copy back from scratch into the mutable view in the same placement

Copy-back must be explicit in the operation implementation and must return a
`Result`; it must not rely on panic/drop behavior. If an operation fails after
partially writing to an output view, the operation's partial-write behavior must
be documented or avoided by writing into scratch first.

Read-write in-place aliasing is allowed only when an operation can prove the
aliasing is safe. Otherwise the implementation must use scratch or reject that
API shape before mutation.

## Materialization Boundary

Materialization and canonicalization belong in `tenferro-tensor`, not
`tenferro-tensor-core`.

Core may expose enough layout metadata for tensor backends to implement
same-placement canonicalization. Core must not allocate dense buffers, run
kernels, or depend on execution crates.

## Linalg And SVD

SVD, QR, eig, and similar APIs remain execution operations. They should accept
the execution tensor/view surface, not a broad generic host-array trait.

For compact-only providers such as LAPACK or cuSOLVER, non-contiguous inputs
are canonicalized within the current placement first. CUDA inputs stay on CUDA;
CPU inputs stay on CPU. Unsupported provider behavior must return an explicit
error rather than falling back to another device.

## Documentation And Repository Rules

The implementation should update repository rules with the public tensor
operation vocabulary:

- standard operation names should stay aligned across runtime, eager, and
  traced surfaces
- metadata-only borrowed equivalents use `_view` suffixes
- public synonyms such as `permute_axes` should not become the primary spelling
  for tensor transpose behavior

User-facing docs should explain that:

- `TypedTensor<T, R>` can represent host or backend-backed typed tensors
- rank-generic typed tensors and views are separate from dtype-erased `Tensor`
- arbitrary strides are view metadata
- compact-only operations may canonicalize within the same placement
- hidden CPU-GPU transfers and silent CPU fallback are forbidden

Repository rules that must be updated include the tensor data-model rules,
range-checking and slicing rules, dense layout rules, and public operation
vocabulary rules. The existing rule text that rejects negative strides in v1
must be replaced by the reachable-range validation contract described above.

## Non-Goals

- Do not add a public `Array` type family.
- Do not make dtype-erased `Tensor` rank-generic.
- Do not add external rank implementations.
- Do not put backend buffers, placement, kernels, or materialization in
  `tenferro-tensor-core`.
- Do not make graph cache keys, AD rules, or backend dispatch layout-polymorphic
  by default.
- Do not silently transfer data between CPU and GPU.
- Do not silently fall back from GPU execution to CPU execution.
