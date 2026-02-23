# Tensor

`Tensor<T>` is the core data type. It wraps a `DataBuffer<T>` with
shape/stride metadata and provides zero-copy view operations.

---

## Core Types

```rust
/// Memory ordering for new allocations only.
/// Not stored on the tensor — strides fully describe the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    ColumnMajor,  // First dimension has stride 1 (Fortran/Julia)
    RowMajor,     // Last dimension has stride 1 (C/NumPy)
}

/// Owned data buffer, device-aware.
/// Internally wraps Arc<BufferInner<T>> for shared ownership.
pub struct DataBuffer<T> {
    inner: Arc<BufferInner<T>>,
}

enum BufferInner<T> {
    Owned(Vec<T>),
    External { ptr: *const T, len: usize, release: Option<Box<dyn FnOnce() + Send>> },
    Gpu { device_ptr: *mut T, len: usize, space: LogicalMemorySpace, release: ... },
}

/// Multi-dimensional dense tensor.
pub struct Tensor<T: Scalar> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,
    conjugated: bool,     // lazy conjugation flag
}
```

Key design points:
- `DataBuffer<T>` uses `Arc<BufferInner<T>>` for shared ownership (PyTorch pattern)
- `clone()` is shallow (Arc refcount++, O(1))
- `conj()` is lazy (Arc clone + conjugated flag flip, O(1))
- Deep copy uses prims operations (`Permute(identity)` or `MakeContiguous`)
- Fields: `dims` (`Vec<usize>`), `strides` (`Vec<isize>`), `offset` (`isize`) — no `SmallVec`
- `MemoryOrder` is only used at allocation time, **not stored** on the tensor
- No direct dependency on strided-rs — prims backends build `StridedView` from
  `buffer.as_slice()` + `dims()` + `strides()` + `offset()`

---

## Constructors

```rust
impl<T: Scalar> Tensor<T> {
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self>;
    pub fn from_vec(data: Vec<T>, dims: &[usize], strides: &[isize], offset: isize) -> Result<Self>;
    pub fn eye(n: usize, memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
}
```

## Metadata

```rust
impl<T: Scalar> Tensor<T> {
    pub fn dims(&self) -> &[usize];
    pub fn strides(&self) -> &[isize];
    pub fn ndim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn offset(&self) -> isize;
    pub fn buffer(&self) -> &DataBuffer<T>;
    pub fn logical_memory_space(&self) -> LogicalMemorySpace;
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice>;
    pub fn set_preferred_compute_device(&mut self, d: Option<ComputeDevice>);
    pub fn effective_compute_devices(&self, op: OpKind) -> Result<Vec<ComputeDevice>>;
    pub fn is_conjugated(&self) -> bool;
}
```

## View Operations (zero-copy)

All view operations return `Tensor<T>` (not `TensorView`). They are
zero-copy because `DataBuffer` uses `Arc` — the returned tensor shares
the same underlying buffer with only metadata (dims, strides, offset)
changed.

```rust
impl<T: Scalar> Tensor<T> {
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>>;
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>>;
    pub fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>>;
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>>;
}
```

## Data Operations

```rust
impl<T: Scalar> Tensor<T> {
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    pub fn is_contiguous(&self) -> bool;
    pub fn conj(&self) -> Tensor<T>;       // lazy: Arc clone + flag flip
    pub fn into_conj(self) -> Tensor<T>;   // lazy: flag flip only (no refcount)
    pub fn tril(&self, diagonal: isize) -> Tensor<T>;
    pub fn triu(&self, diagonal: isize) -> Tensor<T>;
    pub fn wait(&self);
    pub fn is_ready(&self) -> bool;
}
```

## Explicit Memory Movement

```rust
impl<T: Scalar> Tensor<T> {
    /// Asynchronous explicit move between logical memory spaces.
    /// Same source/destination space: zero-copy no-op.
    /// Different spaces: explicit transfer (never implicit in ops).
    pub fn to_memory_space_async(&self, dst: LogicalMemorySpace) -> Result<Tensor<T>>;
}
```

---

## DataBuffer Shared Ownership (Arc)

### Motivation

`Tensor<T>` needs cheap zero-copy operations (permute, broadcast, etc.)
AND must satisfy `Differentiable::Tangent: Clone` from chainrules-core.
Arc-based shared ownership achieves both:

| Operation | Cost | Mechanism |
|-----------|------|-----------|
| `clone()` | O(1) | Arc refcount++ |
| `conj()` | O(1) | Arc clone + flag flip |
| `permute()` | O(1) | Arc clone + metadata change |
| Deep copy | O(n) | Prims `Permute(identity)` or `MakeContiguous` |
| `as_mut_slice()` | — | `Arc::get_mut()` — `Some` only if refcount == 1 |

### DataBuffer API

```rust
impl<T> DataBuffer<T> {
    pub fn from_vec(v: Vec<T>) -> Self;
    pub unsafe fn from_external(ptr, len, release) -> Self;
    pub fn as_slice(&self) -> Option<&[T]>;        // None for GPU
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]>; // None if shared or GPU
    pub fn as_ptr(&self) -> Option<*const T>;       // None for GPU
    pub fn as_device_ptr(&self) -> Option<*const T>; // None for CPU
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn is_owned(&self) -> bool;
    pub fn is_gpu(&self) -> bool;
    pub fn is_unique(&self) -> bool;  // Arc::strong_count == 1
    pub fn gpu_memory_space(&self) -> Option<LogicalMemorySpace>;
}
```

### Mutable Access Pattern

`as_mut_slice()` uses `Arc::get_mut()` — returns `Some` only when
the Arc reference count is 1 (exclusive ownership). If shared:

```rust
// Shared: as_mut_slice() returns None
let a = tensor.clone(); // Arc refcount = 2
assert!(tensor.buffer_mut().as_mut_slice().is_none());

// Deep copy first to get exclusive ownership
let mut deep = /* prims MakeContiguous */ ;
assert!(deep.buffer_mut().as_mut_slice().is_some());
```

Prims backends bypass this via raw pointers (unsafe), guaranteeing
non-overlapping input/output buffers.

---

## View Operations: No Separate TensorView Type

All view operations return `Tensor<T>`, not a separate `TensorView` type.
Because `DataBuffer<T>` uses `Arc`, view operations are zero-copy: they
share the underlying buffer and only change metadata (dims, strides, offset).

```rust
impl<T: Scalar> Tensor<T> {
    // Zero-copy metadata operations → return Tensor (Arc shared)
    fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<Tensor<T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;
    fn select(&self, dim: usize, index: usize) -> Result<Tensor<T>>;
    fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Tensor<T>>;

    // Consume self → Tensor (may reuse buffer if unique)
    fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    fn into_conj(self) -> Tensor<T>;

    // Borrow → new Tensor (Arc shared or new allocation)
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;  // lazy: Arc clone + flag flip
}
```

**einsum takes &Tensor references:**

```rust
pub fn einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    A: Semiring,
    B: TensorPrims<A>;
```

### Ownership Safety Examples

```rust
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, ColumnMajor);

// permute returns Tensor (Arc shared, zero-copy)
let at = a.permute(&[1, 0]).unwrap();
assert_eq!(at.dims(), &[4, 3]);
// a and at share the same buffer — both usable simultaneously

// clone is cheap (Arc refcount++)
let a2 = a.clone();
```

### Design Choice: Arc-Only (No TensorView)

Arc was chosen over a separate `TensorView<'a, T>` because:
1. `Differentiable::Tangent: Clone` in chainrules-core requires Clone on Tensor
2. GPU buffer clone needs device operations — Arc avoids this for shallow clone
3. `conj()` needs buffer sharing for lazy semantics (both CPU and GPU)
4. View operations return owned `Tensor<T>` — simpler API, no lifetime propagation

A borrowed `TensorView<'a, T>` could be added in the future for
specialized use cases (e.g., GPU event wait semantics, avoiding Arc
refcount overhead on read-only access), but the current implementation
uses `Tensor<T>` uniformly

---

## Asynchronous Execution (CompletionEvent)

### Chosen Approach: Tensor Embeds Async State

Rather than separate `einsum` / `einsum_async` functions, `Tensor<T>`
carries an optional `CompletionEvent` that tracks pending accelerator
computation (PyTorch model):

```rust
// Accelerator operations return immediately with event attached
let c = einsum("ij,jk->ik", &[&a_gpu, &b_gpu])?;
//  → GPU submit, c.event = Some(event_1), returns immediately

let d = einsum("ij,jk->ik", &[&c, &e_gpu])?;
//  → detects c.event → sets up stream dependency → no CPU wait

// CPU data access triggers implicit synchronization
let tv = d.tensor_view();  // tensor_view() calls wait() internally
```

For CPU tensors, `event` is always `None` with zero overhead.

### CompletionEvent Variants

```rust
pub struct CompletionEvent {
    inner: CompletionEventInner,
}

enum CompletionEventInner {
    Noop,
    Cuda { _event: *mut c_void },   // cudaEvent_t
    Rocm { _event: *mut c_void },   // hipEvent_t
}
```

### Two-Tier API Contract

| Tier | Methods | Event handling | User visibility |
|------|---------|---------------|-----------------|
| **Public (CPU-read)** | `tensor_view()`, `permute()`, `broadcast()`, `diagonal()`, etc. | **Wait** if pending, return ready data | Yes |
| **Internal (pipeline)** | `pub(crate) as_operand_view()` | **Propagate** event | No (crate-internal) |
| **Accelerator ops** | `einsum` (takes `&[&Tensor]`) | Calls `as_operand_view()` internally, **detects** events → stream dependency | Yes (transparent) |

### Applicability Beyond GPU

`CompletionEvent` applies equally to multi-threaded CPU execution:

- **Contraction tree parallelism**: Independent subtrees dispatched to
  different threads, each result carries a `CompletionEvent`.
- **User-level parallelism**: Independent `einsum` calls on separate threads
  with automatic event-based chaining.

**Implementation note**: `wait(&self)` requires interior mutability
(e.g., `Cell<Option<CompletionEvent>>`) to clear the event field through
a shared reference.

**Current status**: POC `event` field is `Option<CompletionEvent>` (placeholder).
Actual async execution will be implemented with accelerator backends.

---

## Lazy Conjugation

`Tensor::conj()` is always lazy (both CPU and GPU), matching PyTorch's
`torch.conj()` semantics:

| Operation | Cost | Mechanism |
|-----------|------|-----------|
| `tensor.conj()` | O(1) | Arc clone + `conjugated` flag flip |
| `tensor.into_conj()` | O(1) | Flag flip only (no Arc clone) |
| `view.conj()` | O(1) | Flag flip (zero-cost, no refcount) |
| Materialize | O(n) | `Backend::resolve_conj()` via `ElementwiseUnary(Conj)` |

Backends read `is_conjugated()` when building operation descriptors:
- **GPU**: `CUTENSOR_OP_CONJ` / `HIPTENSOR_OP_CONJ` in tensor descriptors
- **CPU**: conjugation applied during computation kernels

---

## Custom Element-Wise Operations

For arbitrary user functions not in `TensorPrims`, use strided-kernel
directly via `buffer().as_slice()`:

```rust
let a_slice = tensor_a.buffer().as_slice().unwrap();
let b_slice = tensor_b.buffer().as_slice().unwrap();
// Build StridedView from dims/strides/offset and use strided_kernel
```
