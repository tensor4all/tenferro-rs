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
pub enum DataBuffer<T> {
    Cpu(StridedArray<T>),
    // Future: Cuda(CudaBuffer<T>), Hip(HipBuffer<T>)
}

/// Multi-dimensional dense tensor.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,  // None = ready, Some = pending accelerator work
}
```

Key design points:
- `DataBuffer<T>` is an enum in `tenferro-tensor` (not a separate crate, no `Arc` wrapping)
- Fields: `dims` (`Vec<usize>`), `strides` (`Vec<isize>`), `offset` (`isize`) — no `SmallVec`
- `MemoryOrder` is only used at allocation time, **not stored** on the tensor
- Bridge to strided-rs via `view()` / `view_mut()`

---

## Constructors

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn zeros(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn ones(dims: &[usize], memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
    pub fn from_slice(data: &[T], dims: &[usize], order: MemoryOrder) -> Result<Self>;
    pub fn from_strided_array(array: StridedArray<T>) -> Self;
    pub fn eye(n: usize, memory_space: LogicalMemorySpace, order: MemoryOrder) -> Self;
}
```

## Metadata

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn dims(&self) -> &[usize];
    pub fn strides(&self) -> &[isize];
    pub fn ndim(&self) -> usize;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn logical_memory_space(&self) -> LogicalMemorySpace;
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice>;
    pub fn set_preferred_compute_device(&mut self, d: Option<ComputeDevice>);
    pub fn effective_compute_devices(&self, op: OpKind) -> Result<Vec<ComputeDevice>>;
}
```

## View Operations (zero-copy)

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn view(&self) -> StridedView<'_, T>;
    pub fn view_mut(&mut self) -> StridedViewMut<'_, T>;
    pub fn permute(&self, perm: &[usize]) -> Result<Tensor<T>>;
    pub fn broadcast(&self, target_dims: &[usize]) -> Result<Tensor<T>>;
    pub fn diagonal(&self, axes: &[(usize, usize)]) -> Result<Tensor<T>>;
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Tensor<T>>;
    pub fn select(dim: usize, index: usize) -> Result<Tensor<T>>;
    pub fn narrow(dim: usize, start: usize, length: usize) -> Result<Tensor<T>>;
}
```

## Data Operations

```rust
impl<T: ScalarBase> Tensor<T> {
    pub fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    pub fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    pub fn is_contiguous(&self) -> bool;
    pub fn conj(&self) -> Tensor<T>;
    pub fn into_conj(self) -> Tensor<T>;
    pub fn tril(&self, diagonal: isize) -> Tensor<T>;
    pub fn triu(&self, diagonal: isize) -> Tensor<T>;
    pub fn wait(&self);
    pub fn is_ready(&self) -> bool;
}
```

## Explicit Memory Movement

```rust
impl<T: ScalarBase> Tensor<T> {
    /// Asynchronous explicit move between logical memory spaces.
    /// Same source/destination space: zero-copy no-op.
    /// Different spaces: explicit transfer (never implicit in ops).
    pub fn to_memory_space_async(&self, dst: LogicalMemorySpace) -> Result<Tensor<T>>;
}
```

---

## Tensor / TensorView Ownership Split

### Motivation

`permute(&self) -> Tensor<T>` is "zero-copy," but `Tensor<T>` owns its
`DataBuffer<T>`. For true zero-copy view operations, the new tensor must
share the original's data buffer. The chosen approach uses
`Tensor` (owned) + `TensorView` (borrowed):

```rust
/// Owned tensor. Holds exclusive ownership of the data buffer.
pub struct Tensor<T: ScalarBase> {
    buffer: DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,
}

/// Borrowed tensor view. References a Tensor's data buffer.
/// Zero-copy, lifetime-tied to the source Tensor.
pub struct TensorView<'a, T: ScalarBase> {
    data: &'a DataBuffer<T>,
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<&'a CompletionEvent>,
}
```

### API Design

**Tensor (owned) methods:**

```rust
impl<T: ScalarBase> Tensor<T> {
    // Public: Borrow → TensorView (zero-copy, waits if pending)
    fn view(&self) -> TensorView<'_, T>;
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'_, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'_, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'_, T>>;

    // Internal: Non-blocking operand view (event propagated)
    pub(crate) fn as_operand_view(&self) -> TensorView<'_, T>;

    // Consume self → Tensor (buffer reuse, guaranteed)
    fn into_contiguous(self, order: MemoryOrder) -> Tensor<T>;
    fn into_conj(self) -> Tensor<T>;

    // Borrow → new Tensor (new allocation, waits if pending)
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**TensorView methods:**

```rust
impl<'a, T: ScalarBase> TensorView<'a, T> {
    fn permute(&self, perm: &[usize]) -> Result<TensorView<'a, T>>;
    fn broadcast(&self, dims: &[usize]) -> Result<TensorView<'a, T>>;
    fn diagonal(&self, axes: &[(usize, usize)]) -> Result<TensorView<'a, T>>;
    fn to_tensor(&self) -> Tensor<T>;
    fn contiguous(&self, order: MemoryOrder) -> Tensor<T>;
    fn conj(&self) -> Tensor<T>;
}
```

**einsum takes &Tensor references (not TensorView):**

```rust
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

pub fn einsum_owned<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: Vec<Tensor<T>>,
) -> Result<Tensor<T>>;
```

### Ownership Safety Examples

```rust
let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, ColumnMajor);
let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, ColumnMajor);

// einsum takes &Tensor — notation handles permutation/broadcast
let c = einsum("ij,jk->ik", &[&a, &b])?;
let c_t = einsum("ji,jk->ik", &[&a, &b])?;  // transposed a via notation

// View operations for CPU data inspection (waits if pending)
let at = a.permute(&[1, 0])?;           // TensorView borrowing a
assert_eq!(at.dims(), &[4, 3]);

// Compile-time safety: can't consume while borrowed
let at = a.permute(&[1, 0])?;
let d = einsum_owned("...", vec![a]);    // ERROR: at borrows a
drop(at);
let d = einsum_owned("...", vec![a])?;   // OK: borrow released
```

### Comparison with Arc-based Approach

| Aspect | TensorView (chosen) | Arc-based |
|--------|---------------------|-----------|
| Buffer uniqueness | Compile-time guarantee | Runtime `Arc::strong_count` check |
| `into_` buffer reuse | Always succeeds | May fail if views exist |
| API types | `Tensor` + `TensorView` | `Tensor` only |
| Lifetime complexity | Yes (`'a` propagates) | None |
| Runtime overhead | Zero | Atomic refcount on clone/drop |
| Rust idiom | `String`/`&str`, `Vec`/`&[T]` | `Arc<T>` |

The Arc approach remains viable if lifetime ergonomics prove too burdensome.

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
println!("{:?}", d.view());  // view() calls wait() internally
```

For CPU tensors, `event` is always `None` with zero overhead.

### Two-Tier API Contract

| Tier | Methods | Event handling | User visibility |
|------|---------|---------------|-----------------|
| **Public (CPU-read)** | `view()`, `permute()`, `broadcast()`, `diagonal()`, `view_mut()`, `to_tensor()`, `contiguous()`, `conj()` | **Wait** if pending, return ready data | Yes |
| **Internal (pipeline)** | `pub(crate) as_operand_view()` | **Propagate** event | No (crate-internal) |
| **Accelerator ops** | `einsum` (takes `&[&Tensor]`) | Calls `as_operand_view()` internally, **detects** events → stream dependency | Yes (transparent) |

### Applicability Beyond GPU

`CompletionEvent` applies equally to multi-threaded CPU execution:

- **Contraction tree parallelism**: Independent subtrees dispatched to
  different threads, each result carries a `CompletionEvent`.
- **User-level parallelism**: Independent `einsum` calls on separate threads
  with automatic event-based chaining.
- **NUMA-aware execution**: `ComputeDevice::Cpu { device_id }` maps to
  `ThreadPool` instances bound to specific core sets via `core_affinity`.

**Implementation note**: `wait(&self)` requires interior mutability
(e.g., `Cell<Option<CompletionEvent>>`) to clear the event field through
a shared reference.

### Alternatives Considered

1. **Separate `einsum_async`**: Rejected — splits the API unnecessarily.
2. **Trait-based (`TensorArg`)**: More extensible but adds API complexity.
   Can be introduced later backward-compatibly.
3. **Tree-level pipelining only**: Doesn't help for user-chained einsum calls.

**Current status**: POC `event` field is `Option<CompletionEvent>` (placeholder).
Actual async execution will be implemented with accelerator backends.

---

## Custom Element-Wise Operations

For arbitrary user functions not in `TensorPrims`, use strided-kernel
directly via `view()`:

```rust
let a_view = tensor_a.view();
let b_view = tensor_b.view();
strided_kernel::zip_map2_into(&mut out.view_mut(), &a_view, &b_view, |a, b| a * b + 1.0);
```
