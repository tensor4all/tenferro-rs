# External Linear Algebra Interop

tenferro keeps its dense tensors in column-major host or backend storage and
exposes that storage through stable borrow APIs. When an operation you need is
not yet available, suitable, or fast enough through tenferro's operation APIs,
you can call [faer] directly on tenferro-owned storage from a downstream
application. This page is the supported contract for that escape hatch. (A
vendor BLAS/LAPACK call is no longer part of it: the CPU stack links one faer
identity and re-exports no CBLAS or LAPACK bindings.)

`tenferro-cpu` re-exports the faer build it links as `tenferro_cpu::faer`, so
your code resolves the same `Par` and `Mat` types and one binary carries one
faer. No faer value crosses a tenferro boundary, so a direct `faer` dependency is
also safe; add one only when you want to control that build yourself, and then
keep it on the version below. tenferro does not wrap faer and does not add a
hidden interop layer to learn.

The example below is compiled and run in CI from
[`docs/tutorial-code`](https://github.com/tensor4all/tenferro-rs/tree/main/docs/tutorial-code)
as a downstream consumer that uses only public tenferro APIs:

```console
cargo run --manifest-path docs/tutorial-code/Cargo.toml --bin faer_interop
```

The binary asserts its numerical results, so a wrong layout or leading dimension
fails the run.

For direct faer work that should remain parallel, use the engine's placement
token. `CpuEngine::exec` lends the engine's provider bundle; a multi-thread
engine's token carries its `threadteam::Team`, and faer runs on it as
`Par::Team`. A token without a team (a serial engine, or a re-entrant call)
takes `Par::Seq`:

```text
let mut exec = engine.exec()?;
let mut token = exec.token();
if token.carries_team() {
    token.team(|team| faer_operation(..., Par::Team(team)))?;
} else {
    faer_operation(..., Par::Seq);
}
```

tenferro hands out no `faer::Par`, so no faer value crosses a tenferro boundary
and a separately pinned `faer` cannot meet a confusing type mismatch (#1939).
Spell the types with `tenferro_cpu::faer`, the re-exported build, to keep one
faer in the binary. That guarantee is for the slice-only route: a direct
dependency at another faer version is safe while only plain column-major slices
cross, and it does not cover the engine token above, where a
`threadteam::TeamExec` from the engine's own team enters faer's `Par::Team` and
the `threadteam` line has to match (see the dependency contract below). A
serial engine has no team and takes `Par::Seq` directly. This route is
CPU/faer-specific and does not claim affinity for external BLAS/OpenMP workers.

## When to use this

Use direct faer calls when tenferro's own APIs do not cover the routine you need
(a faer algorithm with no tenferro operation yet) or when you need faer's
specific algorithms. Prefer
tenferro's own operations otherwise: they manage provider selection, buffer
pools, and thread budgets for you. This escape hatch is not a
license to reimplement what `tenferro-linalg` already provides. On the host it
borrows tenferro storage directly; for CUDA buffers, the raw CUDA scope gives
checked device pointers instead (see
[Direct vendor calls on CUDA](#direct-vendor-calls-on-cuda)). Other GPU
backends (WebGPU/Metal) have no such contract.

## Dependency contract

Depend directly on the same major versions the tenferro workspace pins
([`Cargo.toml`](https://github.com/tensor4all/tenferro-rs/blob/main/Cargo.toml)):

| Crate | Tested version | Purpose |
|---|---|---|
| `faer` (`tenferro_cpu::faer`) | `t4a-faer` `0.33.0` | Pure-Rust column-major linear algebra; the fork the pinned `tlinalg` resolves, re-exported by `tenferro-cpu` |

`faer` is re-exported for convenience rather than required: tenferro hands out
no faer value, so a direct dependency at another compatible version only costs a
second faer build, not a type mismatch (#1939).

The exception is the `threadteam` line, which is `0.5.1` here and appears in
both crates' public types: `tenferro-cpu` re-exports `threadteam::Pool` as
`tenferro_cpu::Pool` (`crates/tenferro-cpu/src/lib.rs:204`), `CpuEngine::new`
takes it (`crates/tenferro-cpu/src/engine.rs:119`), and faer's `Par::Team`
carries a `threadteam::TeamExec` (`t4a-faer 0.33.0/src/lib.rs:1078`). What
matters is the `threadteam` version the two resolve to, not which `t4a-faer`
line they come from. `t4a-faer` 0.32.0 and 0.33.0 both require
`threadteam ^0.5.1`, so Cargo unifies them onto one `threadteam`: the `Pool` and
`TeamExec` are one type, and a `Par::Team` built from a team of that pool is
valid for either faer build. Only the `faer` types stay apart, so a `Par` from
one line cannot be passed where the other's `Par` is expected (`E0308`, two
`faer::Par` types). A line that needs another `threadteam` splits it instead —
`t4a-faer` 0.30 needs `threadteam ^0.4` — and then no team is shared at all:
the 0.4 and 0.5 `TeamExec` are different types, and neither a team nor a
`Par::Team` crosses (`E0308`, two `threadteam` versions in one graph). Depend on
the tested `t4a-faer` version above when a `Par`, a team, or the engine's pool
crosses the boundary.

For faer there is no feature to enable and no system dependency: the CPU stack
links exactly one faer build.

## Layout and leading dimensions

tenferro dense tensors are compact column-major, exactly like Fortran and
`faer`: the leftmost dimension varies fastest in memory and the leading
dimension of an `m x n` matrix is `m`. See
[Memory Order](memory-order.md) for the storage convention.

Before the contiguous fast path, check both properties explicitly. The faer
immutable-view example below does exactly this before borrowing: it asserts
`is_col_major_contiguous()` and checks `placement().memory_kind` is a host
kind, so the zero-copy slice it hands to faer is provably compact column-major.

A non-contiguous view (transpose, slice, broadcast) is rejected by `as_slice()`
with an explicit error and is never silently materialized. A backend-placed
tensor is rejected the same way and is never implicitly downloaded. The caller
performs the explicit copy or transfer: `Engine::to_contiguous_read` for host
non-contiguous views, and the owning engine's explicit download API for backend
buffers (`CpuEngine` rejects foreign backend buffers, so the tensor's own engine
must perform the transfer).

## faer: zero-copy views over tenferro storage

<!-- snippet-source: docs/tutorial-code/src/bin/faer_interop.rs#faer-immutable -->
```rust
fn faer_immutable_view() -> tenferro_tensor::Result<()> {
    // Only a compact column-major host tensor can be borrowed without copying.
    // Check both properties explicitly before using the contiguous path.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    assert!(a.is_col_major_contiguous()?);
    assert_eq!(a.placement().memory_kind, MemoryKind::UnpinnedHost);

    // Zero-copy faer view. Column-major compact storage means the leading
    // dimension is the number of rows and the flat slice is the whole buffer.
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let a_view: MatRef<f64> = MatRef::from_column_major_slice(a.as_slice()?, rows, cols);
    assert_eq!(*a_view.get(1, 2), 6.0); // row 1, column 2

    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0])?;
    let b_view = MatRef::from_column_major_slice(b.as_slice()?, b.shape()[0], b.shape()[1]);

    // faer computes into its own output. Par::Seq keeps the call on this thread;
    // a parallel width comes from an engine token (see faer-engine-parallelism).
    let mut product = Mat::<f64>::zeros(rows, b.shape()[1]);
    matmul(
        product.as_mut(),
        Accum::Replace,
        a_view,
        b_view,
        1.0,
        Par::Seq,
    );

    // A * B for A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]:
    // C = [[58,64],[139,154]]. Positions double as the column-major check:
    // the buffer is [58,139,64,154] (column 0 first, rows fastest).
    let product = product.as_ref();
    assert_close(*product.get(0, 0), 58.0, "C00");
    assert_close(*product.get(1, 0), 139.0, "C10");
    assert_close(*product.get(0, 1), 64.0, "C01");
    assert_close(*product.get(1, 1), 154.0, "C11");
    Ok(())
}
```
<!-- end-snippet-source -->

<!-- snippet-source: docs/tutorial-code/src/bin/faer_interop.rs#faer-mutable -->
```rust
fn faer_mutable_view() -> tenferro_tensor::Result<()> {
    let mut x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;

    // Writable faer view over tenferro-owned host storage. The mutable borrow
    // is exclusive, so no other reference to the buffer can be live. Scoping
    // the view ends the borrow before the tensor is read again below.
    {
        let mut x_view: MatMut<f64> = MatMut::from_column_major_slice_mut(x.host_data_mut()?, 2, 2);
        for column in x_view.as_mut().col_iter_mut() {
            for element in column.iter_mut() {
                *element *= 2.0;
            }
        }
    }

    // The write happened in place: the tensor now owns the scaled buffer.
    assert_slice_close(x.as_slice()?, &[2.0, 4.0, 6.0, 8.0], "scaled x");
    Ok(())
}
```
<!-- end-snippet-source -->

<!-- snippet-source: docs/tutorial-code/src/bin/faer_interop.rs#faer-solve -->
```rust
fn faer_solve_on_zero_copy_views() -> tenferro_tensor::Result<()> {
    // Solve A x = b with faer's LU, reading A and b straight from tenferro
    // storage and writing x into faer-owned memory.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])?;
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![5.0, 4.0])?;
    let a_view = MatRef::from_column_major_slice(a.as_slice()?, 2, 2);
    let b_view = MatRef::from_column_major_slice(b.as_slice()?, 2, 1);

    let lu = a_view.partial_piv_lu(Par::Seq);
    let mut x = Mat::<f64>::zeros(2, 1);
    x.as_mut().copy_from(b_view);
    lu.solve_in_place(x.as_mut(), Par::Seq);

    // Verify A x == b. Exact solution: x = [1.2, 1.4] (3*1.2+1.4 == 5,
    // 1.2+2*1.4 == 4). Check both the solution and the reconstruction.
    assert_close(*x.get(0, 0), 1.2, "x0");
    assert_close(*x.get(1, 0), 1.4, "x1");
    let residual0 = *a_view.get(0, 0) * *x.get(0, 0) + *a_view.get(0, 1) * *x.get(1, 0);
    let residual1 = *a_view.get(1, 0) * *x.get(0, 0) + *a_view.get(1, 1) * *x.get(1, 0);
    assert_close(residual0, 5.0, "A x row 0");
    assert_close(residual1, 4.0, "A x row 1");
    Ok(())
}
```
<!-- end-snippet-source -->

### Non-contiguous and non-host storage

<!-- snippet-source: docs/tutorial-code/src/bin/faer_interop.rs#faer-non-contiguous -->
```rust
fn faer_rejects_non_contiguous_without_materializing() -> tenferro_tensor::Result<()> {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    // Transposing a 2x3 tensor produces a 3x2 view with non-column-major
    // strides. It is metadata only; nothing has been copied.
    let transposed: TypedTensorView<'_, f64> = a.as_view().transpose_view([1, 0])?;
    assert!(!transposed.is_col_major_contiguous()?);

    // as_slice rejects the view explicitly. There is no hidden materialization.
    let error = transposed.as_slice().unwrap_err();
    assert!(
        error.to_string().contains("not slice-contiguous"),
        "unexpected error: {error}"
    );

    // The caller decides to materialize through an explicit engine call.
    let mut engine = CpuEngine::serial();
    let read = transposed.into_tensor_read()?;
    let compact = engine.to_contiguous_read(read)?;
    let compact = compact
        .as_typed::<f64>()
        .expect("materialized an f64 tensor");
    assert!(compact.is_col_major_contiguous()?);
    // 3x2 column-major copy of the transposed matrix [[1,4],[2,5],[3,6]]:
    // column 0 = [1,2,3], column 1 = [4,5,6].
    assert_slice_close(
        compact.as_slice()?,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "compact",
    );
    Ok(())
}
```
<!-- end-snippet-source -->

<!-- snippet-source: docs/tutorial-code/src/bin/faer_interop.rs#faer-non-host -->
```rust
fn faer_rejects_non_host_storage() -> tenferro_tensor::Result<()> {
    // A backend-placed tensor (for example GPU memory, or an external provider
    // allocation) cannot be borrowed as a host slice. Construct one with the
    // public from_buffer_col_major API to show the error contract.
    let backend_buffer: Box<dyn BackendStorage<f64>> =
        Box::new(BackendStorageHandle::<f64>::new_with_len(7, 4));
    let device_tensor = TypedTensor::<f64>::from_buffer_col_major(
        vec![2, 2],
        StorageBuffer::Backend(backend_buffer),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Other("example".into()),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )?;
    assert_eq!(device_tensor.placement().memory_kind, MemoryKind::Device);

    // Host access is rejected explicitly; tenferro never downloads on borrow.
    let error = device_tensor.as_slice().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("backend storage cannot be borrowed as host data"),
        "unexpected error: {error}"
    );

    // The required transfer is an explicit download through the owning engine
    // (CpuEngine rejects foreign backend buffers); only then can faer read the
    // data.
    Ok(())
}
```
<!-- end-snippet-source -->

`faer::MatRef::from_column_major_slice` and
`faer::MatMut::from_column_major_slice_mut` assume the buffer starts at the
first element of the first column with column stride equal to the row count.
That is exactly tenferro's compact column-major contract, which is why the
compactness check above is a precondition, not a formality. faer never takes
ownership: the `MatRef`/`MatMut` borrows live as long as your slice borrow, and
Rust's borrow checker prevents you from holding a `MatMut` while a tenferro
`host_data()` borrow is still live.

## Threading and resource domains

Tenferro operations on a `CpuEngine` run on the pool the caller built for it:
its width, its buffer pool, and (for faer) its selected faer parallelism. Direct
external calls deliberately sit outside that context:

- A direct faer call with `Par::Seq` runs inline on the calling thread.
- A direct faer call that does not use the engine's token (`Par::rayon(n)`, for
  example) uses rayon's **global** thread pool. That is not the team the engine
  computes on: the engine uses only the pool the caller passed to
  `CpuEngine::new`. Use the engine's token (`Par::Team`, above) to run on the
  engine's team and within its budget.

The consequence is that your application owns the coordination: if your outer
loop already runs many tenferro calls in parallel, do not also fan out each
direct call with a large thread budget. See
[Parallelism and Caching](parallelism-and-caching.md) for the oversubscription
rules that apply equally here. If you need the engine's own pool and thread
budget, use tenferro's own operations instead of direct calls.

## Direct vendor calls on CUDA

The CUDA counterpart of the host escape hatch is the raw CUDA scope
(`CudaEngine::with_raw`, issue #1597). Inside it a downstream crate can
call cuBLAS, cuSOLVER or another CUDA library on tenferro-owned device buffers,
on the engine's stream, using only public APIs:

<!-- snippet-source: docs/tutorial-code/src/bin/cuda_vendor_interop.rs#cuda-vendor-cublas -->
```rust
use tenferro_gpu::cuda::cudarc::cublas::{result as cublas, sys as cublas_sys};

/// The cuBLAS handle a session keeps for its engine, bound to the session's stream.
struct CublasHandle(*mut cublas_sys::cublasContext);

// SAFETY: the handle is only ever used from the session that created it, on that session's stream,
// and the engine that caches it is `Send` and not shared.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    fn new(stream: u64) -> tenferro_tensor::Result<Self> {
        // `cudarc` creates the handle and hands back its raw pointer.
        let raw = cublas::create_handle().map_err(|err| Error::backend_source(OP, err))?;
        let handle = CublasHandle(raw);
        // SAFETY: `handle.0` came from `cublasCreate` and stays alive in the engine's cache.
        unsafe { cublas_sys::cublasSetStream_v2(handle.0, stream as *mut _).result() }
            .map_err(|err| Error::backend_source(OP, err))?;
        // SAFETY: same handle; the host pointer mode matches the scalars passed to `dgemm`.
        unsafe {
            cublas_sys::cublasSetPointerMode_v2(
                handle.0,
                cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            )
            .result()
        }
        .map_err(|err| Error::backend_source(OP, err))?;
        Ok(handle)
    }

    fn raw(&self) -> *mut cublas_sys::cublasContext {
        self.0
    }
}

/// `C = A * B` with cuBLAS `dgemm` on tenferro device tensors.
///
/// `a` is `m x k` and `b` is `k x n`; both are compact column-major CUDA
/// tensors resident on `backend`. The output is allocated by tenferro and
/// written by cuBLAS on tenferro's stream.
fn cublas_dgemm(
    engine: &mut CudaEngine,
    a: &TypedTensor<f64>,
    b: &TypedTensor<f64>,
) -> tenferro_tensor::Result<TypedTensor<f64>> {
    let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
    // cuBLAS takes LP64 `int` dimensions; convert instead of casting.
    let dim = |value: usize| {
        i32::try_from(value)
            .map_err(|_| Error::invalid_argument(OP, "shape", "dimension exceeds i32"))
    };
    let (m32, k32, n32) = (dim(m)?, dim(k)?, dim(n)?);

    engine.with_raw(OP, |session| {
        let a_ref = session.tensor(a)?;
        let b_ref = session.tensor(b)?;
        let c = session.alloc_output::<f64>(&[m, n])?;
        let c_ref = session.tensor_mut(&c)?;
        let (alpha, beta) = (1.0_f64, 0.0_f64);

        // A vendor handle is a resource: the engine creates it once, binds it to this scope's
        // stream and lends it back for the session's life, so a hot loop pays neither
        // `cublasCreate` nor a re-bind.
        let stream = session.stream();
        let handle = CublasHandle::new(stream.raw_handle())?;
        // SAFETY: the three device pointers come from checked tensor references that stay
        // borrowed until the `synchronize` below; A is m x k (lda = m), B is k x n (ldb = k) and C
        // is m x n (ldc = m) in compact column-major storage, which is cuBLAS's native layout. C
        // is the only buffer written and does not alias A or B.
        unsafe {
            cublas::dgemm(
                handle.raw(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                m32,
                n32,
                k32,
                &alpha,
                a_ref.raw_ptr().cast(),
                m32,
                b_ref.raw_ptr().cast(),
                k32,
                &beta,
                c_ref.raw_ptr().cast(),
                m32,
            )
            .map_err(|err| Error::backend_source(OP, err))?;
        }
        // `synchronize` is the only host barrier: the GEMM was merely
        // enqueued on the engine's stream.
        session.synchronize()?;
        Ok(c)
    })
}
```
<!-- end-snippet-source -->

The example runs in the GPU CI lane (through `cuda_tutorial`) and on its own:

```console
cargo run --manifest-path docs/tutorial-code/Cargo.toml \
  --no-default-features --features cuda --bin cuda_vendor_interop
```

### Dependency contract

| Crate | Source | Purpose |
|---|---|---|
| `cudarc` | `tenferro_gpu::cuda::cudarc` (re-export, `0.19`) | cuBLAS/driver/NVRTC bindings matching tenferro's CUDA build |

Unlike the host crates above, tenferro-gpu **does** re-export `cudarc`: the
vendor bindings must load the same CUDA libraries for the same CUDA version
selection (`cuda-12080`, `dynamic-loading`) that tenferro uses, and taking them
from the re-export makes a version or feature mismatch impossible. The
re-export enables `driver`, `runtime`, `nvrtc` and `cublas`. For another module,
such as `cusolver`, add `cudarc = { version = "0.19", default-features = false,
features = ["cusolver", "dynamic-loading", "cuda-12080"] }` to your own
manifest; Cargo unifies it with tenferro's copy, so the re-export then exposes
that module too.

### Stream, handle and lifetime rules

- **Stream.** Issue every vendor call on `session.stream().raw_handle()`.
  Work on that stream is ordered after the tenferro work that produced the
  inputs and before the tenferro work that later reads the output, so no extra
  events are needed. Do not use another stream: tenferro does not order work on
  streams it does not own.
- **Pointers.** `session.tensor(&t)` / `session.tensor_mut(&mut t)` validate
  that the tensor is resident on this session's device and return its device
  span; `raw_ptr()` is valid only while that reference (and the session) is
  borrowed. Outputs come from `session.alloc_output`, so tenferro owns and
  frees them. tenferro tensors are compact column-major, so the leading
  dimension of an `m x n` matrix is `m`, exactly as for host BLAS.
- **Barrier.** Successful raw-session calls only enqueue.
  `session.synchronize()` is the host barrier; call it before destroying a
  vendor handle or workspace that queued work still uses, and before reading
  results on the host (or use `download_tensor`, which orders itself).
- **Handles.** Vendor handles (`cublasHandle_t`, `cusolverDnHandle_t`) belong
  to the caller. Create them while the session is active (tenferro's primary
  context is current), bind them to the session's stream with `set_stream` in
  every session that uses them (the stream belongs to the session, not to the
  handle), and destroy them only after a barrier.
- **What is not shared.** tenferro's own cuBLAS, cuSOLVER and cuTENSOR handles
  and workspaces (for example the handle behind `dot_general`) are internal;
  they are not exposed and must not be assumed reusable. Work issued from
  another thread runs on that thread's stream and is outside this contract.

## Related work

- [#1536](https://github.com/tensor4all/tenferro-rs/issues/1536) covers the
  opposite direction: exposing faer-owned storage to tenferro/strided kernels.
- [#1555](https://github.com/tensor4all/tenferro-rs/issues/1555) tracks the
  broader storage ownership redesign; if that work changes the borrow APIs used
  here, the examples above are the checked surface that must be updated with it.

[faer]: https://crates.io/crates/faer
