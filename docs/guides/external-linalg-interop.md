# External Linear Algebra Interop

tenferro keeps its dense tensors in column-major host or backend storage and
exposes that storage through stable borrow APIs. When an operation you need is
not yet available, suitable, or fast enough through tenferro's operation APIs,
you can call [faer] or BLAS/LAPACK directly on tenferro-owned storage from a
downstream application. This page is the supported contract for that escape
hatch.

The contract is **direct dependencies**: your application adds `faer`, and for
BLAS/LAPACK `cblas-sys` + `lapack`, at the tested compatible versions listed
below, and enables the matching CPU provider feature on the tenferro crates.
tenferro does not re-export these crates and does not wrap them; there is no
hidden interop layer to learn.

The examples below are compiled and run in CI from
[`docs/tutorial-code`](https://github.com/tensor4all/tenferro-rs/tree/main/docs/tutorial-code)
as a downstream consumer that uses only public tenferro APIs:

```console
# faer example (no system dependencies)
cargo run --manifest-path docs/tutorial-code/Cargo.toml --bin faer_interop

# BLAS/LAPACK example with exactly one provider feature
cargo run --manifest-path docs/tutorial-code/Cargo.toml --features blas-openblas --bin blas_interop
```

Both binaries assert their numerical results, so a wrong layout or leading
dimension fails the run. In CI the BLAS binary is additionally run against the
system OpenBLAS (`blas` with the native library linked through
`RUSTFLAGS`) so that native symbol linkage is verified without rebuilding
OpenBLAS from source.

For direct faer work that should remain parallel, import
`tenferro_cpu::FaerParallelismExt` and scope the call inside the backend
session:

```text
backend.with_backend_session(|session| {
    session.with_faer_parallelism(|parallel| {
        faer_operation(..., parallel)
    })
})??;
```

This capability supplies the session-selected `faer::Par`: managed multi-thread
inner execution receives bounded `Par::rayon(n)`, while one-thread or nested
execution receives `Par::Seq`. The callback cannot retain the capability or
session executor. A direct `Par::Seq` call remains the simplest portable escape
hatch; an ambient `Par::rayon(n)` is outside tenferro's thread-budget and
affinity contract. The scoped capability is CPU/faer-specific and does not
claim affinity for external BLAS/OpenMP workers.

## When to use this

Use direct external calls when tenferro's own APIs do not cover the routine you
need (an exotic LAPACK driver), when you already maintain a tuned system BLAS
you want to reuse, or when you need faer's specific algorithms. Prefer
tenferro's own operations otherwise: they manage provider selection, buffer
pools, placement, and thread budgets for you. This escape hatch is not a
license to reimplement what `tenferro-linalg` already provides. On the host it
borrows tenferro storage directly; for CUDA buffers, the raw CUDA session gives
checked device pointers instead (see
[Direct vendor calls on CUDA](#direct-vendor-calls-on-cuda)). Other GPU
backends (WebGPU/Metal) have no such contract.

## Dependency contract

Depend directly on the same major versions the tenferro workspace pins
([`Cargo.toml`](https://github.com/tensor4all/tenferro-rs/blob/main/Cargo.toml)):

| Crate | Tested version | Purpose |
|---|---|---|
| `faer` | `0.24` | Pure-Rust column-major linear algebra |
| `cblas-sys` | `0.1` | CBLAS entry points (`cblas_dgemm`, ...) |
| `lapack` | `0.20` | Fortran LAPACK entry points (`dpotrf`, ...) |

`cblas-sys` is pre-1.0, so pin the exact tested minor (`0.1.x`) rather than
expecting semver compatibility from `0.1` alone. tenferro does not re-export
these crates on purpose: keeping the external library a normal direct
dependency means your application controls the version, the provider link, and
the threading environment exactly as if tenferro were not involved.

### Provider features

For faer, the default `native` feature is enough; faer has no system
dependency. For BLAS/LAPACK, enable `blas` plus **exactly one** provider
feature on the tenferro crates:

| Provider | tenferro feature | Library |
|---|---|---|
| OpenBLAS | `blas-openblas` | OpenBLAS (built or linked by the provider crates) |
| Apple Accelerate | `blas-accelerate` | Accelerate framework |
| Intel MKL | `blas-mkl` | Intel oneAPI MKL |

```toml
# The working downstream combination used by docs/tutorial-code (blas
# already implies the cblas-sys + lapack direct dependencies).
tenferro-cpu = { path = "crates/tenferro-cpu", default-features = false, features = ["blas", "blas-openblas"] }
cblas-sys = "0.1"
lapack = "0.20"
openblas-src = "0.10"
```

`openblas-src` is listed (and referenced in the example) because the
from-source provider builds the native library from its build script: its
library target is empty, so it must be linked explicitly for cargo to forward
the native library. tenferro-cpu does the same for `blas-src`/`lapack-src`
internally. If you link a system OpenBLAS yourself instead (for example with
`RUSTFLAGS='-l dylib=openblas -l dylib=lapack'`), you do not need the
`blas-openblas` feature or `openblas-src` at all — just `blas`.

The provider features are additive, but the native provider crates reject
simultaneous selections at build time, so never enable two of them together.
`blas` without a provider feature compiles tenferro but does not link any
native library; you must supply the symbols yourself (for example CI links the
system OpenBLAS with `RUSTFLAGS='-l dylib=openblas -l dylib=lapack'`).

## Layout and leading dimensions

tenferro dense tensors are compact column-major, exactly like Fortran, LAPACK,
and CBLAS `CblasColMajor`: the leftmost dimension varies fastest in memory and
the leading dimension of an `m x n` matrix is `m`. See
[Memory Order](memory-order.md) for the storage convention.

Before the contiguous fast path, check both properties explicitly. The faer
immutable-view example below does exactly this before borrowing: it asserts
`is_col_major_contiguous()` and checks `placement().memory_kind` is a host
kind, so the zero-copy slice it hands to faer is provably compact column-major.

A non-contiguous view (transpose, slice, broadcast) is rejected by `as_slice()`
with an explicit error and is never silently materialized. A backend-placed
tensor is rejected the same way and is never implicitly downloaded. The caller
performs the explicit copy or transfer: the session's `to_contiguous_read` for
host non-contiguous views, and the owning backend's explicit download API for
backend buffers (`CpuBackend` itself rejects foreign backend buffers, so the
tensor's own backend must perform the transfer).

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

    // faer computes into its own output. Par::Seq keeps the call off faer's
    // global rayon pool; Par::Rayon(n) would use that ambient pool instead.
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

    let lu = a_view.partial_piv_lu();
    let mut x = Mat::<f64>::zeros(2, 1);
    x.as_mut().copy_from(b_view);
    lu.solve_in_place(x.as_mut());

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

    // The caller decides to materialize through an explicit backend session.
    let mut backend = CpuBackend::new();
    let read = transposed.into_tensor_read()?;
    let compact = backend.with_backend_session(|session| session.to_contiguous_read(read))??;
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

    // The required transfer is an explicit download through the owning
    // backend (CpuBackend rejects foreign backend buffers); only then can faer
    // read the data.
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

## BLAS and LAPACK: direct calls on host storage

The BLAS example enables exactly one provider feature, passes tenferro host
storage to CBLAS with `CblasColMajor` and correct leading dimensions, and
writes into a tenferro-owned mutable output tensor:

<!-- snippet-source: docs/tutorial-code/src/bin/blas_interop.rs#blas-dgemm -->
```rust
fn blas_dgemm_on_host_storage() -> tenferro_tensor::Result<()> {
    // A (2x3) and B (3x2) are compact column-major host tensors. For
    // CblasColMajor, the leading dimension of an m x n matrix is m.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    let b =
        TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0])?;
    let mut c = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![0.0; 4])?;
    assert!(a.is_col_major_contiguous()? && b.is_col_major_contiguous()?);
    assert!(c.is_col_major_contiguous()?);

    let m = dim_i32(a.shape()[0], "m");
    let k = dim_i32(a.shape()[1], "k");
    let n = dim_i32(b.shape()[1], "n");
    let lda = m; // A is m x k, column-major
    let ldb = k; // B is k x n, column-major
    let ldc = m; // C is m x n, column-major

    // Snapshot the inputs so we can prove the call does not mutate them.
    let a_before = a.as_slice()?.to_vec();
    let b_before = b.as_slice()?.to_vec();

    // SAFETY: a, b, c are live for the duration of the call, their slices are
    // at least m*k, k*n, m*n elements, and the leading dimensions match the
    // column-major layouts verified above. C is the only mutated buffer.
    unsafe {
        cblas_sys::cblas_dgemm(
            CBLAS_LAYOUT::CblasColMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m,
            n,
            k,
            1.0,
            a.as_slice()?.as_ptr(),
            lda,
            b.as_slice()?.as_ptr(),
            ldb,
            0.0,
            c.host_data_mut()?.as_mut_ptr(),
            ldc,
        );
    }

    // Inputs are untouched; the output holds A * B in tenferro's storage.
    assert_slice_close(a.as_slice()?, &a_before, "A unchanged");
    assert_slice_close(b.as_slice()?, &b_before, "B unchanged");
    // C = [[58,64],[139,154]] stored column-major.
    assert_slice_close(c.as_slice()?, &[58.0, 139.0, 64.0, 154.0], "C=A*B");
    Ok(())
}
```
<!-- end-snippet-source -->

LAPACK works the same way through the `lapack` crate; `dpotrf` factors in place
into the mutable tenferro buffer:

<!-- snippet-source: docs/tutorial-code/src/bin/blas_interop.rs#blas-potrf -->
```rust
fn lapack_potrf_on_host_storage() -> tenferro_tensor::Result<()> {
    // A is a 2x2 SPD matrix, column-major [3,1,1,2]. dpotrf factors in place:
    // the lower triangle of the buffer is replaced by L with A = L L^T.
    let mut a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![3.0, 1.0, 1.0, 2.0])?;
    let original = a.as_slice()?.to_vec();
    let n = dim_i32(a.shape()[0], "n");
    let lda = n;
    let mut info: i32 = 0;

    // SAFETY: a holds a live n*n column-major buffer and info is a live output.
    unsafe {
        lapack::dpotrf(b'L', n, a.host_data_mut()?, lda, &mut info);
    }
    assert_eq!(info, 0, "dpotrf failed with info={info}");

    // Reconstruct A = L L^T from the factor (L is unit-free, stored in the
    // lower triangle; the upper triangle is left untouched).
    let data = a.as_slice()?;
    let (l00, l10, l11) = (data[0], data[1], data[3]);
    assert_slice_close(
        &[l00 * l00, l00 * l10, l00 * l10, l10 * l10 + l11 * l11],
        &original,
        "L L^T",
    );
    Ok(())
}
```
<!-- end-snippet-source -->

### LP64/ILP64, complex ABI, and provider thread control

- **LP64/ILP64.** `cblas-sys` and `lapack` use the LP64 convention: dimensions
  and leading dimensions are 32-bit `c_int` (`i32` in Rust). Convert sizes with
  `i32::try_from` and handle overflow instead of casting blindly. There is no
  ILP64 variant of these crates in the tested configuration.
- **Complex ABI.** CBLAS complex entry points take interleaved real/imaginary
  pairs. `cblas-sys` models a complex value as `[f64; 2]`
  (`c_double_complex`); `num_complex::Complex<f64>` is layout-compatible, so a
  `Complex64` slice can be passed where a `*const c_double_complex` is
  expected after an explicit pointer cast. Unlike real entry points, complex
  CBLAS entry points take `alpha` and `beta` **by pointer**. tenferro uses the
  same `num_complex` types for its `C64` dtypes. Do not reinterpret a complex
  slice as twice as many reals and hand that to a complex routine.
- **Provider thread control.** Direct BLAS/LAPACK calls run on the provider's
  own threads, controlled by process-wide environment variables
  (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, and
  often `OMP_NUM_THREADS`). These are read at process start; set them before
  launching your application. Direct calls do not join tenferro's CPU execution
  domain. Their threads inherit the calling thread's CPU mask, so a call made
  from a tenferro worker starts inside that domain's CPU set, but tenferro
  cannot control the provider's fan-out or an affinity policy the provider
  installs itself.

## Threading and resource domains

Tenferro operations on a `CpuBackend` run inside that backend's execution
context: its thread count, buffer pool, and (for faer) its selected faer
parallelism. Direct external calls deliberately sit outside that context:

- A direct faer call with `Par::Seq` runs inline on the calling thread.
- A direct faer call with `Par::Rayon(n)` uses rayon's **global** thread pool.
  That is not the pool tenferro's faer-backed kernels use: `CpuContext`
  constructs its own Rayon pools scoped to the requested thread count and the
  domain CPU set. A direct call therefore does not participate in tenferro's CPU
  resource domain at all — not its pool, not its budget, not its affinity.
- A direct BLAS/LAPACK call runs on the provider's own threads as described
  above.

The consequence is that your application owns the coordination: if your outer
loop already runs many tenferro calls in parallel, do not also fan out each
direct call with a large thread budget. See
[Parallelism and Caching](parallelism-and-caching.md) for the oversubscription
rules that apply equally here. If you need tenferro-managed NUMA placement,
use tenferro's own operations instead of direct calls.

## Direct vendor calls on CUDA

The CUDA counterpart of the host escape hatch is the raw CUDA session
(`CudaExecSession::with_raw`, issue #1597). Inside it a downstream crate can
call cuBLAS, cuSOLVER or another CUDA library on tenferro-owned device buffers,
on tenferro's execution stream, using only public APIs:

<!-- snippet-source: docs/tutorial-code/src/bin/cuda_vendor_interop.rs#cuda-vendor-cublas -->
```rust
use tenferro_gpu::cuda::cudarc::cublas::{result as cublas, sys as cublas_sys};

/// `C = A * B` with cuBLAS `dgemm` on tenferro device tensors.
///
/// `a` is `m x k` and `b` is `k x n`; both are compact column-major CUDA
/// tensors resident on `backend`. The output is allocated by tenferro and
/// written by cuBLAS on tenferro's stream.
fn cublas_dgemm(
    backend: &mut CudaBackend,
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

    backend.with_backend_session(|backend_session| {
        with_cuda_exec_session(backend_session, |cuda| {
            cuda.with_raw(OP, |session| {
                let a_ref = session.tensor(a)?;
                let b_ref = session.tensor(b)?;
                let mut c = session.alloc_output::<f64>(&[m, n])?;
                let c_ref = session.tensor_mut(&mut c)?;
                let (alpha, beta) = (1.0_f64, 0.0_f64);

                // The handle is created inside the session, bound to the
                // session's stream, and destroyed after the host barrier. A
                // longer-lived handle must be re-bound with `set_stream` in
                // every session: the stream belongs to the session, not the
                // handle.
                let handle =
                    cublas::create_handle().map_err(|err| Error::backend_source(OP, err))?;
                // SAFETY: `raw_handle` is the session's live CUDA stream and is
                // used only inside this session. The three device pointers come
                // from checked tensor references that stay borrowed until the
                // `synchronize` below; A is m x k (lda = m), B is k x n
                // (ldb = k) and C is m x n (ldc = m) in compact column-major
                // storage, which is cuBLAS's native layout. C is the only
                // buffer written and does not alias A or B.
                let launched = unsafe {
                    cublas::set_stream(
                        handle,
                        session.stream().raw_handle() as cublas_sys::cudaStream_t,
                    )
                    .and_then(|()| {
                        cublas::dgemm(
                            handle,
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
                    })
                };
                // `synchronize` is the only host barrier: the GEMM was merely
                // enqueued on the session's stream.
                let synchronized = session.synchronize();
                // SAFETY: the handle is not used after this point, and the
                // barrier above (or the failed enqueue) means no work using it
                // is still pending on the stream.
                let destroyed = unsafe { cublas::destroy_handle(handle) };
                launched.map_err(|err| Error::backend_source(OP, err))?;
                synchronized?;
                destroyed.map_err(|err| Error::backend_source(OP, err))?;
                Ok(c)
            })
        })
        .ok_or_else(|| Error::runtime_state(OP, "backend session is not CUDA"))?
    })?
}
```
<!-- end-snippet-source -->

The example runs in the GPU CI lane (through `cuda_tutorial`) and on its own:

```console
cargo run --manifest-path docs/tutorial-code/Cargo.toml \
  --no-default-features --features cuda,native --bin cuda_vendor_interop
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
