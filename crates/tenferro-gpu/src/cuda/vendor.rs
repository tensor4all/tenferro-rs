//! Vendor operations on the runtime's own stream: cuBLAS GEMM, strided-batched GEMM, and
//! the conjugate dot BLAS-1 folds its callers into.
//!
//! The handle is created once per runtime with host pointer mode and the runtime's
//! stream, and every call runs inside the runtime's raw scope, so the operands are
//! registered, the addresses belong to this runtime and the library call is ordered with
//! the kernels that produced or will consume its operands.

use cudarc::cublas::sys as cublas;
use t4a_cuda_kernels::DType;
use t4a_cuda_runtime::{Buffer as DeviceBuffer, RawResource};

use crate::gpu::GpuError;

use super::runtime::CudaRuntime;

/// One GEMM request over runtime allocations, in column-major order.
#[derive(Debug)]
#[doc(hidden)]
pub struct GemmRequest<'a> {
    /// Dtype of all three operands.
    pub dtype: DType,
    /// Left operand and whether it is transposed.
    pub a: &'a DeviceBuffer,
    /// Whether `a` is transposed.
    pub trans_a: bool,
    /// Right operand.
    pub b: &'a DeviceBuffer,
    /// Whether `b` is transposed.
    pub trans_b: bool,
    /// Destination.
    pub c: &'a DeviceBuffer,
    /// `(m, n, k)` of the product.
    pub shape: (i64, i64, i64),
    /// Leading dimensions `(lda, ldb, ldc)` in elements.
    pub leading: (i64, i64, i64),
    /// `(batches, stride_a, stride_b, stride_c)` in elements; `batches = 1` is a plain
    /// GEMM.
    pub batches: (i64, i64, i64, i64),
}

/// A cuBLAS handle bound to one runtime's stream.
#[doc(hidden)]
pub struct Cublas {
    handle: cublas::cublasHandle_t,
    runtime: CudaRuntime,
    /// Set when a pointer-mode change failed, which leaves the handle's mode unproven.
    ///
    /// Every entry reads its coefficients in the mode it sets, so a handle with an unproven mode
    /// would let a GEMM read stack coefficients as device addresses. The engine replaces such a
    /// handle instead of reusing it.
    mode_unproven: bool,
}

// SAFETY: a handle carries no submission authority of its own; every call runs inside
// the owning runtime's raw scope on its stream.
unsafe impl Send for Cublas {}

impl Cublas {
    /// Create the handle, set host pointer mode and bind the runtime's stream.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unavailable`] when cuBLAS cannot be loaded or initialised, and the
    /// runtime's typed admission failure.
    pub fn new(runtime: &CudaRuntime) -> Result<Cublas, GpuError> {
        // Every entry point below resolves a symbol through the loader, and a library that is
        // not there answers a resolution by panicking. An operation path that requires cuBLAS
        // must report the typed load error the GPU backend contract asks for, so the library is
        // probed before any symbol is bound.
        // SAFETY: the probe opens the library and binds nothing; it enqueues no work.
        if !unsafe { cublas::is_culib_present() } {
            return Err(GpuError::Unavailable {
                detail: "cuBLAS is not loadable: no libcublas candidate dlopens".to_string(),
            });
        }
        let handle = {
            // SAFETY: the raw scope activates the owning context, so the handle is created
            // in it and bound to this runtime's stream; nothing is enqueued.
            unsafe {
                runtime
                    .with_raw_locked(Vec::new(), |raw| {
                        let mut handle = std::mem::MaybeUninit::<cublas::cublasHandle_t>::uninit();
                        let status = cublas::cublasCreate_v2(handle.as_mut_ptr());
                        if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            return Err(GpuError::Unavailable {
                                detail: format!("cublasCreate_v2 failed: {status:?}"),
                            });
                        }
                        let handle = handle.assume_init();
                        let status = cublas::cublasSetPointerMode_v2(
                            handle,
                            cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                        );
                        if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            let _ = cublas::cublasDestroy_v2(handle);
                            return Err(GpuError::Unavailable {
                                detail: format!("cublasSetPointerMode_v2 failed: {status:?}"),
                            });
                        }
                        let stream = raw.stream() as usize as cublas::cudaStream_t;
                        let status = cublas::cublasSetStream_v2(handle, stream);
                        if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            let _ = cublas::cublasDestroy_v2(handle);
                            return Err(GpuError::Unavailable {
                                detail: format!("cublasSetStream_v2 failed: {status:?}"),
                            });
                        }
                        Ok(handle)
                    })
                    .map_err(|error| GpuError::Unavailable {
                        detail: error.to_string(),
                    })?
            }
        }?;
        Ok(Cublas {
            handle,
            runtime: runtime.clone(),
            mode_unproven: false,
        })
    }

    /// Refuse a handle whose pointer mode a failed change left unproven.
    fn ensure_mode_proven(&self) -> Result<(), GpuError> {
        if self.mode_unproven {
            return Err(GpuError::Unavailable {
                detail: "the cuBLAS handle's pointer mode is unproven after a failed \
                         pointer-mode change; the engine recreates it"
                    .to_string(),
            });
        }
        Ok(())
    }

    /// Whether a failed pointer-mode change left this handle unusable.
    pub(crate) fn mode_unproven(&self) -> bool {
        self.mode_unproven
    }

    /// One dense or strided-batched GEMM in column-major order.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] for an operand layout the adapter does not take (a
    /// non-positive extent, a leading dimension below the operand's row count, or an extent,
    /// leading dimension or batch count above `i32::MAX`, which these 32-bit cuBLAS entry
    /// points cannot express), and the library's typed failure.
    pub fn gemm(&mut self, call: &GemmRequest<'_>) -> Result<(), GpuError> {
        self.ensure_mode_proven()?;
        let (m, n, k) = call.shape;
        let (lda, ldb, ldc) = call.leading;
        let (batches, stride_a, stride_b, stride_c) = call.batches;
        let a_rows = if call.trans_a { k } else { m };
        let b_rows_expected = if call.trans_b { n } else { k };
        if m <= 0 || n <= 0 || k <= 0 || batches <= 0 {
            return Err(GpuError::unsupported("gemm", "extents must be positive"));
        }
        if lda < a_rows || ldb < b_rows_expected || ldc < m {
            return Err(GpuError::unsupported(
                "gemm",
                "a leading dimension is below the operand's row count",
            ));
        }
        // Column-major GEMM: cuBLAS takes the transposed operand's row count as its
        // leading-dimension bound, which is what the checks above use.
        let trans_a = transpose(call.trans_a);
        let trans_b = transpose(call.trans_b);
        let handle = self.handle;
        let dtype = call.dtype;
        let (a, b, c) = (call.a, call.b, call.c);
        let (m, n, k) = (
            narrow_i32("gemm", "m", m)?,
            narrow_i32("gemm", "n", n)?,
            narrow_i32("gemm", "k", k)?,
        );
        let (lda, ldb, ldc) = (
            narrow_i32("gemm", "lda", lda)?,
            narrow_i32("gemm", "ldb", ldb)?,
            narrow_i32("gemm", "ldc", ldc)?,
        );
        let batches = narrow_i32("gemm", "batch count", batches)?;
        // The strides travel in the 64-bit strided-batched ABI (`c_longlong`), so they are
        // not narrowed here.
        // SAFETY: the raw scope registered all three operands in this runtime, so the
        // addresses belong to it and stay alive until the scope's completion; the
        // library call runs on the runtime's stream.
        unsafe {
            self.runtime
                .with_raw_locked(
                    vec![
                        RawResource::Buffer(a),
                        RawResource::Buffer(b),
                        RawResource::Buffer(c),
                    ],
                    |raw| {
                        let ptr =
                            |buffer: &DeviceBuffer| -> Result<*mut std::ffi::c_void, GpuError> {
                                raw.device_ptr(buffer)
                                    .map(|address| address as usize as *mut std::ffi::c_void)
                                    .map_err(|error| GpuError::Unavailable {
                                        detail: error.to_string(),
                                    })
                            };
                        let (a, b, c) = (ptr(a)?, ptr(b)?, ptr(c)?);
                        let status = if batches == 1 {
                            gemm_single(
                                handle, dtype, trans_a, trans_b, m, n, k, a, lda, b, ldb, c, ldc,
                            )
                        } else {
                            gemm_batched(
                                handle, dtype, trans_a, trans_b, m, n, k, a, lda, stride_a, b, ldb,
                                stride_b, c, ldc, stride_c, batches,
                            )
                        };
                        match status {
                            cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
                            other => Err(GpuError::Unsupported {
                                op: "gemm",
                                detail: format!("cuBLAS reported {other:?}"),
                            }),
                        }
                    },
                )
                .map_err(|error| GpuError::Unavailable {
                    detail: error.to_string(),
                })?
        }
    }
}

impl Cublas {
    /// One conjugate dot over two flat spans: `result <- sum(conj(x) * y)`.
    ///
    /// The complex widths conjugate their left operand in the library; the real widths are
    /// plain dots, which is the same sum because conjugation is the identity there. The
    /// result is one element written to `result` on this runtime's stream, so the caller's
    /// next read of it is stream-ordered and no host barrier is taken here.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] for a non-positive or `i32`-overflowing element count, a
    /// dtype without a dot entry, or a library status the call reports, and
    /// [`GpuError::Unavailable`] when the runtime cannot run the call.
    pub(crate) fn dot(
        &mut self,
        dtype: DType,
        len: i64,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        result: &DeviceBuffer,
    ) -> Result<(), GpuError> {
        self.ensure_mode_proven()?;
        if len <= 0 {
            return Err(GpuError::unsupported(
                "dot",
                "the element count must be positive",
            ));
        }
        let n = narrow_i32("dot", "element count", len)?;
        let Some(entry) = DotEntry::of(dtype) else {
            return Err(GpuError::unsupported(
                "dot",
                format!("{dtype:?} has no cuBLAS dot entry"),
            ));
        };
        let handle = self.handle;
        // A pointer-mode change that fails leaves the handle's mode unproven; the flag crosses the
        // closure and is recorded on the handle once the scope is over.
        let mode_unproven = std::cell::Cell::new(false);
        // SAFETY: the raw scope registered all three operands in this runtime, so the
        // addresses belong to it and stay alive until the scope's completion; the library
        // call runs on the runtime's stream.
        let outcome = unsafe {
            self.runtime
                .with_raw_locked(
                    vec![
                        RawResource::Buffer(x),
                        RawResource::Buffer(y),
                        RawResource::Buffer(result),
                    ],
                    |raw| {
                        let ptr =
                            |buffer: &DeviceBuffer| -> Result<*mut std::ffi::c_void, GpuError> {
                                raw.device_ptr(buffer)
                                    .map(|address| address as usize as *mut std::ffi::c_void)
                                    .map_err(|error| GpuError::Unavailable {
                                        detail: error.to_string(),
                                    })
                            };
                        let (x, y, result) = (ptr(x)?, ptr(y)?, ptr(result)?);
                        let status = cublas::cublasSetPointerMode_v2(
                            handle,
                            cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
                        );
                        if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            mode_unproven.set(true);
                            return Err(GpuError::Unsupported {
                                op: "dot",
                                detail: format!(
                                    "the handle refused device pointer mode: {status:?}"
                                ),
                            });
                        }
                        let status = entry.call(handle, n, x, y, result);
                        // The handle is shared with the GEMM entries, which read their
                        // coefficients on the host, so host pointer mode comes back on every
                        // path out of here, whatever the dot reported.
                        let restored = cublas::cublasSetPointerMode_v2(
                            handle,
                            cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                        );
                        if restored != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            mode_unproven.set(true);
                            return Err(GpuError::Unsupported {
                                op: "dot",
                                detail: format!(
                                    "the handle refused host pointer mode: {restored:?}"
                                ),
                            });
                        }
                        if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                            return Err(GpuError::Unsupported {
                                op: "dot",
                                detail: format!("cuBLAS reported {status:?}"),
                            });
                        }
                        Ok(())
                    },
                )
                .map_err(|error| GpuError::Unavailable {
                    detail: error.to_string(),
                })?
        };
        self.mode_unproven |= mode_unproven.get();
        outcome
    }
}

/// The cuBLAS dot entry of each width; the complex ones conjugate their left operand.
#[derive(Clone, Copy)]
enum DotEntry {
    /// `cublasSdot`: `sum(x * y)` in `f32`.
    S,
    /// `cublasDdot`: `sum(x * y)` in `f64`.
    D,
    /// `cublasCdotc`: `sum(conj(x) * y)` in `c32`.
    C,
    /// `cublasZdotc`: `sum(conj(x) * y)` in `c64`.
    Z,
}

impl DotEntry {
    /// `None` for a dtype cuBLAS has no dot for; the caller reports the refusal.
    const fn of(dtype: DType) -> Option<DotEntry> {
        match dtype {
            DType::F32 => Some(DotEntry::S),
            DType::F64 => Some(DotEntry::D),
            DType::C32 => Some(DotEntry::C),
            DType::C64 => Some(DotEntry::Z),
            DType::I32 | DType::I64 | DType::Bool => None,
        }
    }

    /// Run this entry over `n` unit-stride elements.
    ///
    /// # Safety
    ///
    /// `x` and `y` must address `n` device elements and `result` one result element of this
    /// entry's dtype, all registered in the raw scope that is running the call.
    unsafe fn call(
        self,
        handle: cublas::cublasHandle_t,
        n: i32,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        result: *mut std::ffi::c_void,
    ) -> cublas::cublasStatus_t {
        match self {
            DotEntry::S => {
                cublas::cublasSdot_v2(handle, n, x.cast(), 1, y.cast(), 1, result.cast())
            }
            DotEntry::D => {
                cublas::cublasDdot_v2(handle, n, x.cast(), 1, y.cast(), 1, result.cast())
            }
            DotEntry::C => {
                cublas::cublasCdotc_v2(handle, n, x.cast(), 1, y.cast(), 1, result.cast())
            }
            DotEntry::Z => {
                cublas::cublasZdotc_v2(handle, n, x.cast(), 1, y.cast(), 1, result.cast())
            }
        }
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        let handle = self.handle;
        // SAFETY: the raw scope activates the context the handle was created in; the
        // runtime synchronises first so no queued call still uses it.
        self.runtime.synchronize_best_effort();
        unsafe {
            let _ = self.runtime.with_raw_locked(Vec::new(), |_raw| {
                let _ = cublas::cublasDestroy_v2(handle);
            });
        }
    }
}

/// Narrow one cuBLAS size argument to its 32-bit entry-point type.
///
/// cuBLAS's extents, leading dimensions and batch count are `i32`. An `as` cast above
/// `i32::MAX` wraps silently and runs a GEMM of a different size, so the value is refused
/// with the field and its value instead of being narrowed.
///
/// # Errors
///
/// [`GpuError::Unsupported`] naming `field` and `value` when it does not fit.
pub(crate) fn narrow_i32(
    op: &'static str,
    field: &'static str,
    value: i64,
) -> Result<i32, GpuError> {
    i32::try_from(value).map_err(|_| {
        GpuError::unsupported(
            op,
            format!("{field} = {value} does not fit the i32 range of the cuBLAS entry points"),
        )
    })
}

fn transpose(transposed: bool) -> cublas::cublasOperation_t {
    if transposed {
        cublas::cublasOperation_t::CUBLAS_OP_T
    } else {
        cublas::cublasOperation_t::CUBLAS_OP_N
    }
}

/// A `double2`-shaped coefficient with CUDA's 16-byte alignment (never a
/// `num_complex` value, whose alignment is 8).
#[repr(C, align(16))]
pub(crate) struct ComplexCoefficient {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

#[allow(clippy::too_many_arguments)]
fn gemm_single(
    handle: cublas::cublasHandle_t,
    dtype: DType,
    trans_a: cublas::cublasOperation_t,
    trans_b: cublas::cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    a: *mut std::ffi::c_void,
    lda: i32,
    b: *mut std::ffi::c_void,
    ldb: i32,
    c: *mut std::ffi::c_void,
    ldc: i32,
) -> cublas::cublasStatus_t {
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let alpha32 = 1.0f32;
    let beta32 = 0.0f32;
    let complex = ComplexCoefficient { re: 1.0, im: 0.0 };
    let complex_zero = ComplexCoefficient { re: 0.0, im: 0.0 };
    // SAFETY: the pointers were validated by the caller's scope; the coefficients are
    // live stack values in host pointer mode.
    unsafe {
        match dtype {
            DType::F32 => cublas::cublasSgemm_v2(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                &alpha32,
                a.cast(),
                lda,
                b.cast(),
                ldb,
                &beta32,
                c.cast(),
                ldc,
            ),
            DType::F64 => cublas::cublasDgemm_v2(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                &alpha,
                a.cast(),
                lda,
                b.cast(),
                ldb,
                &beta,
                c.cast(),
                ldc,
            ),
            DType::C32 => cublas::cublasCgemm_v2(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&complex as *const ComplexCoefficient).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&complex_zero as *const ComplexCoefficient).cast(),
                c.cast(),
                ldc,
            ),
            DType::C64 => cublas::cublasZgemm_v2(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&complex as *const ComplexCoefficient).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&complex_zero as *const ComplexCoefficient).cast(),
                c.cast(),
                ldc,
            ),
            other => {
                let _ = other;
                cublas::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_batched(
    handle: cublas::cublasHandle_t,
    dtype: DType,
    trans_a: cublas::cublasOperation_t,
    trans_b: cublas::cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    a: *mut std::ffi::c_void,
    lda: i32,
    stride_a: i64,
    b: *mut std::ffi::c_void,
    ldb: i32,
    stride_b: i64,
    c: *mut std::ffi::c_void,
    ldc: i32,
    stride_c: i64,
    batches: i32,
) -> cublas::cublasStatus_t {
    let alpha = 1.0f64;
    let beta = 0.0f64;
    let alpha32 = 1.0f32;
    let beta32 = 0.0f32;
    let complex = ComplexCoefficient { re: 1.0, im: 0.0 };
    let complex_zero = ComplexCoefficient { re: 0.0, im: 0.0 };
    // SAFETY: as `gemm_single`, with the caller's batch strides.
    unsafe {
        match dtype {
            DType::F32 => cublas::cublasSgemmStridedBatched(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                &alpha32,
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                &beta32,
                c.cast(),
                ldc,
                stride_c,
                batches,
            ),
            DType::F64 => cublas::cublasDgemmStridedBatched(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                &alpha,
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                &beta,
                c.cast(),
                ldc,
                stride_c,
                batches,
            ),
            DType::C32 => cublas::cublasCgemmStridedBatched(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&complex as *const ComplexCoefficient).cast(),
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                (&complex_zero as *const ComplexCoefficient).cast(),
                c.cast(),
                ldc,
                stride_c,
                batches,
            ),
            DType::C64 => cublas::cublasZgemmStridedBatched(
                handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                (&complex as *const ComplexCoefficient).cast(),
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                (&complex_zero as *const ComplexCoefficient).cast(),
                c.cast(),
                ldc,
                stride_c,
                batches,
            ),
            other => {
                let _ = other;
                cublas::cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED
            }
        }
    }
}
