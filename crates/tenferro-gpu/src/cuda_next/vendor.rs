//! Vendor operations on the runtime's own stream: cuBLAS GEMM and strided-batched GEMM.
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
        })
    }

    /// One dense or strided-batched GEMM in column-major order.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] for an operand layout the adapter does not take (a
    /// non-positive extent or a leading dimension below the operand's row count), and the
    /// library's typed failure.
    pub fn gemm(&mut self, call: &GemmRequest<'_>) -> Result<(), GpuError> {
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
        let (m, n, k) = (m as i32, n as i32, k as i32);
        let (lda, ldb, ldc) = (lda as i32, ldb as i32, ldc as i32);
        let (batches, stride_a, stride_b, stride_c) =
            (batches as i32, stride_a, stride_b, stride_c);
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
struct Complex128Coefficient {
    re: f64,
    im: f64,
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
    let complex = Complex128Coefficient { re: 1.0, im: 0.0 };
    let complex_zero = Complex128Coefficient { re: 0.0, im: 0.0 };
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
                (&complex as *const Complex128Coefficient).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&complex_zero as *const Complex128Coefficient).cast(),
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
                (&complex as *const Complex128Coefficient).cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                (&complex_zero as *const Complex128Coefficient).cast(),
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
    let complex = Complex128Coefficient { re: 1.0, im: 0.0 };
    let complex_zero = Complex128Coefficient { re: 0.0, im: 0.0 };
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
                (&complex as *const Complex128Coefficient).cast(),
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                (&complex_zero as *const Complex128Coefficient).cast(),
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
                (&complex as *const Complex128Coefficient).cast(),
                a.cast(),
                lda,
                stride_a,
                b.cast(),
                ldb,
                stride_b,
                (&complex_zero as *const Complex128Coefficient).cast(),
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
