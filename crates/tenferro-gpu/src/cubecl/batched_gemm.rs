//! Standard batched matmul fast path through cuBLAS `gemmStridedBatchedEx`.
//!
//! `dot_general` covers far more than a batched matrix multiplication, so the
//! general contraction stays on cuTENSOR. A *standard* batched matmul — one
//! contraction axis per operand, both matrix axes innermost, and batch axes
//! trailing in the same ascending order on both sides — is exactly the shape
//! cuBLAS' strided-batched GEMM addresses without any copy, and a batched GEMM
//! of that shape is one kernel there instead of the several small launches and
//! plan-level synchronizations a cuTENSOR contraction of the same shape issues.
//!
//! Scope of this path:
//!
//! - only the allocating form (`alpha = 1`, `beta = 0`): the accumulate form
//!   keeps the cuTENSOR `C = D = out` contraction;
//! - only compact column-major owned operands, i.e. every batch slice is a
//!   contiguous `rows x cols` block and consecutive batches differ by exactly
//!   `rows * cols` elements;
//! - only uncontracted output axes, so the output's own column-major layout is
//!   the cuBLAS `C` layout (`ldc = m`, batch stride `m * n`);
//! - no conjugation: `gemmStridedBatchedEx` has no conjugation flag, so a
//!   conjugated complex operand falls back to cuTENSOR. Real conjugation is the
//!   identity and stays on this path;
//! - `batch_count > 1`; a single matrix keeps the existing cuTENSOR path.
//!
//! Everything else — permuted axes, several contraction axes, a batch axis
//! inside the matrix axes, a strided borrowed view — returns `None` from
//! [`BatchedGemmLaunch::detect`] and is served by [`super::gemm`] unchanged.
//!
//! Execution contract: one `cublas{S,D,C,Z}gemmStridedBatchedEx` enqueued on the
//! session stream from the runtime's cached per-stream cuBLAS handle, in host
//! pointer mode. Success means the GEMM was enqueued; no host barrier is taken,
//! and stream ordering and the cross-stream retirement hand-off are the same as
//! the other vendor calls (`CudaRuntime::with_cublas_handle`).

use std::ffi::c_void;

use cudarc::cublas::sys as cublas;
use num_complex::{Complex32, Complex64};

use super::dispatch::cubecl_buffer;
use super::gemm::{cross_stream_handles, typed_device_ptr};
use super::runtime::check_cublas;
use super::CudaRuntime;
use crate::config::DotGeneralConfig;
use crate::{TensorScalar, TypedTensor};
use cubecl::frontend::Scalar;

const OP: &str = "dot_general";

#[cfg(test)]
thread_local! {
    /// Counts launched cuBLAS strided-batched GEMMs so a test can prove the
    /// fast path was actually taken instead of only that the result matched.
    /// Thread-local: the CUDA session runs its callback on the calling thread,
    /// so a test observes only its own launches while tests run in parallel.
    static BATCHED_GEMM_LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_batched_gemm_launches_for_test() {
    BATCHED_GEMM_LAUNCHES.with(|launches| launches.set(0));
}

#[cfg(test)]
pub(crate) fn batched_gemm_launches_for_test() -> usize {
    BATCHED_GEMM_LAUNCHES.with(std::cell::Cell::get)
}

/// The validated `cublasGemmStridedBatchedEx` parameters of one standard
/// batched matmul.
///
/// Every value is derived from the operand shapes, so this is the complete
/// description of the launch: which stored axis is the matrix row axis
/// (`trans_*`), the `m x n x k` factorization, the three leading dimensions,
/// and the three batch strides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchedGemmLaunch {
    trans_a: cublas::cublasOperation_t,
    trans_b: cublas::cublasOperation_t,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
    batch_count: i32,
}

impl BatchedGemmLaunch {
    /// Detect the cuBLAS-addressable standard batched matmul, or `None`.
    ///
    /// `lhs_shape`, `rhs_shape`, and `config` are already validated; this only
    /// decides whether the form is representable, so a rejected case simply
    /// keeps the general cuTENSOR path. A size that does not fit the portable
    /// `c_int`/`c_longlong` cuBLAS parameters is also rejected here, so the fast
    /// path never has to report a size error of its own.
    pub(super) fn detect(
        lhs_shape: &[usize],
        rhs_shape: &[usize],
        config: &DotGeneralConfig,
    ) -> Option<Self> {
        let lhs_contracting: &[usize] = &config.lhs_contracting_dims;
        let rhs_contracting: &[usize] = &config.rhs_contracting_dims;
        let lhs_batch: &[usize] = &config.lhs_batch_dims;
        let rhs_batch: &[usize] = &config.rhs_batch_dims;
        let [lhs_axis] = lhs_contracting else {
            return None;
        };
        let [rhs_axis] = rhs_contracting else {
            return None;
        };
        let rank = lhs_shape.len();
        if rank < 2 || rhs_shape.len() != rank {
            return None;
        }
        // The batch axes must be the trailing axes of both operands, in the
        // same ascending order. Then each batch slice of a compact
        // column-major operand is one contiguous matrix block and consecutive
        // flattened batch indices differ by exactly `rows * cols` elements —
        // the only batch layout a single `stride_a`/`stride_b`/`stride_c` can
        // address. A batch axis among the matrix axes interleaves the batch
        // index into the matrix block and is rejected here.
        if lhs_batch != rhs_batch || !lhs_batch.iter().copied().eq(2..rank) {
            return None;
        }
        // The remaining two axes of each operand are its matrix pair
        // (free/contracting), so a contraction axis at 0 or 1 is the only
        // possibility left; the free axis is the other one.
        if *lhs_axis >= 2 || *rhs_axis >= 2 {
            return None;
        }
        let lhs_transposed = *lhs_axis == 0;
        let rhs_transposed = *rhs_axis == 1;
        let (m, k) = (lhs_shape[1 - *lhs_axis], lhs_shape[*lhs_axis]);
        let (rhs_k, n) = (rhs_shape[*rhs_axis], rhs_shape[1 - *rhs_axis]);
        if m == 0 || n == 0 || k == 0 || k != rhs_k {
            return None;
        }
        let mut batch_count = 1usize;
        for &axis in lhs_batch {
            if lhs_shape[axis] != rhs_shape[axis] {
                return None;
            }
            batch_count = batch_count.checked_mul(lhs_shape[axis])?;
        }
        // One matrix stays on cuTENSOR: the win this path targets is the batched
        // kernel count, and a plain GEMM is already a single plan.
        if batch_count < 2 {
            return None;
        }
        // Column-major: the leading dimension of a stored matrix is its own row
        // count, so with `opA = N` it is `m` and with `opA = T` the stored
        // matrix is `k x m` and it is `k` — in both cases axis 0's extent.
        Some(Self {
            trans_a: cublas_operation(lhs_transposed),
            trans_b: cublas_operation(rhs_transposed),
            m: i32::try_from(m).ok()?,
            n: i32::try_from(n).ok()?,
            k: i32::try_from(k).ok()?,
            lda: i32::try_from(lhs_shape[0]).ok()?,
            ldb: i32::try_from(rhs_shape[0]).ok()?,
            ldc: i32::try_from(m).ok()?,
            stride_a: i64::try_from(lhs_shape[0].checked_mul(lhs_shape[1])?).ok()?,
            stride_b: i64::try_from(rhs_shape[0].checked_mul(rhs_shape[1])?).ok()?,
            stride_c: i64::try_from(m.checked_mul(n)?).ok()?,
            batch_count: i32::try_from(batch_count).ok()?,
        })
    }
}

fn cublas_operation(transposed: bool) -> cublas::cublasOperation_t {
    if transposed {
        cublas::cublasOperation_t::CUBLAS_OP_T
    } else {
        cublas::cublasOperation_t::CUBLAS_OP_N
    }
}

/// Scalar family of the cuBLAS strided-batched GEMM binding.
pub(super) trait BatchedGemmScalar: TensorScalar + Scalar + Copy + 'static {
    /// cuBLAS symbol reported in provider errors.
    const GEMM_NAME: &'static str;
    /// Data type of `A`, `B`, and `C`.
    const DATA_TYPE: cublas::cudaDataType;
    /// Arithmetic precision cuBLAS accumulates the product in.
    const COMPUTE_TYPE: cublas::cublasComputeType_t;

    /// Enqueue `C = op(A) * op(B)` for every batch slice, with `alpha = 1` and
    /// `beta = 0`.
    ///
    /// # Safety
    ///
    /// `a`, `b`, and `c` must be live device pointers to `batch_count` strided
    /// matrix slices matching `args` on the handle's device, and `c` must be
    /// distinct from `a` and `b`.
    unsafe fn gemm_strided_batched_ex(
        handle: cublas::cublasHandle_t,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> cublas::cublasStatus_t;
}

/// One `cublasGemmStridedBatchedEx` call; `alpha`/`beta` point at locals of the
/// FFI scalar type of `data_type` whose values are the overwrite coefficients.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_strided_batched_ex(
    handle: cublas::cublasHandle_t,
    args: &BatchedGemmLaunch,
    alpha: *const c_void,
    a: *const c_void,
    b: *const c_void,
    beta: *const c_void,
    c: *mut c_void,
    data_type: cublas::cudaDataType,
    compute_type: cublas::cublasComputeType_t,
) -> cublas::cublasStatus_t {
    cublas::cublasGemmStridedBatchedEx(
        handle,
        args.trans_a,
        args.trans_b,
        args.m,
        args.n,
        args.k,
        alpha,
        a,
        data_type,
        args.lda,
        args.stride_a,
        b,
        data_type,
        args.ldb,
        args.stride_b,
        beta,
        c,
        data_type,
        args.ldc,
        args.stride_c,
        args.batch_count,
        compute_type,
        cublas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
    )
}

impl BatchedGemmScalar for f32 {
    const GEMM_NAME: &'static str = "cublasSgemmStridedBatchedEx";
    const DATA_TYPE: cublas::cudaDataType = cublas::cudaDataType::CUDA_R_32F;
    const COMPUTE_TYPE: cublas::cublasComputeType_t =
        cublas::cublasComputeType_t::CUBLAS_COMPUTE_32F;

    unsafe fn gemm_strided_batched_ex(
        handle: cublas::cublasHandle_t,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> cublas::cublasStatus_t {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        launch_gemm_strided_batched_ex(
            handle,
            args,
            (&alpha as *const f32).cast(),
            a,
            b,
            (&beta as *const f32).cast(),
            c,
            Self::DATA_TYPE,
            Self::COMPUTE_TYPE,
        )
    }
}

impl BatchedGemmScalar for f64 {
    const GEMM_NAME: &'static str = "cublasDgemmStridedBatchedEx";
    const DATA_TYPE: cublas::cudaDataType = cublas::cudaDataType::CUDA_R_64F;
    const COMPUTE_TYPE: cublas::cublasComputeType_t =
        cublas::cublasComputeType_t::CUBLAS_COMPUTE_64F;

    unsafe fn gemm_strided_batched_ex(
        handle: cublas::cublasHandle_t,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> cublas::cublasStatus_t {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        launch_gemm_strided_batched_ex(
            handle,
            args,
            (&alpha as *const f64).cast(),
            a,
            b,
            (&beta as *const f64).cast(),
            c,
            Self::DATA_TYPE,
            Self::COMPUTE_TYPE,
        )
    }
}

impl BatchedGemmScalar for Complex32 {
    const GEMM_NAME: &'static str = "cublasCgemmStridedBatchedEx";
    const DATA_TYPE: cublas::cudaDataType = cublas::cudaDataType::CUDA_C_32F;
    const COMPUTE_TYPE: cublas::cublasComputeType_t =
        cublas::cublasComputeType_t::CUBLAS_COMPUTE_32F;

    unsafe fn gemm_strided_batched_ex(
        handle: cublas::cublasHandle_t,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> cublas::cublasStatus_t {
        // INVARIANT: `cudarc`'s `cuComplex` is `#[repr(C)] { x: f32, y: f32 }`,
        // the same re/im layout as `Complex32`, but it is the FFI type, so the
        // host coefficients are built as it instead of transmuted.
        let alpha = cublas::cuComplex { x: 1.0, y: 0.0 };
        let beta = cublas::cuComplex { x: 0.0, y: 0.0 };
        launch_gemm_strided_batched_ex(
            handle,
            args,
            (&alpha as *const cublas::cuComplex).cast(),
            a,
            b,
            (&beta as *const cublas::cuComplex).cast(),
            c,
            Self::DATA_TYPE,
            Self::COMPUTE_TYPE,
        )
    }
}

impl BatchedGemmScalar for Complex64 {
    const GEMM_NAME: &'static str = "cublasZgemmStridedBatchedEx";
    const DATA_TYPE: cublas::cudaDataType = cublas::cudaDataType::CUDA_C_64F;
    const COMPUTE_TYPE: cublas::cublasComputeType_t =
        cublas::cublasComputeType_t::CUBLAS_COMPUTE_64F;

    unsafe fn gemm_strided_batched_ex(
        handle: cublas::cublasHandle_t,
        args: &BatchedGemmLaunch,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    ) -> cublas::cublasStatus_t {
        // INVARIANT: `cudarc`'s `cuDoubleComplex` is `#[repr(C)] { x: f64,
        // y: f64 }`, the same re/im layout as `Complex64`, but it is the FFI
        // type, so the host coefficients are built as it instead of transmuted.
        let alpha = cublas::cuDoubleComplex { x: 1.0, y: 0.0 };
        let beta = cublas::cuDoubleComplex { x: 0.0, y: 0.0 };
        launch_gemm_strided_batched_ex(
            handle,
            args,
            (&alpha as *const cublas::cuDoubleComplex).cast(),
            a,
            b,
            (&beta as *const cublas::cuDoubleComplex).cast(),
            c,
            Self::DATA_TYPE,
            Self::COMPUTE_TYPE,
        )
    }
}

/// Run one standard batched matmul as a single cuBLAS strided-batched GEMM.
///
/// The output is a fresh compact allocation, so it is resolved the same way the
/// cuTENSOR path resolves it: `lhs`/`rhs`/`output` read their device addresses
/// through the runtime's memoized-address fast path, and cross-stream operands
/// take the same completion hand-off as every other vendor enqueue. No extra
/// stream synchronization is introduced.
pub(super) fn run_allocating<T: BatchedGemmScalar>(
    rt: &CudaRuntime,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    output: &TypedTensor<T>,
    args: &BatchedGemmLaunch,
) -> crate::Result<()> {
    let lhs_ptr = typed_device_ptr(rt, lhs, OP)?;
    let rhs_ptr = typed_device_ptr(rt, rhs, OP)?;
    let output_ptr = typed_device_ptr(rt, output, OP)?;
    let cross_stream_handles = cross_stream_handles(
        rt,
        [
            cubecl_buffer(lhs, OP)?.handle(),
            cubecl_buffer(rhs, OP)?.handle(),
            cubecl_buffer(output, OP)?.handle(),
        ],
    );
    rt.with_cublas_handle(
        OP,
        // Host pointer mode reads the coefficients from the stack locals during
        // the call, which neither blocks nor needs a per-call device scalar
        // upload and its ordering. This is the same choice the cuBLAS `geam`
        // AXPBY path makes.
        cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
        cross_stream_handles,
        |handle| {
            // SAFETY: residency checks inside `typed_device_ptr` tie all three
            // pointers to this runtime's device; `detect` proved the shapes are
            // the strided-batched layout these pointers describe, and the
            // coefficients are stack locals live for the whole call.
            check_cublas(OP, T::GEMM_NAME, unsafe {
                T::gemm_strided_batched_ex(handle, args, lhs_ptr, rhs_ptr, output_ptr)
            })
        },
    )?;
    #[cfg(test)]
    BATCHED_GEMM_LAUNCHES.with(|launches| launches.set(launches.get() + 1));
    Ok(())
}

#[cfg(test)]
mod tests;
