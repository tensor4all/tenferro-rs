//! cuBLAS-backed BLAS1 session operations for the CUDA backend.
//!
//! Implements the [`tenferro_tensor::backend::BackendSession`] BLAS1 hooks —
//! `vdot_read`, `norm_squared_read`, and `axpby_read_into_accum` — as single
//! cuBLAS calls on the runtime's CubeCL stream, so a TDVP/Krylov-style loop
//! does not have to compose them from full `dot_general` contractions.
//!
//! Execution contract:
//! - `vdot_read` enqueues `cublas{S,D}dot`/`cublas{C,Z}dotc` with the cuBLAS
//!   **device pointer mode**, writing `sum(conj(lhs) * rhs)` into a freshly
//!   allocated rank-0 device tensor. Success means the reduction was enqueued;
//!   no host barrier is taken.
//! - `norm_squared_read` reinterprets the (compact) input as a real component
//!   span and enqueues a self-`dot`, writing `sum(|x|^2)` into a rank-0 F32 or
//!   F64 device tensor, again in device pointer mode.
//! - `axpby_read_into_accum` enqueues one in-place vector
//!   `cublas{S,D,C,Z}geam` (`y <- alpha * x + beta * y`); the exact-dtype
//!   coefficients are read from host memory at enqueue time (host pointer
//!   mode), which does not block.
//!
//! Non-contiguous inputs are canonicalized on the device through the backend's
//! existing `to_contiguous_read` path before the cuBLAS call; tensors never
//! move between host and device here. Compact views with a nonzero offset are
//! consumed in place via pointer arithmetic. The per-(device, stream) cuBLAS
//! handle cache lives on [`CudaRuntime`].
//!
//! When the cuBLAS shared library cannot be loaded, these operations fail with
//! a typed load error; they do not fall back to native CubeCL kernels.

use std::ffi::c_void;

use cubecl::prelude::{CubeElement, CubePrimitive};
use cudarc::cublas::sys as cublas;
use num_complex::{Complex32, Complex64};

use tenferro_tensor::backend::{
    validate_axpby_read_into_accum, validate_norm_squared_read, validate_vdot_read,
};
use tenferro_tensor::{ContractionScalar, DType, TensorRead, TensorWrite};

use super::dispatch::{
    alloc_output, cubecl_buffer, cubecl_view_buffer, cubecl_view_mut_buffer,
    ensure_resident_on_runtime, ensure_view_mut_resident_on_runtime,
    ensure_view_resident_on_runtime, prepared_view_access, prepared_view_mut_access,
};
use super::error::unsupported_dtype;
use super::gemm::typed_device_ptr;
use super::interop::{alloc_zero_output, cuda_device_ptr_from_addr};
use super::runtime::check_cublas;
use super::{CudaBackend, CudaRuntime};
use crate::backend::TensorStructural;
use crate::{
    Error, Tensor, TensorScalar, TensorView, TensorViewMut, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

const VDOT_OP: &str = "BackendSession::vdot_read";
const NORM_OP: &str = "BackendSession::norm_squared_read";
const AXPBY_OP: &str = "BackendSession::axpby_read_into_accum";

/// Compute `sum(conj(lhs) * rhs)` into a rank-0 device tensor via cuBLAS.
///
/// # Errors
///
/// See [`tenferro_tensor::backend::BackendSession::vdot_read`]; additionally
/// returns [`Error::Io`] when the cuBLAS library cannot be loaded and
/// [`Error::RuntimeState`] when an input is not resident on this runtime.
pub(super) fn vdot_read(
    backend: &mut CudaBackend,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    validate_vdot_read(&lhs, &rhs)?;
    let lhs_materialized = if lhs.is_col_major_contiguous()? {
        None
    } else {
        Some(Box::new(backend.to_contiguous_read(lhs.clone())?))
    };
    let rhs_materialized = if rhs.is_col_major_contiguous()? {
        None
    } else {
        Some(Box::new(backend.to_contiguous_read(rhs.clone())?))
    };
    let lhs = lhs_materialized
        .as_deref()
        .map(TensorRead::from_tensor)
        .unwrap_or(lhs);
    let rhs = rhs_materialized
        .as_deref()
        .map(TensorRead::from_tensor)
        .unwrap_or(rhs);
    match lhs.dtype() {
        DType::F32 => vdot_typed::<f32>(backend, &lhs, &rhs),
        DType::F64 => vdot_typed::<f64>(backend, &lhs, &rhs),
        DType::C32 => vdot_typed::<Complex32>(backend, &lhs, &rhs),
        DType::C64 => vdot_typed::<Complex64>(backend, &lhs, &rhs),
        // INVARIANT: shared validation restricts the dtype to F32/F64/C32/C64.
        dtype => Err(unsupported_dtype(VDOT_OP, dtype)),
    }
}

/// Compute `sum(|x|^2)` into a rank-0 real device tensor via cuBLAS.
///
/// # Errors
///
/// See [`tenferro_tensor::backend::BackendSession::norm_squared_read`];
/// additionally returns [`Error::Io`] when the cuBLAS library cannot be
/// loaded and [`Error::RuntimeState`] when the input is not resident on this
/// runtime.
pub(super) fn norm_squared_read(
    backend: &mut CudaBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    validate_norm_squared_read(&input)?;
    let materialized = if input.is_col_major_contiguous()? {
        None
    } else {
        Some(Box::new(backend.to_contiguous_read(input.clone())?))
    };
    let input = materialized
        .as_deref()
        .map(TensorRead::from_tensor)
        .unwrap_or(input);
    match input.dtype() {
        DType::F32 => norm_squared_typed::<f32>(backend, &input),
        DType::F64 => norm_squared_typed::<f64>(backend, &input),
        DType::C32 => norm_squared_typed::<Complex32>(backend, &input),
        DType::C64 => norm_squared_typed::<Complex64>(backend, &input),
        // INVARIANT: shared validation restricts the dtype to F32/F64/C32/C64.
        dtype => Err(unsupported_dtype(NORM_OP, dtype)),
    }
}

/// Apply `y <- alpha * x + beta * y` in place via one cuBLAS vector `geam`.
///
/// # Errors
///
/// See [`tenferro_tensor::backend::BackendSession::axpby_read_into_accum`];
/// additionally returns [`Error::Io`] when the cuBLAS library cannot be
/// loaded and [`Error::RuntimeState`] when an operand is not resident on this
/// runtime.
pub(super) fn axpby_read_into_accum(
    backend: &mut CudaBackend,
    alpha: ContractionScalar,
    x: TensorRead<'_>,
    beta: ContractionScalar,
    mut y: TensorWrite<'_>,
) -> crate::Result<()> {
    validate_axpby_read_into_accum(alpha, &x, beta, &y)?;
    let materialized = if x.is_col_major_contiguous()? {
        None
    } else {
        Some(Box::new(backend.to_contiguous_read(x.clone())?))
    };
    let x = materialized
        .as_deref()
        .map(TensorRead::from_tensor)
        .unwrap_or(x);
    match x.dtype() {
        DType::F32 => axpby_typed::<f32>(backend, alpha, &x, beta, &mut y),
        DType::F64 => axpby_typed::<f64>(backend, alpha, &x, beta, &mut y),
        DType::C32 => axpby_typed::<Complex32>(backend, alpha, &x, beta, &mut y),
        DType::C64 => axpby_typed::<Complex64>(backend, alpha, &x, beta, &mut y),
        // INVARIANT: shared validation restricts the dtype to F32/F64/C32/C64.
        dtype => Err(unsupported_dtype(AXPBY_OP, dtype)),
    }
}

/// Read-slot operand: an owned compact tensor or a compact contiguous view.
enum ReadRef<'a, 'b, T> {
    Owned(&'a TypedTensor<T>),
    View(&'a TypedTensorView<'b, T>),
}

impl<T: TensorScalar + 'static> ReadRef<'_, '_, T> {
    fn n_elements(&self) -> usize {
        match self {
            Self::Owned(tensor) => tensor.n_elements(),
            Self::View(view) => view.n_elements(),
        }
    }

    fn ensure_resident(&self, rt: &CudaRuntime, op: &'static str) -> crate::Result<()> {
        match self {
            Self::Owned(tensor) => ensure_resident_on_runtime(rt, tensor, op),
            Self::View(view) => ensure_view_resident_on_runtime(rt, view, op),
        }
    }

    fn device_ptr(&self, rt: &CudaRuntime, op: &'static str) -> crate::Result<*mut c_void> {
        match self {
            Self::Owned(tensor) => typed_device_ptr(rt, tensor, op),
            Self::View(view) => {
                let prepared = prepared_view_access(view, op)?;
                offset_device_ptr::<T>(rt, prepared, view.offset(), op)
            }
        }
    }

    fn handle<'a>(&'a self, op: &'static str) -> crate::Result<&'a cubecl_runtime::server::Handle> {
        match self {
            Self::Owned(tensor) => Ok(cubecl_buffer(tensor, op)?.handle()),
            Self::View(view) => Ok(cubecl_view_buffer(view, op)?.handle()),
        }
    }
}

/// Write-slot operand: an owned compact tensor or a compact contiguous view.
enum WriteRef<'a, 'b, T> {
    Owned(&'a mut TypedTensor<T>),
    View(&'a mut TypedTensorViewMut<'b, T>),
}

impl<T: TensorScalar + 'static> WriteRef<'_, '_, T> {
    fn ensure_resident(&self, rt: &CudaRuntime, op: &'static str) -> crate::Result<()> {
        match self {
            Self::Owned(tensor) => ensure_resident_on_runtime(rt, tensor, op),
            Self::View(view) => ensure_view_mut_resident_on_runtime(rt, view, op),
        }
    }

    fn device_ptr(&mut self, rt: &CudaRuntime, op: &'static str) -> crate::Result<*mut c_void> {
        match self {
            Self::Owned(tensor) => typed_device_ptr(rt, tensor, op),
            Self::View(view) => {
                let offset = view.offset();
                let prepared = prepared_view_mut_access(view, op)?;
                offset_device_ptr::<T>(rt, prepared, offset, op)
            }
        }
    }

    fn handle<'a>(&'a self, op: &'static str) -> crate::Result<&'a cubecl_runtime::server::Handle> {
        match self {
            Self::Owned(tensor) => Ok(cubecl_buffer(tensor, op)?.handle()),
            Self::View(view) => Ok(cubecl_view_mut_buffer(view, op)?.handle()),
        }
    }
}

fn cross_stream_handles<'a>(
    rt: &CudaRuntime,
    handles: impl IntoIterator<Item = &'a cubecl_runtime::server::Handle>,
) -> Vec<cubecl_runtime::server::Handle> {
    handles
        .into_iter()
        .filter(|handle| !rt.is_current_stream_slot(handle))
        .cloned()
        .collect()
}

/// Resolve a compact view region to `base + offset * size_of::<T>()`.
fn offset_device_ptr<T: 'static>(
    rt: &CudaRuntime,
    prepared: super::dispatch::CubeclPreparedAccess,
    offset: isize,
    op: &'static str,
) -> crate::Result<*mut c_void> {
    let offset = usize::try_from(offset)
        .map_err(|_| Error::invalid_argument(op, "layout", "view offset must be nonnegative"))?;
    let resource = rt
        .client()
        .get_resource(prepared.into_handle())
        .map_err(|err| Error::backend_source(op, err))?;
    let offset_bytes = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::invalid_argument(op, "layout", "view byte offset overflows"))?;
    let addr = resource
        .resource()
        .ptr
        .checked_add(offset_bytes as u64)
        .ok_or_else(|| Error::invalid_argument(op, "layout", "view device address overflows"))?;
    cuda_device_ptr_from_addr(addr, op)
}

fn read_ref<'a, 'b, T: CublasScalar>(read: &'a TensorRead<'b>) -> Option<ReadRef<'a, 'b, T>> {
    match read {
        TensorRead::Tensor(tensor) => T::unwrap_tensor(tensor).map(ReadRef::Owned),
        TensorRead::View(view) => T::unwrap_view(view).map(ReadRef::View),
    }
}

fn write_ref<'a, 'b, T: CublasScalar>(
    write: &'a mut TensorWrite<'b>,
) -> Option<WriteRef<'a, 'b, T>> {
    match write {
        TensorWrite::Tensor(tensor) => T::unwrap_tensor_mut(tensor).map(WriteRef::Owned),
        TensorWrite::View(view) => T::unwrap_view_mut(view).map(WriteRef::View),
    }
}

fn validated_dtype_changed() -> Error {
    // INVARIANT: shared validation proves all operands share one supported
    // dtype before the typed dispatch; reaching this arm is an internal bug.
    Error::Internal("validated BLAS1 dtype changed before execution".into())
}

/// Convert an element count to the cuBLAS 64-bit length parameter.
pub(super) fn blas1_len(n: usize, op: &'static str) -> crate::Result<i64> {
    i64::try_from(n).map_err(|_| {
        Error::invalid_argument(op, "shape", format!("element count {n} exceeds i64::MAX"))
    })
}

fn vdot_typed<T: CublasScalar>(
    backend: &CudaBackend,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
) -> crate::Result<Tensor> {
    let rt = backend.runtime();
    let (Some(lhs), Some(rhs)) = (read_ref::<T>(lhs), read_ref::<T>(rhs)) else {
        return Err(validated_dtype_changed());
    };
    lhs.ensure_resident(rt, VDOT_OP)?;
    rhs.ensure_resident(rt, VDOT_OP)?;
    rt.set_current_cuda_context(VDOT_OP)?;
    let len = lhs.n_elements();
    if len == 0 {
        return Ok(T::wrap_tensor(alloc_zero_output::<T>(rt, &[])?));
    }
    let out = alloc_output::<T>(rt, &[])?;
    let out_ptr = typed_device_ptr(rt, &out, VDOT_OP)?;
    let x = lhs.device_ptr(rt, VDOT_OP)?;
    let y = rhs.device_ptr(rt, VDOT_OP)?;
    let n = blas1_len(len, VDOT_OP)?;
    let cross_stream_handles = cross_stream_handles(
        rt,
        [
            lhs.handle(VDOT_OP)?,
            rhs.handle(VDOT_OP)?,
            cubecl_buffer(&out, VDOT_OP)?.handle(),
        ],
    );
    rt.with_cublas_handle(
        VDOT_OP,
        cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
        cross_stream_handles,
        |handle| {
            // SAFETY: residency checks above tie `x`, `y`, and `out_ptr` to this
            // runtime's device; shared validation proves compact spans of `n`
            // elements and `out` is a fresh rank-0 allocation.
            check_cublas(VDOT_OP, T::DOTC_NAME, unsafe {
                T::dotc(handle, n, x, y, out_ptr)
            })
        },
    )?;
    Ok(T::wrap_tensor(out))
}

fn norm_squared_typed<T: CublasScalar>(
    backend: &CudaBackend,
    input: &TensorRead<'_>,
) -> crate::Result<Tensor> {
    let rt = backend.runtime();
    let Some(input) = read_ref::<T>(input) else {
        return Err(validated_dtype_changed());
    };
    input.ensure_resident(rt, NORM_OP)?;
    rt.set_current_cuda_context(NORM_OP)?;
    let len = input.n_elements();
    if len == 0 {
        return Ok(<T as CublasScalar>::Real::wrap_tensor(alloc_zero_output::<
            <T as CublasScalar>::Real,
        >(rt, &[])?));
    }
    let out = alloc_output::<<T as CublasScalar>::Real>(rt, &[])?;
    let out_ptr = typed_device_ptr(rt, &out, NORM_OP)?;
    let x = input.device_ptr(rt, NORM_OP)?;
    let real_len = len.checked_mul(T::REAL_COMPONENTS).ok_or_else(|| {
        Error::invalid_argument(NORM_OP, "shape", "real component count overflows")
    })?;
    let n = blas1_len(real_len, NORM_OP)?;
    let cross_stream_handles = cross_stream_handles(
        rt,
        [
            input.handle(NORM_OP)?,
            cubecl_buffer(&out, NORM_OP)?.handle(),
        ],
    );
    rt.with_cublas_handle(
        NORM_OP,
        cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
        cross_stream_handles,
        |handle| {
            // INVARIANT: `Complex32`/`Complex64` are `repr(C)` `[re, im]`
            // pairs, so `2 * len` real components self-dot to `sum(|z|^2)`.
            // SAFETY: residency checks tie `x`/`out_ptr` to this runtime; the
            // spans hold `n` real components and one output scalar.
            check_cublas(NORM_OP, <T as CublasScalar>::Real::DOT_NAME, unsafe {
                <T as CublasScalar>::Real::dot(handle, n, x, x, out_ptr)
            })
        },
    )?;
    Ok(<T as CublasScalar>::Real::wrap_tensor(out))
}

fn axpby_typed<T: CublasScalar>(
    backend: &CudaBackend,
    alpha: ContractionScalar,
    x: &TensorRead<'_>,
    beta: ContractionScalar,
    y: &mut TensorWrite<'_>,
) -> crate::Result<()> {
    let rt = backend.runtime();
    let Some(x) = read_ref::<T>(x) else {
        return Err(validated_dtype_changed());
    };
    let Some(mut y) = write_ref::<T>(y) else {
        return Err(validated_dtype_changed());
    };
    let (Some(alpha), Some(beta)) = (T::from_scalar(alpha), T::from_scalar(beta)) else {
        return Err(validated_dtype_changed());
    };
    x.ensure_resident(rt, AXPBY_OP)?;
    y.ensure_resident(rt, AXPBY_OP)?;
    rt.set_current_cuda_context(AXPBY_OP)?;
    let len = x.n_elements();
    if len == 0 {
        return Ok(());
    }
    let x_ptr = x.device_ptr(rt, AXPBY_OP)?;
    let y_ptr = y.device_ptr(rt, AXPBY_OP)?;
    let n = blas1_len(len, AXPBY_OP)?;
    let cross_stream_handles = cross_stream_handles(rt, [x.handle(AXPBY_OP)?, y.handle(AXPBY_OP)?]);
    rt.with_cublas_handle(
        AXPBY_OP,
        cublas::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
        cross_stream_handles,
        |handle| {
            // SAFETY: residency checks tie both pointers to this runtime;
            // shared validation proves compact same-shape, non-overlapping
            // spans matching in-place geam (`C == B`, `ldb == ldc`). Host-mode
            // coefficients are consumed synchronously during enqueue.
            check_cublas(AXPBY_OP, T::GEAM_NAME, unsafe {
                T::geam_accum(handle, n, &alpha, x_ptr, &beta, y_ptr)
            })
        },
    )?;
    Ok(())
}

/// Scalar-family dispatch for the cuBLAS BLAS1 bindings.
pub(super) trait CublasScalar:
    CubeElement + CubePrimitive + TensorScalar + Clone + Default + Send + Sync + 'static
{
    /// Real accumulator scalar: `Self` for real dtypes, the underlying real
    /// float for complex dtypes.
    type Real: CublasRealScalar;
    /// Real components per element (1 for real dtypes, 2 for complex).
    const REAL_COMPONENTS: usize;
    /// cuBLAS symbol name reported in provider errors for [`Self::dotc`].
    const DOTC_NAME: &'static str;
    /// cuBLAS symbol name reported in provider errors for [`Self::geam_accum`].
    const GEAM_NAME: &'static str;

    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>>;
    fn unwrap_view<'a, 'b>(view: &'a TensorView<'b>) -> Option<&'a TypedTensorView<'b, Self>>;
    fn unwrap_tensor_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>>;
    fn unwrap_view_mut<'a, 'b>(
        view: &'a mut TensorViewMut<'b>,
    ) -> Option<&'a mut TypedTensorViewMut<'b, Self>>;
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor;
    fn from_scalar(value: ContractionScalar) -> Option<Self>;

    /// Enqueue `result <- sum(conj(x) * y)` over `n` device elements.
    ///
    /// # Safety
    ///
    /// `x` and `y` must be live device pointers to at least `n` compact
    /// elements of `Self` on the handle's device, and `result` must be a live
    /// device pointer to one `Self` (device pointer mode).
    unsafe fn dotc(
        handle: cublas::cublasHandle_t,
        n: i64,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t;

    /// Enqueue the in-place vector update `y <- alpha * x + beta * y`.
    ///
    /// # Safety
    ///
    /// `x` and `y` must be live non-overlapping device pointers to at least
    /// `n` compact elements of `Self` on the handle's device; `alpha` and
    /// `beta` must be live host pointers (host pointer mode).
    unsafe fn geam_accum(
        handle: cublas::cublasHandle_t,
        n: i64,
        alpha: *const Self,
        x: *const c_void,
        beta: *const Self,
        y: *mut c_void,
    ) -> cublas::cublasStatus_t;
}

/// Real scalar family used for norm-squared accumulation.
pub(super) trait CublasRealScalar: CublasScalar {
    /// cuBLAS symbol name reported in provider errors for [`Self::dot`].
    const DOT_NAME: &'static str;

    /// Enqueue `result <- sum(x * y)` over `n` device elements.
    ///
    /// # Safety
    ///
    /// Same contract as [`CublasScalar::dotc`].
    unsafe fn dot(
        handle: cublas::cublasHandle_t,
        n: i64,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t;
}

macro_rules! impl_cublas_scalar {
    (
        $ty:ty, $variant:ident, $real:ty, $components:expr, $ffi:ty,
        $dotc:ident, $geam:ident
    ) => {
        impl CublasScalar for $ty {
            type Real = $real;
            const REAL_COMPONENTS: usize = $components;
            const DOTC_NAME: &'static str = stringify!($dotc);
            const GEAM_NAME: &'static str = stringify!($geam);

            fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
                match tensor {
                    Tensor::$variant(tensor) => Some(tensor),
                    _ => None,
                }
            }

            fn unwrap_view<'a, 'b>(
                view: &'a TensorView<'b>,
            ) -> Option<&'a TypedTensorView<'b, Self>> {
                match view {
                    TensorView::$variant(view) => Some(view),
                    _ => None,
                }
            }

            fn unwrap_tensor_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
                match tensor {
                    Tensor::$variant(tensor) => Some(tensor),
                    _ => None,
                }
            }

            fn unwrap_view_mut<'a, 'b>(
                view: &'a mut TensorViewMut<'b>,
            ) -> Option<&'a mut TypedTensorViewMut<'b, Self>> {
                match view {
                    TensorViewMut::$variant(view) => Some(view),
                    _ => None,
                }
            }

            fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor {
                Tensor::$variant(tensor)
            }

            fn from_scalar(value: ContractionScalar) -> Option<Self> {
                match value {
                    ContractionScalar::$variant(value) => Some(value),
                    _ => None,
                }
            }

            unsafe fn dotc(
                handle: cublas::cublasHandle_t,
                n: i64,
                x: *const c_void,
                y: *const c_void,
                result: *mut c_void,
            ) -> cublas::cublasStatus_t {
                // INVARIANT: `Complex32`/`Complex64` are `repr(C)` re/im pairs
                // with the same layout as `cuComplex`/`cuDoubleComplex`.
                cublas::$dotc(
                    handle,
                    n,
                    x.cast::<$ffi>(),
                    1,
                    y.cast::<$ffi>(),
                    1,
                    result.cast::<$ffi>(),
                )
            }

            unsafe fn geam_accum(
                handle: cublas::cublasHandle_t,
                n: i64,
                alpha: *const Self,
                x: *const c_void,
                beta: *const Self,
                y: *mut c_void,
            ) -> cublas::cublasStatus_t {
                let ld = n.max(1);
                // In-place `geam` form 2: `C = alpha * op(A) + beta * C` with
                // `B == C`, `ldb == ldc`, and `transb == N`, treating the
                // vectors as `n x 1` column-major matrices.
                cublas::$geam(
                    handle,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    n,
                    1,
                    alpha.cast::<$ffi>(),
                    x.cast::<$ffi>(),
                    ld,
                    beta.cast::<$ffi>(),
                    y.cast::<$ffi>().cast_const(),
                    ld,
                    y.cast::<$ffi>(),
                    ld,
                )
            }
        }
    };
}

impl_cublas_scalar!(f32, F32, f32, 1, f32, cublasSdot_v2_64, cublasSgeam_64);
impl_cublas_scalar!(f64, F64, f64, 1, f64, cublasDdot_v2_64, cublasDgeam_64);
impl_cublas_scalar!(
    Complex32,
    C32,
    f32,
    2,
    cublas::cuComplex,
    cublasCdotc_v2_64,
    cublasCgeam_64
);
impl_cublas_scalar!(
    Complex64,
    C64,
    f64,
    2,
    cublas::cuDoubleComplex,
    cublasZdotc_v2_64,
    cublasZgeam_64
);

impl CublasRealScalar for f32 {
    const DOT_NAME: &'static str = "cublasSdot_v2_64";

    unsafe fn dot(
        handle: cublas::cublasHandle_t,
        n: i64,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t {
        <f32 as CublasScalar>::dotc(handle, n, x, y, result)
    }
}

impl CublasRealScalar for f64 {
    const DOT_NAME: &'static str = "cublasDdot_v2_64";

    unsafe fn dot(
        handle: cublas::cublasHandle_t,
        n: i64,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t {
        <f64 as CublasScalar>::dotc(handle, n, x, y, result)
    }
}
