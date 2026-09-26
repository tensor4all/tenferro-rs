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
//!   mode), which does not block. An arbitrary-stride `x` is a layout no
//!   BLAS-1 vendor entry can address, so it runs one native strided-source
//!   kernel instead, reading `x` in place rather than canonicalizing it.
//!
//! `vdot_read` and `norm_squared_read` still canonicalize a non-contiguous
//! input on the device through the backend's existing `to_contiguous_read`
//! path before the cuBLAS call; tensors never move between host and device
//! here. Compact views with a nonzero offset are consumed in place via pointer
//! arithmetic. The per-(device, stream) cuBLAS handle cache lives on
//! [`CudaRuntime`].
//!
//! When the cuBLAS shared library cannot be loaded, these operations fail with
//! a typed load error; they do not fall back to native CubeCL kernels.

use std::ffi::c_void;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl::client::ComputeClient;
use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, CubePrimitive};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use cudarc::cublas::sys as cublas;
use num_complex::{Complex32, Complex64};

use tenferro_tensor::backend::{
    validate_axpby_read_into_accum, validate_norm_squared_read, validate_vdot_read,
};
use tenferro_tensor::{ContractionScalar, DType, TensorRead, TensorWrite};

use super::dispatch::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d, cubecl_buffer,
    cubecl_view_buffer, cubecl_view_mut_buffer, ensure_resident_on_runtime,
    ensure_view_mut_resident_on_runtime, ensure_view_resident_on_runtime, prepared_view_access,
    prepared_view_mut_access, typed_tensor_array_arg, typed_tensor_mut_array_arg,
    typed_view_array_arg, typed_view_mut_array_arg,
};
use super::error::unsupported_dtype;
use super::gemm::typed_device_ptr;
use super::interop::{alloc_zero_output, offset_device_ptr, upload_typed_tensor};
use super::runtime::check_cublas;
use super::{CudaBackend, CudaRuntime};
use crate::{
    Error, Tensor, TensorScalar, TensorView, TensorViewMut, TypedTensor, TypedTensorView,
    TypedTensorViewMut,
};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
/// Counts native strided-source AXPBY passes so a regression that reintroduces
/// a hidden `x` canonicalization (which would route back through cuBLAS) fails
/// the data-movement assertion instead of only the numerics.
#[cfg(test)]
static STRIDED_SOURCE_PASSES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_strided_source_passes_for_test() {
    STRIDED_SOURCE_PASSES.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn strided_source_passes_for_test() -> usize {
    STRIDED_SOURCE_PASSES.load(Ordering::SeqCst)
}

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
        Some(Box::new(super::ops::to_contiguous_read(
            backend,
            lhs.clone(),
        )?))
    };
    let rhs_materialized = if rhs.is_col_major_contiguous()? {
        None
    } else {
        Some(Box::new(super::ops::to_contiguous_read(
            backend,
            rhs.clone(),
        )?))
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
        Some(Box::new(super::ops::to_contiguous_read(
            backend,
            input.clone(),
        )?))
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
    match x.dtype() {
        DType::F32 => axpby_typed::<f32>(backend, alpha, &x, beta, &mut y),
        DType::F64 => axpby_typed::<f64>(backend, alpha, &x, beta, &mut y),
        DType::C32 => axpby_typed::<Complex32>(backend, alpha, &x, beta, &mut y),
        DType::C64 => axpby_typed::<Complex64>(backend, alpha, &x, beta, &mut y),
        // INVARIANT: shared validation restricts the dtype to F32/F64/C32/C64.
        dtype => Err(unsupported_dtype(AXPBY_OP, dtype)),
    }
}

/// Read-slot operand: an owned compact tensor or a borrowed view.
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

    /// Whether this operand is a compact column-major span, which is the
    /// layout the cuBLAS vector entries can address directly.
    fn is_compact(&self) -> crate::Result<bool> {
        match self {
            Self::Owned(_) => Ok(true),
            Self::View(view) => view.is_col_major_contiguous(),
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

    fn offset(&self) -> isize {
        match self {
            Self::Owned(_) => 0,
            Self::View(view) => view.offset(),
        }
    }
}

impl<T: CubeElement + TensorScalar + Clone + 'static> WriteRef<'_, '_, T> {
    /// Bind the destination's whole root allocation for a native kernel launch.
    fn array_arg(&mut self, op: &'static str) -> crate::Result<ArrayArg<CubeclCudaRuntime>> {
        match self {
            Self::Owned(tensor) => typed_tensor_mut_array_arg(tensor, op),
            Self::View(view) => typed_view_mut_array_arg(view, op),
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

/// Convert an element count to the portable cuBLAS length parameter.
pub(super) fn blas1_len(n: usize, op: &'static str) -> crate::Result<i32> {
    i32::try_from(n).map_err(|_| {
        Error::invalid_argument(op, "shape", format!("element count {n} exceeds i32::MAX"))
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
    if !x.is_compact()? {
        let ReadRef::View(x_view) = x else {
            // INVARIANT: an owned runtime tensor is always compact
            // column-major, so only a view can report a strided layout.
            return Err(Error::Internal(
                "compact owned AXPBY operand reported a strided layout".into(),
            ));
        };
        return axpby_strided_source_typed::<T>(rt, alpha, x_view, beta, &mut y);
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
                T::geam_accum(handle, n, alpha, x_ptr, beta, y_ptr)
            })
        },
    )?;
    Ok(())
}

/// Apply `y <- alpha * x + beta * y` for an arbitrary-stride or offset `x`.
///
/// cuBLAS vector entries can only address a compact span, and the shared
/// contract keeps `y` compact, so this is the layout the vendor path does not
/// cover. One native kernel reads `x` through its own strides and offset, which
/// keeps the operation a single pass with no intermediate allocation, instead
/// of canonicalizing `x` into scratch first.
fn axpby_strided_source_typed<T: CublasScalar>(
    rt: &CudaRuntime,
    alpha: T,
    x: &TypedTensorView<'_, T>,
    beta: T,
    y: &mut WriteRef<'_, '_, T>,
) -> crate::Result<()> {
    #[cfg(test)]
    STRIDED_SOURCE_PASSES.fetch_add(1, Ordering::SeqCst);
    let plan = NativeStridedSourcePlan::new(AXPBY_OP, x.shape(), x.strides(), x.offset())?;
    let y_offset = i64::try_from(y.offset()).map_err(|_| {
        Error::invalid_argument(AXPBY_OP, "layout", "destination offset overflows i64")
    })?;
    // The coefficients become an explicit two-element device constant, the same
    // boundary the in-place scaling kernels use. Operand tensors never move
    // between host and device here.
    let coefficients = upload_typed_tensor(rt, vec![2], vec![alpha, beta])?;
    let coefficients_arg = typed_tensor_array_arg(&coefficients, AXPBY_OP)?;
    let x_arg = typed_view_array_arg(x, AXPBY_OP)?;
    let y_arg = y.array_arg(AXPBY_OP)?;
    unsafe {
        // SAFETY: Both operand bindings cover their whole root allocation. The
        // plan proved every logical coordinate of `x` stays inside its
        // allocation span, shared validation proved `y` is a compact injective
        // span of the same shape that does not overlap `x`, the coefficient
        // array holds exactly the two elements the kernel reads, and the launch
        // domain visits each logical coordinate exactly once.
        T::launch_axpby_strided_source(
            rt.client(),
            cube_count_for_len(plan.len)?,
            cube_dim_1d(),
            y_arg,
            x_arg,
            coefficients_arg,
            &plan.dims,
            &plan.strides,
            plan.offset,
            y_offset,
            plan.len,
        );
    }
    Ok(())
}

/// Validated launch metadata for a strided-source BLAS-1 pass.
struct NativeStridedSourcePlan {
    dims: Vec<usize>,
    strides: Vec<i64>,
    offset: i64,
    len: usize,
}

impl NativeStridedSourcePlan {
    fn new(
        op: &'static str,
        shape: &[usize],
        strides: &[isize],
        offset: isize,
    ) -> crate::Result<Self> {
        let len = tenferro_tensor::validate::checked_shape_product(op, "shape", shape)?;
        let offset = i64::try_from(offset)
            .map_err(|_| Error::invalid_argument(op, "layout", "source offset overflows i64"))?;
        let strides = strides
            .iter()
            .map(|&stride| {
                i64::try_from(stride).map_err(|_| {
                    Error::invalid_argument(op, "layout", "source stride overflows i64")
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            dims: shape.to_vec(),
            strides,
            offset,
            len,
        })
    }
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
        n: i32,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t;

    /// Launch the native strided-source `y <- alpha * x + beta * y` kernel.
    ///
    /// `dims`, `x_strides`, and `x_offset` describe the source region inside
    /// its allocation; `y_offset` is the destination's compact start, and
    /// `coefficients` is a two-element device array holding `[alpha, beta]`.
    ///
    /// # Safety
    ///
    /// Both array arguments must bind the whole root allocation of a live
    /// operand on this client's device, the source region and the destination
    /// span must stay inside their allocations, the destination span must be
    /// injective and disjoint from the source, and `count`/`dim` must cover
    /// exactly `len` logical elements.
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_axpby_strided_source(
        client: &ComputeClient<CubeclCudaRuntime>,
        count: CubeCount,
        dim: CubeDim,
        y: ArrayArg<CubeclCudaRuntime>,
        x: ArrayArg<CubeclCudaRuntime>,
        coefficients: ArrayArg<CubeclCudaRuntime>,
        dims: &[usize],
        x_strides: &[i64],
        x_offset: i64,
        y_offset: i64,
        len: usize,
    );

    /// Enqueue the in-place vector update `y <- alpha * x + beta * y`.
    ///
    /// The coefficients are taken by value: the implementation copies them
    /// into locals of the cuBLAS FFI scalar type before passing host pointers
    /// (host pointer mode), because `cuDoubleComplex` is 16-byte aligned while
    /// `Complex64` is only 8-byte aligned, so a `*const Self` cast is not a
    /// valid `cuDoubleComplex` pointer.
    ///
    /// # Safety
    ///
    /// `x` and `y` must be live non-overlapping device pointers to at least
    /// `n` compact elements of `Self` on the handle's device.
    unsafe fn geam_accum(
        handle: cublas::cublasHandle_t,
        n: i32,
        alpha: Self,
        x: *const c_void,
        beta: Self,
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
        n: i32,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t;
}

macro_rules! impl_cublas_scalar {
    (
        $ty:ty, $variant:ident, $real:ty, $components:expr, $ffi:ty,
        $dotc:ident, $geam:ident, $axpby_kernel:path
    ) => {
        impl CublasScalar for $ty {
            type Real = $real;
            const REAL_COMPONENTS: usize = $components;
            const DOTC_NAME: &'static str = stringify!($dotc);
            const GEAM_NAME: &'static str = stringify!($geam);

            fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
                tensor.as_typed::<Self>()
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
                tensor.as_typed_mut::<Self>()
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
                Tensor::from_typed::<preset_scalar!($variant)>(tensor)
            }

            fn from_scalar(value: ContractionScalar) -> Option<Self> {
                match value {
                    ContractionScalar::$variant(value) => Some(value),
                    _ => None,
                }
            }

            unsafe fn launch_axpby_strided_source(
                client: &ComputeClient<CubeclCudaRuntime>,
                count: CubeCount,
                dim: CubeDim,
                y: ArrayArg<CubeclCudaRuntime>,
                x: ArrayArg<CubeclCudaRuntime>,
                coefficients: ArrayArg<CubeclCudaRuntime>,
                dims: &[usize],
                x_strides: &[i64],
                x_offset: i64,
                y_offset: i64,
                len: usize,
            ) {
                let rank = dims.len();
                $axpby_kernel(
                    client,
                    count,
                    dim,
                    y,
                    x,
                    coefficients,
                    comptime_sequence(dims),
                    comptime_sequence(x_strides),
                    x_offset,
                    y_offset,
                    len,
                    rank,
                );
            }

            unsafe fn dotc(
                handle: cublas::cublasHandle_t,
                n: i32,
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
                n: i32,
                alpha: Self,
                x: *const c_void,
                beta: Self,
                y: *mut c_void,
            ) -> cublas::cublasStatus_t {
                let ld = n.max(1);
                // Host-mode coefficients must satisfy the FFI scalar's
                // alignment (`cuDoubleComplex` is 16-byte aligned, `Complex64`
                // only 8). Copy each value into a local of the FFI type so the
                // pointer handed to cuBLAS is aligned; `transmute_copy` reads
                // the source unaligned when the destination is stricter.
                // INVARIANT: `Self` and `$ffi` have identical size and layout
                // (re/im pairs for complex, the same primitive for real).
                const _: () = assert!(core::mem::size_of::<$ty>() == core::mem::size_of::<$ffi>());
                let alpha: $ffi = core::mem::transmute_copy(&alpha);
                let beta: $ffi = core::mem::transmute_copy(&beta);
                // In-place `geam` form 2: `C = alpha * op(A) + beta * C` with
                // `B == C`, `ldb == ldc`, and `transb == N`, treating the
                // vectors as `n x 1` column-major matrices.
                cublas::$geam(
                    handle,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    cublas::cublasOperation_t::CUBLAS_OP_N,
                    n,
                    1,
                    &alpha,
                    x.cast::<$ffi>(),
                    ld,
                    &beta,
                    y.cast::<$ffi>().cast_const(),
                    ld,
                    y.cast::<$ffi>(),
                    ld,
                )
            }
        }
    };
}

impl_cublas_scalar!(
    f32,
    F32,
    f32,
    1,
    f32,
    cublasSdot_v2,
    cublasSgeam,
    crate::kernels::elementwise::axpby_strided_source_float::launch_unchecked::<
        f32,
        CubeclCudaRuntime,
    >
);
impl_cublas_scalar!(
    f64,
    F64,
    f64,
    1,
    f64,
    cublasDdot_v2,
    cublasDgeam,
    crate::kernels::elementwise::axpby_strided_source_float::launch_unchecked::<
        f64,
        CubeclCudaRuntime,
    >
);
impl_cublas_scalar!(
    Complex32,
    C32,
    f32,
    2,
    cublas::cuComplex,
    cublasCdotc_v2,
    cublasCgeam,
    crate::kernels::elementwise::axpby_strided_source_complex::launch_unchecked::<
        Complex32,
        CubeclCudaRuntime,
    >
);
impl_cublas_scalar!(
    Complex64,
    C64,
    f64,
    2,
    cublas::cuDoubleComplex,
    cublasZdotc_v2,
    cublasZgeam,
    crate::kernels::elementwise::axpby_strided_source_complex::launch_unchecked::<
        Complex64,
        CubeclCudaRuntime,
    >
);

impl CublasRealScalar for f32 {
    const DOT_NAME: &'static str = "cublasSdot_v2";

    unsafe fn dot(
        handle: cublas::cublasHandle_t,
        n: i32,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t {
        <f32 as CublasScalar>::dotc(handle, n, x, y, result)
    }
}

impl CublasRealScalar for f64 {
    const DOT_NAME: &'static str = "cublasDdot_v2";

    unsafe fn dot(
        handle: cublas::cublasHandle_t,
        n: i32,
        x: *const c_void,
        y: *const c_void,
        result: *mut c_void,
    ) -> cublas::cublasStatus_t {
        <f64 as CublasScalar>::dotc(handle, n, x, y, result)
    }
}
