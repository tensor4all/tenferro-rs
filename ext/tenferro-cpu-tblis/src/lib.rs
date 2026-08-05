#![deny(missing_docs)]

//! External TBLIS-backed `dot_general` provider for `tenferro-cpu`.
//!
//! This crate is a standalone, unpublished extension example. It replaces only
//! the complete general-contraction provider slot; all unsupported contraction
//! shapes and all non-contraction CPU operations remain owned by the selected
//! `tenferro-cpu` fallback backend.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
//! use tenferro_cpu_tblis::TblisGeneralContractionProvider;
//!
//! let bundle = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
//!     .prefer_general_contraction_provider(Arc::new(TblisGeneralContractionProvider::new()))
//!     .build()?;
//! let backend = CpuBackend::new().with_provider_bundle(bundle.clone())?;
//! assert!(backend.provider_bundle().shares_identity_with(&bundle));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::{Mutex, MutexGuard, OnceLock};

use num_complex::{Complex32, Complex64};
use num_traits::One;
use smallvec::SmallVec;
use tblis_ffi::tblis::{
    label_type, len_type, stride_type, tblis_get_num_threads, tblis_scalar, tblis_scalar_scalar,
    tblis_set_num_threads, tblis_tensor, tblis_tensor_mult, type_t, TYPE_DCOMPLEX, TYPE_DOUBLE,
    TYPE_SCOMPLEX, TYPE_SINGLE,
};
use tenferro_cpu::provider::{
    CpuContractionAxes, CpuDotGeneralRequest, CpuExecutionContext, CpuGeneralContractionProvider,
    CpuOperand, CpuProviderOutcome, CpuProviderUnsupported,
};
use tenferro_cpu::{CpuPlacementControl, CpuProviderExecutionCapabilities, CpuThreadCountControl};
use tenferro_tensor::{
    col_major_strides, ContractionScalar, Error, Result, Tensor, TensorRead, TensorScalar,
    TensorView, TensorViewMut, TensorWrite, TypedTensor, TypedTensorView, TypedTensorViewMut,
};

#[cfg(all(feature = "runtime", feature = "source-build"))]
compile_error!(
    "enable at most one TBLIS loading route: use default runtime loading or --no-default-features --features source-build"
);

#[cfg(not(any(feature = "runtime", feature = "source-build")))]
compile_error!("enable one TBLIS loading route: runtime or source-build");

#[cfg(feature = "source-build")]
extern crate tblis_src as _;

const OP: &str = "dot_general";

/// `tenferro-cpu` general-contraction provider backed by TBLIS.
///
/// The adapter uses TBLIS only for supported floating and complex
/// `dot_general` layouts. In preferred mode, unsupported requests are reported
/// without mutating output so that the base CPU provider can run them.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGeneralContractionProvider;
/// use tenferro_cpu_tblis::TblisGeneralContractionProvider;
///
/// let provider: &dyn CpuGeneralContractionProvider =
///     &TblisGeneralContractionProvider::new();
/// let _ = provider.execution_capabilities();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TblisGeneralContractionProvider;

impl TblisGeneralContractionProvider {
    /// Create a TBLIS general-contraction provider.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu_tblis::TblisGeneralContractionProvider;
    ///
    /// let provider = TblisGeneralContractionProvider::new();
    /// assert_eq!(format!("{provider:?}"), "TblisGeneralContractionProvider");
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl CpuGeneralContractionProvider for TblisGeneralContractionProvider {
    /// Return the CPU execution-resource contract for TBLIS calls.
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        CpuProviderExecutionCapabilities {
            thread_count: CpuThreadCountControl::BinaryClampToOne,
            placement: CpuPlacementControl::CallingThread,
            worker_local_sequential: true,
            accepts_sequential: true,
            accepts_outer: true,
            accepts_inner: true,
        }
    }

    /// Execute a validated `dot_general` request through TBLIS when supported.
    ///
    /// # Errors
    ///
    /// Returns a runtime-state error if the provider's process-local TBLIS
    /// thread-control lock is poisoned. Returns invalid-argument errors when
    /// dimensions, strides, offsets, or ranks exceed the C ABI ranges accepted
    /// by TBLIS. Propagates host-buffer access errors from input and output
    /// tensors. Unsupported dtypes, layouts, and unavailable runtime libraries
    /// are reported as [`CpuProviderOutcome::Unsupported`] rather than errors.
    fn dot_general(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        let _ = context;
        execute_general_request(request)
    }
}

trait TblisTensorRead<T> {
    fn shape(&self) -> &[usize];
    fn strides(&self) -> Result<Vec<isize>>;
    fn offset(&self) -> isize;
    fn host_data(&self) -> Result<&[T]>;
}

impl<T: TensorScalar> TblisTensorRead<T> for TypedTensor<T> {
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> Result<Vec<isize>> {
        col_major_strides(self.shape())
    }

    fn offset(&self) -> isize {
        0
    }

    fn host_data(&self) -> Result<&[T]> {
        self.host_data()
    }
}

impl<'a, T: 'static> TblisTensorRead<T> for TypedTensorView<'a, T> {
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> Result<Vec<isize>> {
        Ok(self.strides().to_vec())
    }

    fn offset(&self) -> isize {
        self.offset()
    }

    fn host_data(&self) -> Result<&[T]> {
        self.host_storage()
    }
}

trait TblisGemm: Copy + One + 'static {
    const TYPE: type_t;

    fn scalar(value: Self) -> tblis_scalar;
}

impl TblisGemm for f32 {
    const TYPE: type_t = TYPE_SINGLE;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { s: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for f64 {
    const TYPE: type_t = TYPE_DOUBLE;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { d: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for Complex32 {
    const TYPE: type_t = TYPE_SCOMPLEX;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { c: value },
            type_: Self::TYPE,
        }
    }
}

impl TblisGemm for Complex64 {
    const TYPE: type_t = TYPE_DCOMPLEX;

    fn scalar(value: Self) -> tblis_scalar {
        tblis_scalar {
            data: tblis_scalar_scalar { z: value },
            type_: Self::TYPE,
        }
    }
}

struct TblisPlan {
    lhs_len: SmallVec<[len_type; 8]>,
    rhs_len: SmallVec<[len_type; 8]>,
    out_len: SmallVec<[len_type; 8]>,
    lhs_stride: SmallVec<[stride_type; 8]>,
    rhs_stride: SmallVec<[stride_type; 8]>,
    out_stride: SmallVec<[stride_type; 8]>,
    lhs_labels: SmallVec<[label_type; 8]>,
    rhs_labels: SmallVec<[label_type; 8]>,
    out_labels: SmallVec<[label_type; 8]>,
}

#[derive(Clone, Copy)]
struct TblisExecution<T> {
    alpha: T,
    beta: T,
    lhs_conj: bool,
    rhs_conj: bool,
}

impl<T> TblisExecution<T> {
    fn new(alpha: T, beta: T, lhs_conj: bool, rhs_conj: bool) -> Self {
        Self {
            alpha,
            beta,
            lhs_conj,
            rhs_conj,
        }
    }
}

fn execute_general_request(
    request: CpuDotGeneralRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let (lhs, rhs, output, axes, accumulation) = request.into_parts();
    let dtype = lhs.dtype();
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (accumulation.alpha, accumulation.beta)
            {
                let execution =
                    TblisExecution::new(alpha, beta, accumulation.lhs_conj, accumulation.rhs_conj);
                match (lhs, rhs, &mut *output) {
                    (
                        TensorRead::Tensor(Tensor::$owned(lhs)),
                        TensorRead::Tensor(Tensor::$owned(rhs)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::Tensor(Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(Tensor::$owned(rhs)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::Tensor(Tensor::$owned(lhs)),
                        TensorRead::Tensor(Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::Tensor(Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_request_typed(lhs, rhs, axes, output, execution),
                    _ => {}
                }
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(dtype),
    ))
}

fn execute_request_typed<L, R, T>(
    lhs: &L,
    rhs: &R,
    axes: CpuContractionAxes<'_>,
    output: &mut TypedTensorViewMut<'_, T>,
    execution: TblisExecution<T>,
) -> Result<CpuProviderOutcome>
where
    L: TblisTensorRead<T>,
    R: TblisTensorRead<T>,
    T: TblisGemm,
{
    let output_strides = output.strides().to_vec();
    let Some(plan) = plan_from_axes(lhs, rhs, &axes, output.shape(), &output_strides)? else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Layout(CpuOperand::Output),
        ));
    };
    if !runtime_available()? {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ));
    }
    let lhs_ptr = checked_base_ptr(lhs.host_data()?.as_ptr(), lhs.offset(), "lhs")?;
    let rhs_ptr = checked_base_ptr(rhs.host_data()?.as_ptr(), rhs.offset(), "rhs")?;
    execute(plan, lhs_ptr, rhs_ptr, output, execution)?;
    Ok(CpuProviderOutcome::Executed)
}

fn plan_from_axes<L, R, T>(
    lhs: &L,
    rhs: &R,
    axes: &CpuContractionAxes<'_>,
    out_shape: &[usize],
    out_strides: &[isize],
) -> Result<Option<TblisPlan>>
where
    L: TblisTensorRead<T>,
    R: TblisTensorRead<T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let mut expected_out_shape = SmallVec::<[usize; 8]>::new();
    expected_out_shape.extend(axes.lhs_free_axes().map(|axis| lhs_shape[axis]));
    expected_out_shape.extend(axes.rhs_free_axes().map(|axis| rhs_shape[axis]));
    expected_out_shape.extend(axes.batch_pairs().map(|(lhs_axis, _)| lhs_shape[lhs_axis]));
    if expected_out_shape.as_slice() != out_shape {
        return Ok(None);
    }
    if lhs_rank == 0 || rhs_rank == 0 || out_shape.is_empty() {
        return Ok(None);
    }
    if lhs_shape
        .iter()
        .chain(rhs_shape)
        .chain(out_shape)
        .any(|&dim| dim == 0)
    {
        return Ok(None);
    }
    if lhs.offset() < 0 || rhs.offset() < 0 {
        return Ok(None);
    }

    let lhs_strides = lhs.strides()?;
    let rhs_strides = rhs.strides()?;
    if lhs_strides
        .iter()
        .chain(rhs_strides.iter())
        .chain(out_strides.iter())
        .any(|&stride| stride <= 0)
    {
        return Ok(None);
    }
    if lhs_rank > c_int::MAX as usize
        || rhs_rank > c_int::MAX as usize
        || out_shape.len() > c_int::MAX as usize
    {
        return Ok(None);
    }

    let mut labels = TblisLabelAllocator::new();
    let mut lhs_labels = SmallVec::<[label_type; 8]>::from_elem(0, lhs_rank);
    let mut rhs_labels = SmallVec::<[label_type; 8]>::from_elem(0, rhs_rank);
    let mut out_labels = SmallVec::<[label_type; 8]>::new();
    for axis in axes.lhs_free_axes() {
        let Some(label) = labels.next() else {
            return Ok(None);
        };
        lhs_labels[axis] = label;
        out_labels.push(label);
    }
    for axis in axes.rhs_free_axes() {
        let Some(label) = labels.next() else {
            return Ok(None);
        };
        rhs_labels[axis] = label;
        out_labels.push(label);
    }
    for (lhs_axis, rhs_axis) in axes.batch_pairs() {
        let Some(label) = labels.next() else {
            return Ok(None);
        };
        lhs_labels[lhs_axis] = label;
        rhs_labels[rhs_axis] = label;
        out_labels.push(label);
    }
    for (lhs_axis, rhs_axis) in axes.contracting_pairs() {
        let Some(label) = labels.next() else {
            return Ok(None);
        };
        lhs_labels[lhs_axis] = label;
        rhs_labels[rhs_axis] = label;
    }
    if lhs_labels.contains(&0) || rhs_labels.contains(&0) {
        return Ok(None);
    }

    lhs_labels.push(0);
    rhs_labels.push(0);
    out_labels.push(0);
    Ok(Some(TblisPlan {
        lhs_len: dims_to_tblis(lhs_shape)?,
        rhs_len: dims_to_tblis(rhs_shape)?,
        out_len: dims_to_tblis(out_shape)?,
        lhs_stride: strides_to_tblis(lhs_strides.as_slice())?,
        rhs_stride: strides_to_tblis(rhs_strides.as_slice())?,
        out_stride: strides_to_tblis(out_strides)?,
        lhs_labels,
        rhs_labels,
        out_labels,
    }))
}

fn execute<T>(
    mut plan: TblisPlan,
    lhs_ptr: *const T,
    rhs_ptr: *const T,
    out: &mut TypedTensorViewMut<'_, T>,
    execution: TblisExecution<T>,
) -> Result<()>
where
    T: TblisGemm,
{
    let _runtime = TblisRuntimeCall::enter()?;
    let out_storage_len = out.host_storage()?.len();
    let out_offset =
        checked_output_base_offset(out.shape(), out.strides(), out.offset(), out_storage_len)?;
    let out_storage = out.host_storage_mut()?;
    let out_ptr = out_storage.as_mut_ptr();

    let mut lhs_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: c_int::from(execution.lhs_conj),
        scalar: T::scalar(execution.alpha),
        data: lhs_ptr.cast_mut() as *mut c_void,
        ndim: c_int::try_from(plan.lhs_len.len()).map_err(|_| {
            Error::invalid_argument(OP, "configuration", "TBLIS lhs rank exceeds c_int range")
        })?,
        len: plan.lhs_len.as_mut_ptr(),
        stride: plan.lhs_stride.as_mut_ptr(),
    };
    let mut rhs_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: c_int::from(execution.rhs_conj),
        scalar: T::scalar(T::one()),
        data: rhs_ptr.cast_mut() as *mut c_void,
        ndim: c_int::try_from(plan.rhs_len.len()).map_err(|_| {
            Error::invalid_argument(OP, "configuration", "TBLIS rhs rank exceeds c_int range")
        })?,
        len: plan.rhs_len.as_mut_ptr(),
        stride: plan.rhs_stride.as_mut_ptr(),
    };
    let mut out_tensor = tblis_tensor {
        type_: T::TYPE,
        conj: 0,
        scalar: T::scalar(execution.beta),
        data: out_ptr.wrapping_add(out_offset) as *mut c_void,
        ndim: c_int::try_from(plan.out_len.len()).map_err(|_| {
            Error::invalid_argument(OP, "configuration", "TBLIS output rank exceeds c_int range")
        })?,
        len: plan.out_len.as_mut_ptr(),
        stride: plan.out_stride.as_mut_ptr(),
    };

    // SAFETY: the CPU runtime validated dtype, shapes, axes, and reachable
    // ranges before provider entry. This adapter additionally accepts only
    // positive strides and host-backed operands, clamps TBLIS to one thread for
    // the duration of the call, and keeps all descriptor buffers live.
    unsafe {
        tblis_tensor_mult(
            std::ptr::null(),
            std::ptr::null(),
            &lhs_tensor,
            plan.lhs_labels.as_ptr(),
            &rhs_tensor,
            plan.rhs_labels.as_ptr(),
            &mut out_tensor,
            plan.out_labels.as_ptr(),
        );
    }
    let _ = (&mut lhs_tensor, &mut rhs_tensor);
    Ok(())
}

struct TblisRuntimeCall<'a> {
    _lock: MutexGuard<'a, ()>,
    previous_threads: c_uint,
}

impl TblisRuntimeCall<'_> {
    fn enter() -> Result<Self> {
        let lock = tblis_runtime_lock().lock().map_err(|_| {
            Error::runtime_state(OP, "TBLIS runtime thread-control lock was poisoned")
        })?;
        let previous_threads = unsafe { tblis_get_num_threads() };
        unsafe {
            tblis_set_num_threads(1);
        }
        Ok(Self {
            _lock: lock,
            previous_threads,
        })
    }
}

impl Drop for TblisRuntimeCall<'_> {
    fn drop(&mut self) {
        unsafe {
            tblis_set_num_threads(self.previous_threads);
        }
    }
}

fn tblis_runtime_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn runtime_available() -> Result<bool> {
    #[cfg(feature = "runtime")]
    {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();

        Ok(*AVAILABLE.get_or_init(|| {
            // INVARIANT: `tblis-ffi` 0.2.6 exposes only panic-based dynamic
            // loading. This one-time, pre-FFI availability probe caches the
            // result and never catches a panic from a native TBLIS kernel.
            std::panic::catch_unwind(|| unsafe {
                tblis_ffi::tblis::dyload_lib();
            })
            .is_ok()
        }))
    }
    #[cfg(feature = "source-build")]
    {
        Ok(true)
    }
}

fn checked_base_ptr<T>(base: *const T, offset: isize, operand: &'static str) -> Result<*const T> {
    if offset < 0 {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            format!("TBLIS {operand} offset must be non-negative"),
        ));
    }
    Ok(base.wrapping_add(usize::try_from(offset).map_err(|_| {
        Error::invalid_argument(
            OP,
            "configuration",
            format!("TBLIS {operand} offset does not fit in usize"),
        )
    })?))
}

fn checked_output_base_offset(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    storage_len: usize,
) -> Result<usize> {
    if offset < 0 {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output offset must be non-negative",
        ));
    }
    if shape.len() != strides.len() {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output shape and stride ranks must match",
        ));
    }
    let base = usize::try_from(offset).map_err(|_| {
        Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output offset does not fit in usize",
        )
    })?;
    let mut max_delta = 0usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim == 0 {
            return Err(Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output dimensions must be non-zero",
            ));
        }
        if stride <= 0 {
            return Err(Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output strides must be positive",
            ));
        }
        let stride = usize::try_from(stride).map_err(|_| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output stride does not fit in usize",
            )
        })?;
        let axis_delta = (dim - 1).checked_mul(stride).ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output reachable range overflows usize",
            )
        })?;
        max_delta = max_delta.checked_add(axis_delta).ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                "TBLIS output reachable range overflows usize",
            )
        })?;
    }
    let max_index = base.checked_add(max_delta).ok_or_else(|| {
        Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output reachable range overflows usize",
        )
    })?;
    if max_index >= storage_len {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "TBLIS output reachable range is outside host storage",
        ));
    }
    Ok(base)
}

fn dims_to_tblis(dims: &[usize]) -> Result<SmallVec<[len_type; 8]>> {
    dims.iter()
        .map(|&dim| {
            len_type::try_from(dim).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "configuration",
                    format!("TBLIS dimension {dim} exceeds len_type range"),
                )
            })
        })
        .collect()
}

fn strides_to_tblis(strides: &[isize]) -> Result<SmallVec<[stride_type; 8]>> {
    strides
        .iter()
        .map(|&stride| {
            stride_type::try_from(stride).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "configuration",
                    format!("TBLIS stride {stride} exceeds stride_type range"),
                )
            })
        })
        .collect()
}

struct TblisLabelAllocator {
    next: usize,
}

impl TblisLabelAllocator {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn next(&mut self) -> Option<label_type> {
        const LABELS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let byte = *LABELS.get(self.next)?;
        self.next += 1;
        Some(byte as c_char)
    }
}
