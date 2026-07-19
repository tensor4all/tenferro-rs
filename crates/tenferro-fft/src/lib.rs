//! FFT extension operations for tenferro.
//!
//! This crate is an out-of-tree `ExtensionOp` package with an explicit
//! [`FftBackend`] capability. [`tenferro_cpu::CpuBackend`] implements the
//! capability through RustFFT. Metal/WebGPU and CUDA backends require separate
//! explicit implementations; unsupported requests return an error and never
//! fall back to CPU or transfer tensor data. Concrete non-AD execution uses
//! [`TensorFftExt`] and [`TensorReadFftExt`]. Eager execution uses
//! `EagerTensorFftExt` when `autodiff` is enabled, and traced graph
//! construction uses [`TracedTensorFftExt`].
//!
//! # Examples
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//! use tenferro_fft::{FftNorm, TracedTensorFftExt};
//!
//! let x = TracedTensor::from_vec_col_major(
//!     vec![4],
//!     vec![
//!         Complex64::new(1.0, 0.0),
//!         Complex64::new(2.0, 0.0),
//!         Complex64::new(3.0, 0.0),
//!         Complex64::new(4.0, 0.0),
//!     ],
//! )
//! .unwrap();
//! let y = x.fft(None, -1, FftNorm::Backward).unwrap();
//!
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! executor.register_extension(tenferro_fft::register_runtime).unwrap();
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.shape(), &[4]);
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! ```
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_fft::{FftNorm, TensorFftExt};
//! use tenferro_tensor::Tensor;
//!
//! let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let mut backend = CpuBackend::new();
//! let out = x.fft(None, -1, FftNorm::Backward, &mut backend).unwrap();
//!
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! ```

use std::any::Any;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
#[cfg(feature = "autodiff")]
use tenferro_ad::extension::{
    ExtensionLinearTransposeRule, ExtensionLinearizeRule, ExtensionPrimalVjpRule,
    ExtensionRegistryError, ExtensionRuleSet,
};
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "autodiff")]
use tenferro_ops::ad::{transpose_input::TransposeInputRef, PrimitiveRuleBuilder};
#[cfg(feature = "autodiff")]
use tenferro_ops::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ShapeGuardContext;
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{apply, ExtensionExecutionContext, ExtensionOp, HostReference};
use tenferro_runtime::{Error, ErrorPhase, Result, TracedTensor};
use tenferro_tensor::{CacheStats, DType, ErrorKind, Tensor, TensorRead, ValidationError};
#[cfg(feature = "autodiff")]
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

mod backend;
mod cache;
mod cpu;
#[cfg(feature = "autodiff")]
mod eager_ext;
mod spec;

pub use backend::{FftBackend, FftExecutionCache};
pub use cache::{
    fft_plan_cache_selector, FftPlanCache, DEFAULT_FFT_PLAN_CACHE_CAPACITY, FFT_PLAN_CACHE_NAME,
};
#[cfg(feature = "autodiff")]
pub use eager_ext::EagerTensorFftExt;
pub use spec::{FftNorm, FftOperation, FftPlanSpec};

/// Extension family id used by the tenferro FFT extension.
///
/// # Examples
///
/// ```
/// assert_eq!(
///     tenferro_fft::FFT_EXTENSION_FAMILY_ID,
///     "tenferro-fft.fft.v1"
/// );
/// ```
pub const FFT_EXTENSION_FAMILY_ID: &str = "tenferro-fft.fft.v1";

/// Reusable concrete FFT executor with an explicitly owned backend-neutral cache.
#[derive(Default)]
pub struct FftExecutor {
    plans: FftPlanCache,
}

impl FftExecutor {
    /// Create an executor from a caller-configured FFT execution cache.
    pub fn new(plans: FftPlanCache) -> Self {
        Self { plans }
    }

    /// Inspect the owned backend-neutral FFT cache.
    pub const fn plan_cache(&self) -> &FftPlanCache {
        &self.plans
    }

    /// Mutably inspect or configure the owned backend-neutral FFT cache.
    pub fn plan_cache_mut(&mut self) -> &mut FftPlanCache {
        &mut self.plans
    }

    /// Snapshot aggregate statistics for every backend cache namespace.
    pub fn cache_stats(&self) -> CacheStats {
        self.plans.stats()
    }

    /// Remove every retained backend plan or workspace from this executor.
    pub fn clear_cache(&mut self) {
        self.plans.clear();
    }

    /// Execute a complex or full-spectrum real FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for unsupported dtypes, or a typed backend source for execution.
    pub fn fft<B: FftBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_fft_operation("FftExecutor::fft", input.dtype())?,
            "FftExecutor::fft",
            n,
            axis,
            norm,
            backend,
        )
    }

    /// Execute an inverse complex FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-complex input, or a typed backend source for execution.
    pub fn ifft<B: FftBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_ifft_operation("FftExecutor::ifft", input.dtype())?,
            "FftExecutor::ifft",
            n,
            axis,
            norm,
            backend,
        )
    }

    /// Execute a real FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-real input, or a typed backend source for execution.
    pub fn rfft<B: FftBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_rfft_operation("FftExecutor::rfft", input.dtype())?,
            "FftExecutor::rfft",
            n,
            axis,
            norm,
            backend,
        )
    }

    /// Execute an inverse real FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`,
    /// `InvalidArgument`, or spectrum-length details,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-complex input, or a typed backend source for execution.
    pub fn irfft<B: FftBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_irfft_operation("FftExecutor::irfft", input.dtype())?,
            "FftExecutor::irfft",
            n,
            axis,
            norm,
            backend,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute<B: FftBackend>(
        &mut self,
        input: &Tensor,
        operation: FftOperation,
        op_name: &'static str,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let spec = concrete_fft_spec(
            op_name,
            operation,
            input.dtype(),
            input.shape(),
            n,
            axis,
            norm,
        )?;
        backend.execute_fft(
            input,
            &spec,
            FftExecutionCache::caller_owned(&mut self.plans),
        )
    }
}

/// FFT extension methods for [`TracedTensor`].
pub trait TracedTensorFftExt {
    /// Build a traced complex or full-spectrum real FFT.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`, or `Error::Extension` with
    /// `ErrorKind::Unsupported` for integer, boolean, or otherwise unsupported
    /// dtypes.
    ///
    /// # Deferred errors
    ///
    /// Symbolic axis extents and extension execution failures are checked at
    /// compile or execution time after concrete inputs are bound.
    fn fft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor>;

    /// Build a traced inverse complex FFT.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`, or `Error::Extension` with
    /// `ErrorKind::Unsupported` when the input is not `C32`/`C64`.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape and extension execution failures may be deferred to
    /// compile or execution.
    fn ifft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor>;

    /// Build a traced one-sided real FFT.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`, or `Error::Extension` with
    /// `ErrorKind::Unsupported` when the input is not `F32`/`F64`.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape and extension execution failures may be deferred to
    /// compile or execution.
    fn rfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor>;

    /// Build a traced inverse one-sided real FFT.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n` or spectrum length, or
    /// `Error::Extension` with `ErrorKind::Unsupported` for non-complex input.
    ///
    /// # Deferred errors
    ///
    /// Symbolic spectrum lengths and extension execution failures may be
    /// deferred to compile or execution.
    fn irfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor>;
}

impl TracedTensorFftExt for TracedTensor {
    fn fft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
        fft(self, n, axis, norm)
    }

    fn ifft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
        ifft(self, n, axis, norm)
    }

    fn rfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
        rfft(self, n, axis, norm)
    }

    fn irfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
        irfft(self, n, axis, norm)
    }
}

/// Backend-explicit FFT methods for concrete [`Tensor`] values.
///
/// This is the non-AD immediate execution surface. It uses unsuffixed method
/// names because the receiver is an owned compact tensor value. Use
/// [`TensorReadFftExt`] when the input is a borrowed view or other
/// [`TensorRead`] value.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_fft::{FftNorm, TensorFftExt};
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let mut backend = CpuBackend::new();
///
/// let spectrum = input.fft(None, -1, FftNorm::Backward, &mut backend)?;
/// assert_eq!(spectrum.shape(), &[4]);
/// assert_eq!(spectrum.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorFftExt {
    /// Execute a one-dimensional FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for an
    /// integer or boolean input, or a typed backend source for execution.
    fn fft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-complex input, or a typed backend source for execution.
    fn ifft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-`F32`/`F64` input, or a typed backend source for execution.
    fn rfft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds`, `InvalidArgument`,
    /// or spectrum-length details, `Error::Extension` with
    /// `ErrorKind::Unsupported` for a non-complex input, or a typed backend
    /// source for execution.
    fn irfft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorFftExt for Tensor {
    fn fft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let spec = concrete_fft_spec(
            "TensorFftExt::fft",
            concrete_fft_operation("TensorFftExt::fft", self.dtype())?,
            self.dtype(),
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &spec, backend)
    }

    fn ifft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let spec = concrete_fft_spec(
            "TensorFftExt::ifft",
            concrete_ifft_operation("TensorFftExt::ifft", self.dtype())?,
            self.dtype(),
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &spec, backend)
    }

    fn rfft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let spec = concrete_fft_spec(
            "TensorFftExt::rfft",
            concrete_rfft_operation("TensorFftExt::rfft", self.dtype())?,
            self.dtype(),
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &spec, backend)
    }

    fn irfft<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let spec = concrete_fft_spec(
            "TensorFftExt::irfft",
            concrete_irfft_operation("TensorFftExt::irfft", self.dtype())?,
            self.dtype(),
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &spec, backend)
    }
}

/// Backend-explicit FFT methods for read-only tensor inputs.
///
/// The `_read` suffix follows the repository convention for APIs that
/// explicitly accept [`TensorRead`] values such as borrowed views.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_fft::{FftNorm, TensorReadFftExt};
/// use tenferro_tensor::{TensorRead, TensorView};
///
/// let shape = [4usize];
/// let data = [1.0_f64, 2.0, 3.0, 4.0];
/// let input = TensorRead::from_view(TensorView::f64(&shape, &data)?);
/// let mut backend = CpuBackend::new();
///
/// let spectrum = input.fft_read(None, -1, FftNorm::Backward, &mut backend)?;
/// assert_eq!(spectrum.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorReadFftExt {
    /// Execute a one-dimensional FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for an
    /// integer or boolean input, or a typed backend source for materialization
    /// or execution.
    fn fft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-complex input, or a typed backend source for materialization.
    fn ifft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-`F32`/`F64` input, or a typed backend source for materialization.
    fn rfft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds`, `InvalidArgument`,
    /// or spectrum-length details, `Error::Extension` with
    /// `ErrorKind::Unsupported` for a non-complex input, or a typed backend
    /// source for materialization.
    fn irfft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorReadFftExt for TensorRead<'_> {
    fn fft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_fft_operation("TensorReadFftExt::fft_read", self.dtype())?,
            "TensorReadFftExt::fft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn ifft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_ifft_operation("TensorReadFftExt::ifft_read", self.dtype())?,
            "TensorReadFftExt::ifft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn rfft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_rfft_operation("TensorReadFftExt::rfft_read", self.dtype())?,
            "TensorReadFftExt::rfft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn irfft_read<B: FftBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_irfft_operation("TensorReadFftExt::irfft_read", self.dtype())?,
            "TensorReadFftExt::irfft_read",
            n,
            axis,
            norm,
            backend,
        )
    }
}

#[derive(Debug, thiserror::Error)]
enum FftError {
    #[error("{op} does not support dtype {dtype:?}; expected {expected}")]
    UnsupportedDType {
        op: &'static str,
        dtype: DType,
        expected: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct FftOp {
    operation: FftOperation,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
}

impl FftOp {
    fn new(operation: FftOperation, axis: usize, n: Option<usize>, norm: FftNorm) -> Self {
        Self {
            operation,
            axis,
            n,
            norm,
        }
    }

    #[cfg(feature = "autodiff")]
    fn c2c_adjoint(&self) -> Option<Self> {
        match self.operation {
            FftOperation::C2cForward => Some(Self {
                operation: FftOperation::C2cInverse,
                axis: self.axis,
                n: self.n,
                norm: self.norm.c2c_adjoint(),
            }),
            FftOperation::C2cInverse => Some(Self {
                operation: FftOperation::C2cForward,
                axis: self.axis,
                n: self.n,
                norm: self.norm.c2c_adjoint(),
            }),
            FftOperation::R2cFull | FftOperation::R2cOnesided | FftOperation::C2r => None,
        }
    }
}

impl ExtensionOp for FftOp {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        let operation = match self.operation {
            FftOperation::C2cForward => 0,
            FftOperation::C2cInverse => 1,
            FftOperation::R2cOnesided => 2,
            FftOperation::R2cFull => 3,
            FftOperation::C2r => 4,
        };
        hasher.write_u8(operation);
        hasher.write_usize(self.axis);
        match self.n {
            Some(n) => {
                hasher.write_u8(1);
                hasher.write_usize(n);
            }
            None => hasher.write_u8(0),
        }
        let norm = match self.norm {
            FftNorm::Backward => 0,
            FftNorm::Forward => 1,
            FftNorm::Ortho => 2,
        };
        hasher.write_u8(norm);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<FftOp>()
            .is_some_and(|that| self == that)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtype = ctx.input_dtype(0)?;
        let input_shape = ctx.input_shape(0)?;
        if self.axis >= input_shape.len() {
            return Err(tenferro_tensor::Error::axis_out_of_bounds(
                "tenferro-fft",
                self.axis,
                input_shape.len(),
            ));
        }

        let mut out_shape = input_shape.to_vec();
        let output_dtype = match self.operation {
            FftOperation::C2cForward | FftOperation::C2cInverse => {
                if !matches!(input_dtype, DType::C32 | DType::C64) {
                    return Err(tensor_unsupported_dtype(
                        "tenferro-fft",
                        input_dtype,
                        "C32 or C64",
                    ));
                }
                input_dtype
            }
            FftOperation::R2cFull | FftOperation::R2cOnesided => {
                let len = transform_len_dim(self.n, &input_shape[self.axis]);
                out_shape[self.axis] = if self.operation.is_onesided() {
                    len / 2usize + 1usize
                } else {
                    len
                };
                match input_dtype {
                    DType::F32 => DType::C32,
                    DType::F64 => DType::C64,
                    _ => {
                        return Err(tensor_unsupported_dtype(
                            "tenferro-fft",
                            input_dtype,
                            "F32 or F64",
                        ));
                    }
                }
            }
            FftOperation::C2r => {
                out_shape[self.axis] = output_dim_c2r(&input_shape[self.axis], self.n)?;
                match input_dtype {
                    DType::C32 => DType::F32,
                    DType::C64 => DType::F64,
                    _ => {
                        return Err(tensor_unsupported_dtype(
                            "tenferro-fft",
                            input_dtype,
                            "C32 or C64",
                        ));
                    }
                }
            }
        };

        if self.operation.is_c2c() {
            out_shape[self.axis] = transform_len_dim(self.n, &input_shape[self.axis]);
        }

        Ok(vec![(output_dtype, out_shape)])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for FftOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        execute_host_fft_op(self, inputs)
    }
}

fn execute_host_fft_op(op: &FftOp, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "inputs",
            format!("expected 1 input, got {}", inputs.len()),
        ));
    }
    let input = inputs[0];
    let spec = validated_fft_plan_spec(
        fft_op_name(op.operation),
        op.operation,
        input.dtype(),
        input.shape(),
        op.n,
        op.axis,
        op.norm,
    )?;
    cpu::validate_host_fft_input(fft_op_name(op.operation), input)?;
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    let output = cpu::execute_fft_with_plans(input, &spec, &mut plans)?;
    Ok(vec![output])
}

fn execute_concrete_fft_op<B: FftBackend>(
    input: &Tensor,
    spec: &FftPlanSpec,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    backend.execute_fft(input, spec, FftExecutionCache::caller_owned(&mut plans))
}

#[allow(clippy::too_many_arguments)]
fn execute_concrete_fft_read_op<B: FftBackend>(
    input: &TensorRead<'_>,
    operation: FftOperation,
    op_name: &'static str,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let spec = concrete_fft_spec(
        op_name,
        operation,
        input.dtype(),
        input.shape(),
        n,
        axis,
        norm,
    )?;
    let materialized =
        backend.with_backend_session(|exec| exec.to_contiguous_read(input.clone()))?;
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    backend.execute_fft(
        &materialized,
        &spec,
        FftExecutionCache::caller_owned(&mut plans),
    )
}

#[allow(clippy::too_many_arguments)]
fn concrete_fft_spec(
    op: &'static str,
    operation: FftOperation,
    input_dtype: DType,
    input_shape: &[usize],
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> tenferro_tensor::Result<FftPlanSpec> {
    validate_concrete_n(op, n)?;
    let axis = normalize_concrete_axis(op, axis, input_shape.len())?;
    validated_fft_plan_spec(op, operation, input_dtype, input_shape, n, axis, norm)
}

#[allow(clippy::too_many_arguments)]
fn validated_fft_plan_spec(
    op: &'static str,
    operation: FftOperation,
    input_dtype: DType,
    input_shape: &[usize],
    n: Option<usize>,
    axis: usize,
    norm: FftNorm,
) -> tenferro_tensor::Result<FftPlanSpec> {
    validate_concrete_n(op, n)?;
    validate_operation_dtype(op, operation, input_dtype)?;
    validate_axis(op, input_shape, axis)?;
    validate_concrete_transform_len(op, input_shape, n, axis)?;
    if operation == FftOperation::C2r {
        output_shape_c2r(input_shape, axis, n)?;
    }
    Ok(FftPlanSpec::new(
        operation,
        axis,
        n,
        norm,
        input_dtype,
        input_shape.to_vec(),
    ))
}

fn concrete_fft_operation(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2cForward),
        DType::F32 | DType::F64 => Ok(FftOperation::R2cFull),
        DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "F32, F64, C32, or C64"))
        }
    }
}

fn concrete_ifft_operation(
    op: &'static str,
    dtype: DType,
) -> tenferro_tensor::Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2cInverse),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "C32 or C64"))
        }
    }
}

fn concrete_rfft_operation(
    op: &'static str,
    dtype: DType,
) -> tenferro_tensor::Result<FftOperation> {
    match dtype {
        DType::F32 | DType::F64 => Ok(FftOperation::R2cOnesided),
        DType::C32 | DType::C64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "F32 or F64"))
        }
    }
}

fn concrete_irfft_operation(
    op: &'static str,
    dtype: DType,
) -> tenferro_tensor::Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2r),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "C32 or C64"))
        }
    }
}

fn validate_operation_dtype(
    op: &'static str,
    operation: FftOperation,
    dtype: DType,
) -> tenferro_tensor::Result<()> {
    let supported = match operation {
        FftOperation::C2cForward | FftOperation::C2cInverse | FftOperation::C2r => {
            matches!(dtype, DType::C32 | DType::C64)
        }
        FftOperation::R2cFull | FftOperation::R2cOnesided => {
            matches!(dtype, DType::F32 | DType::F64)
        }
    };
    if supported {
        Ok(())
    } else {
        Err(tensor_unsupported_dtype(
            op,
            dtype,
            expected_dtype_description(operation),
        ))
    }
}

fn validate_concrete_n(op: &'static str, n: Option<usize>) -> tenferro_tensor::Result<()> {
    if n == Some(0) {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "n",
            "transform length must be positive",
        ));
    }
    Ok(())
}

fn validate_concrete_transform_len(
    op: &'static str,
    input_shape: &[usize],
    n: Option<usize>,
    axis: usize,
) -> tenferro_tensor::Result<()> {
    if n.is_none() && input_shape.get(axis).copied() == Some(0) {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "n",
            "transform length must be positive",
        ));
    }
    Ok(())
}

fn normalize_concrete_axis(
    op: &'static str,
    axis: isize,
    rank: usize,
) -> tenferro_tensor::Result<usize> {
    if rank == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "rank",
            "FFT requires rank >= 1",
        ));
    }
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs()).ok_or_else(|| {
            tenferro_tensor::Error::axis_out_of_bounds(op, axis.unsigned_abs(), rank)
        })?
    };
    if normalized >= rank {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(
            op, normalized, rank,
        ));
    }
    Ok(normalized)
}

fn tensor_unsupported_dtype(
    op: &'static str,
    dtype: DType,
    expected: &'static str,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::extension(
        op,
        FFT_EXTENSION_FAMILY_ID,
        ErrorKind::Unsupported,
        FftError::UnsupportedDType {
            op,
            dtype,
            expected,
        },
    )
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct FftAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionLinearizeRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let fft_op = fft_payload(op, ADRuleKind::Jvp)?;
        if !fft_op.operation.is_c2c() {
            return Err(ADRuleError::unsupported(
                fft_ad_family_id(fft_op.operation),
                ADRuleKind::Jvp,
            ));
        }

        match tangent_in[0] {
            Some(dx) => {
                let outputs = builder.add_operation(
                    StdTensorOp::Extension(Arc::new(fft_op.clone())),
                    vec![ValueRef::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true],
                    },
                );
                Ok(vec![Some(outputs[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionLinearTransposeRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[tidu::PrimitiveTransposeInput<StdTensorOp>],
        active_mask: &[bool],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let inputs: Vec<_> = inputs.iter().map(TransposeInputRef::new).collect();
        transpose_fft_adjoint_from_transpose_inputs(
            op,
            builder,
            cotangent_out,
            &inputs,
            active_mask,
            ctx,
        )
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionPrimalVjpRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn primal_vjp(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        transpose_fft_adjoint(op, builder, cotangent_out, inputs, None, ctx)
    }
}

#[cfg(feature = "autodiff")]
fn transpose_fft_adjoint(
    op: &dyn ExtensionOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[ValueRef<StdTensorOp>],
    active_mask: Option<&[bool]>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some((adjoint, fft_op)) = emit_c2c_adjoint(op, builder, cotangent_out, active_mask)? else {
        return Ok(vec![None]);
    };
    let restored = restore_c2c_adjoint_input_length(builder, adjoint, inputs, fft_op, ctx)?;
    Ok(vec![Some(restored)])
}

#[cfg(feature = "autodiff")]
fn transpose_fft_adjoint_from_transpose_inputs(
    op: &dyn ExtensionOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    inputs: &[TransposeInputRef<'_>],
    active_mask: &[bool],
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValueId>>> {
    let Some((adjoint, fft_op)) = emit_c2c_adjoint(op, builder, cotangent_out, Some(active_mask))?
    else {
        return Ok(vec![None]);
    };
    let restored = restore_c2c_adjoint_input_length_from_transpose_input(
        builder, adjoint, inputs, fft_op, ctx,
    )?;
    Ok(vec![Some(restored)])
}

#[cfg(feature = "autodiff")]
fn emit_c2c_adjoint<'a>(
    op: &'a dyn ExtensionOp,
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent_out: &[Option<LocalValueId>],
    active_mask: Option<&[bool]>,
) -> ADRuleResult<Option<(LocalValueId, &'a FftOp)>> {
    let fft_op = fft_payload(op, ADRuleKind::Transpose)?;
    if !fft_op.operation.is_c2c() {
        return Err(ADRuleError::unsupported(
            fft_ad_family_id(fft_op.operation),
            ADRuleKind::Transpose,
        ));
    }
    if active_mask.is_some_and(|mask| !mask.first().copied().unwrap_or(false)) {
        return Ok(None);
    }

    let Some(ct) = cotangent_out.first().copied().flatten() else {
        return Ok(None);
    };
    let adjoint_op = fft_op
        .c2c_adjoint()
        .ok_or_else(|| ADRuleError::unsupported(FFT_EXTENSION_FAMILY_ID, ADRuleKind::Transpose))?;
    let outputs = builder.add_operation(
        StdTensorOp::Extension(Arc::new(adjoint_op)),
        vec![ValueRef::Local(ct)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    Ok(Some((outputs[0], fft_op)))
}

#[cfg(feature = "autodiff")]
fn restore_c2c_adjoint_input_length(
    builder: &mut dyn PrimitiveRuleBuilder,
    adjoint: LocalValueId,
    inputs: &[ValueRef<StdTensorOp>],
    fft_op: &FftOp,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<LocalValueId> {
    let Some(transform_len) = fft_op.n else {
        return Ok(adjoint);
    };
    let Some(input) = inputs.first() else {
        return Err(ADRuleError::invalid_input(
            FFT_EXTENSION_FAMILY_ID,
            ADRuleKind::Transpose,
            "FFT transpose rule expected one primal input",
        ));
    };
    if ctx
        .shape_of(input)
        .ok()
        .and_then(|shape| shape.get(fft_op.axis).and_then(SymDim::constant_value))
        == Some(transform_len)
    {
        return Ok(adjoint);
    }

    let size = builder.add_operation(
        StdTensorOp::ShapeOf { axis: fft_op.axis },
        vec![input.clone()],
        OperationRole::Linearized {
            active_mask: vec![false],
        },
    )[0];
    let truncated = builder.add_operation(
        StdTensorOp::DynamicTruncate { axis: fft_op.axis },
        vec![ValueRef::Local(adjoint), ValueRef::Local(size)],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0];
    let padded = builder.add_operation(
        StdTensorOp::PadToMatch { axis: fft_op.axis },
        vec![ValueRef::Local(truncated), input.clone()],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0];
    Ok(padded)
}

#[cfg(feature = "autodiff")]
fn restore_c2c_adjoint_input_length_from_transpose_input(
    builder: &mut dyn PrimitiveRuleBuilder,
    adjoint: LocalValueId,
    inputs: &[TransposeInputRef<'_>],
    fft_op: &FftOp,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<LocalValueId> {
    let Some(transform_len) = fft_op.n else {
        return Ok(adjoint);
    };
    let Some(input) = inputs.first() else {
        return Err(ADRuleError::invalid_input(
            FFT_EXTENSION_FAMILY_ID,
            ADRuleKind::Transpose,
            "FFT transpose rule expected one primal input",
        ));
    };
    let metadata = input.metadata_value();
    if ctx
        .shape_of(&metadata)
        .ok()
        .and_then(|shape| shape.get(fft_op.axis).and_then(SymDim::constant_value))
        == Some(transform_len)
    {
        return Ok(adjoint);
    }

    let shape_source = input.shape_source_value(FFT_EXTENSION_FAMILY_ID, 0)?;
    let size = builder.add_operation(
        StdTensorOp::ShapeOf { axis: fft_op.axis },
        vec![shape_source.clone()],
        OperationRole::Linearized {
            active_mask: vec![false],
        },
    )[0];
    let truncated = builder.add_operation(
        StdTensorOp::DynamicTruncate { axis: fft_op.axis },
        vec![ValueRef::Local(adjoint), ValueRef::Local(size)],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0];
    let padded = builder.add_operation(
        StdTensorOp::PadToMatch { axis: fft_op.axis },
        vec![ValueRef::Local(truncated), shape_source],
        OperationRole::Linearized {
            active_mask: vec![true, false],
        },
    )[0];
    Ok(padded)
}

/// Return the explicit FFT extension AD rule set.
#[cfg(feature = "autodiff")]
///
/// # Errors
///
/// Returns [`ExtensionRegistryError::MalformedFamilyId`] if the FFT family
/// identifier is invalid, or [`ExtensionRegistryError::DuplicateRule`] if a
/// rule for the family and role is already registered.
pub fn ad_rules() -> std::result::Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new()
        .with_linearize(Arc::new(FftAdRule))?
        .with_linear_transpose(Arc::new(FftAdRule))?
        .with_primal_vjp(Arc::new(FftAdRule))
}

fn execute_fft_extension<B: FftBackend + 'static>(
    op: &FftOp,
    inputs: &[&Tensor],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "inputs",
            format!("expected 1 input, got {}", inputs.len()),
        ));
    }
    let input = inputs[0];
    let spec = validated_fft_plan_spec(
        fft_op_name(op.operation),
        op.operation,
        input.dtype(),
        input.shape(),
        op.n,
        op.axis,
        op.norm,
    )?;
    let (backend, caches) = ctx.parts_mut();
    let output = backend.execute_fft(input, &spec, FftExecutionCache::runtime_owned(caches))?;
    Ok(vec![output])
}

fn execute_fft_extension_reads<B: FftBackend + 'static>(
    op: &FftOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // FFT capabilities consume compact tensors on their existing placement;
    // materialization uses the explicitly selected backend session.
    let materialized_inputs = ctx.backend_mut().with_backend_session(|exec| {
        inputs
            .iter()
            .cloned()
            .map(|input| exec.to_contiguous_read(input))
            .collect::<tenferro_tensor::Result<Vec<_>>>()
    })?;
    let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
    execute_fft_extension(op, &input_refs, ctx)
}

define_extension_runtime! {
    runtime = FftRuntime,
    family_id = FFT_EXTENSION_FAMILY_ID,
    op_type = FftOp,
    execute = execute_fft_extension,
    execute_reads = execute_fft_extension_reads,
    register_fn = register_runtime,
    backend_bound = FftBackend,
}

/// Build a one-dimensional FFT along `axis`.
///
/// Complex inputs use a complex-to-complex transform. Real inputs use a
/// real-to-complex transform that returns the full complex spectrum.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]).unwrap();
/// let y = x.fft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
/// ```
fn fft(input: &TracedTensor, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
    let operation = runtime_forward_fft_operation(input.dtype)?;
    apply_unary_fft("fft", input, operation, n, axis, norm)
}

/// Build a one-dimensional inverse FFT along `axis`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let spectrum = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)]).unwrap();
/// let y = spectrum.ifft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(1.0, 0.0));
/// ```
fn ifft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype("ifft", input.dtype, &[DType::C32, DType::C64], "C32 or C64")?;
    apply_unary_fft("ifft", input, FftOperation::C2cInverse, n, axis, norm)
}

/// Build a one-dimensional real FFT along `axis`.
///
/// The output keeps only the Hermitian one-sided spectrum with axis length
/// `n / 2 + 1`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = x.rfft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.shape(), &[2]);
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
/// ```
fn rfft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype("rfft", input.dtype, &[DType::F32, DType::F64], "F32 or F64")?;
    apply_unary_fft("rfft", input, FftOperation::R2cOnesided, n, axis, norm)
}

/// Build a one-dimensional inverse real FFT along `axis`.
///
/// If `n` is `None`, the output length is inferred as twice one less than the
/// input spectrum length.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let spectrum = TracedTensor::from_vec_col_major(
///     vec![2],
///     vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)],
/// )
/// .unwrap();
/// let y = spectrum.irfft(Some(2), -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_fft::register_runtime).unwrap();
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
/// ```
fn irfft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype(
        "irfft",
        input.dtype,
        &[DType::C32, DType::C64],
        "C32 or C64",
    )?;
    apply_unary_fft("irfft", input, FftOperation::C2r, n, axis, norm)
}

fn apply_unary_fft(
    op_name: &'static str,
    input: &TracedTensor,
    operation: FftOperation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    let concrete_shape = input.try_concrete_shape();
    let op = Arc::new(prepare_runtime_fft_op(
        op_name,
        operation,
        input.rank,
        concrete_shape.as_deref(),
        n,
        axis,
        norm,
    )?);
    let mut outputs = apply(op, &[input])?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("FFT extension declares exactly one output".into()))
}

fn normalize_axis(op: &'static str, axis: isize, rank: usize) -> Result<usize> {
    if rank == 0 {
        return Err(runtime_invalid_argument(
            op,
            "rank",
            "FFT requires rank >= 1",
        ));
    }
    let normalized = if axis >= 0 {
        axis as usize
    } else {
        rank.checked_sub(axis.unsigned_abs())
            .ok_or_else(|| runtime_axis_out_of_bounds(op, axis.unsigned_abs(), rank))?
    };
    if normalized >= rank {
        return Err(runtime_axis_out_of_bounds(op, normalized, rank));
    }
    Ok(normalized)
}

fn validate_n(op: &'static str, n: Option<usize>) -> Result<()> {
    if n == Some(0) {
        return Err(runtime_invalid_argument(
            op,
            "n",
            "transform length must be positive",
        ));
    }
    Ok(())
}

fn prepare_runtime_fft_op(
    op: &'static str,
    operation: FftOperation,
    rank: usize,
    concrete_shape: Option<&[usize]>,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<FftOp> {
    validate_n(op, n)?;
    let axis = normalize_axis(op, axis, rank)?;
    if n.is_none() && concrete_shape.and_then(|shape| shape.get(axis).copied()) == Some(0) {
        return Err(runtime_invalid_argument(
            op,
            "n",
            "transform length must be positive",
        ));
    }
    if operation == FftOperation::C2r {
        if let Some(shape) = concrete_shape {
            output_shape_c2r(shape, axis, n)?;
        }
    }
    Ok(FftOp::new(operation, axis, n, norm))
}

fn runtime_forward_fft_operation(dtype: DType) -> Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2cForward),
        DType::F32 | DType::F64 => Ok(FftOperation::R2cFull),
        DType::I32 | DType::I64 | DType::Bool => Err(runtime_unsupported_dtype(
            "fft",
            dtype,
            "F32, F64, C32, or C64",
        )),
    }
}

fn require_runtime_dtype(
    op: &'static str,
    dtype: DType,
    supported: &[DType],
    expected: &'static str,
) -> Result<()> {
    if supported.contains(&dtype) {
        Ok(())
    } else {
        Err(runtime_unsupported_dtype(op, dtype, expected))
    }
}

fn runtime_invalid_argument(
    op: &'static str,
    argument: &'static str,
    message: impl Into<String>,
) -> Error {
    Error::validation(
        op,
        ErrorPhase::GraphBuild,
        ValidationError::InvalidArgument {
            argument,
            message: message.into(),
        },
    )
}

fn runtime_axis_out_of_bounds(op: &'static str, axis: usize, rank: usize) -> Error {
    Error::validation(
        op,
        ErrorPhase::GraphBuild,
        ValidationError::AxisOutOfBounds { axis, rank },
    )
}

fn runtime_unsupported_dtype(op: &'static str, dtype: DType, expected: &'static str) -> Error {
    Error::extension(
        op,
        ErrorPhase::GraphBuild,
        FFT_EXTENSION_FAMILY_ID,
        ErrorKind::Unsupported,
        FftError::UnsupportedDType {
            op,
            dtype,
            expected,
        },
    )
}

fn transform_len_dim(n: Option<usize>, input_dim: &SymDim) -> SymDim {
    n.map(SymDim::from).unwrap_or_else(|| input_dim.clone())
}

fn expected_dtype_description(operation: FftOperation) -> &'static str {
    match operation {
        FftOperation::C2cForward | FftOperation::C2cInverse | FftOperation::C2r => "C32 or C64",
        FftOperation::R2cFull | FftOperation::R2cOnesided => "F32 or F64",
    }
}

fn fft_op_name(operation: FftOperation) -> &'static str {
    match operation {
        FftOperation::C2cForward => "fft",
        FftOperation::C2cInverse => "ifft",
        FftOperation::R2cFull | FftOperation::R2cOnesided => "rfft",
        FftOperation::C2r => "irfft",
    }
}

#[cfg(feature = "autodiff")]
fn fft_ad_family_id(operation: FftOperation) -> &'static str {
    match operation {
        FftOperation::C2cForward | FftOperation::C2cInverse => FFT_EXTENSION_FAMILY_ID,
        FftOperation::R2cFull | FftOperation::R2cOnesided => "tenferro-fft.rfft.v1",
        FftOperation::C2r => "tenferro-fft.irfft.v1",
    }
}

#[cfg(feature = "autodiff")]
fn fft_payload(op: &dyn ExtensionOp, rule: ADRuleKind) -> ADRuleResult<&FftOp> {
    op.as_any()
        .downcast_ref::<FftOp>()
        .ok_or_else(|| ADRuleError::unsupported(FFT_EXTENSION_FAMILY_ID, rule))
}

fn output_shape_c2c(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
) -> tenferro_tensor::Result<Vec<usize>> {
    let len = transform_len(shape, axis, n)?;
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len;
    Ok(out_shape)
}

fn output_shape_r2c(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
    onesided: bool,
) -> tenferro_tensor::Result<Vec<usize>> {
    let len = transform_len(shape, axis, n)?;
    let mut out_shape = shape.to_vec();
    out_shape[axis] = if onesided { len / 2 + 1 } else { len };
    Ok(out_shape)
}

fn output_shape_c2r(
    shape: &[usize],
    axis: usize,
    n: Option<usize>,
) -> tenferro_tensor::Result<Vec<usize>> {
    validate_axis("irfft", shape, axis)?;
    let input_len = shape[axis];
    let len = match n {
        Some(len) => len,
        None => default_c2r_output_len(input_len)?,
    };
    if len == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "irfft",
            "output length",
            "must be positive",
        ));
    }
    validate_c2r_spectrum_len(input_len, len)?;
    let mut out_shape = shape.to_vec();
    out_shape[axis] = len;
    Ok(out_shape)
}

fn output_dim_c2r(input_dim: &SymDim, n: Option<usize>) -> tenferro_tensor::Result<SymDim> {
    match (input_dim.constant_value(), n) {
        (Some(input_len), Some(output_len)) => {
            if output_len == 0 {
                return Err(tenferro_tensor::Error::invalid_argument(
                    "irfft",
                    "output length",
                    "must be positive",
                ));
            }
            validate_c2r_spectrum_len(input_len, output_len)?;
            Ok(SymDim::from(output_len))
        }
        (Some(input_len), None) => Ok(SymDim::from(default_c2r_output_len(input_len)?)),
        (None, Some(output_len)) => {
            if output_len == 0 {
                return Err(tenferro_tensor::Error::invalid_argument(
                    "irfft",
                    "output length",
                    "must be positive",
                ));
            }
            Ok(SymDim::from(output_len))
        }
        (None, None) => Ok((input_dim.clone() - 1usize) * 2usize),
    }
}

fn default_c2r_output_len(input_len: usize) -> tenferro_tensor::Result<usize> {
    if input_len == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "irfft",
            "input spectrum axis length",
            "must be positive",
        ));
    }
    input_len
        .checked_sub(1)
        .and_then(|len| len.checked_mul(2))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                "irfft",
                "default output length",
                "overflows usize",
            )
        })
}

fn validate_c2r_spectrum_len(
    input_len: usize,
    output_len: usize,
) -> tenferro_tensor::Result<usize> {
    let expected = output_len / 2 + 1;
    if input_len != expected {
        return Err(tenferro_tensor::Error::invalid_argument(
            "irfft",
            "spectrum",
            format!(
                "one-sided spectrum axis length mismatch: expected {expected} for output length {output_len}, got {input_len}"
            ),
        ));
    }
    Ok(expected)
}

fn transform_len(shape: &[usize], axis: usize, n: Option<usize>) -> tenferro_tensor::Result<usize> {
    validate_axis("fft", shape, axis)?;
    let len = n.unwrap_or(shape[axis]);
    if len == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "fft",
            "transform length",
            "must be positive",
        ));
    }
    Ok(len)
}

fn validate_axis(op: &'static str, shape: &[usize], axis: usize) -> tenferro_tensor::Result<()> {
    if axis >= shape.len() {
        return Err(tenferro_tensor::Error::axis_out_of_bounds(
            op,
            axis,
            shape.len(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod concrete_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_infer_output_meta_rejects_invalid_trait_inputs_without_panicking() {
        let op = FftOp::new(FftOperation::R2cOnesided, 0, None, FftNorm::Backward);
        let shape = [SymDim::from(4usize)];

        assert!(
            tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[], &[&shape]).is_err()
        );
        assert!(
            tenferro_ops::ext_op::invoke_extension_shape_inference(&op, &[DType::F64], &[])
                .is_err()
        );
        assert!(tenferro_ops::ext_op::invoke_extension_shape_inference(
            &op,
            &[DType::I64],
            &[&shape]
        )
        .is_err());

        let bad_axis = FftOp::new(FftOperation::C2cForward, 2, None, FftNorm::Backward);
        assert!(tenferro_ops::ext_op::invoke_extension_shape_inference(
            &bad_axis,
            &[DType::C64],
            &[&shape]
        )
        .is_err());
    }

    #[test]
    fn checked_shape_product_rejects_overflow_before_allocation() {
        let err = cpu::checked_shape_product("fft", "output", &[usize::MAX, 2])
            .expect_err("overflowing output shape should be rejected");

        assert!(err.to_string().contains("overflows usize"), "{err}");
    }

    #[test]
    fn irfft_default_output_length_rejects_overflow() {
        let err = output_shape_c2r(&[usize::MAX], 0, None)
            .expect_err("default irfft output length should reject overflow");

        assert!(err.to_string().contains("overflows usize"), "{err}");
    }

    #[test]
    fn normalize_axis_handles_large_rank_without_isize_cast_wrap() {
        assert_eq!(normalize_axis("fft", 0, usize::MAX).unwrap(), 0);
        assert_eq!(
            normalize_axis("fft", -1, usize::MAX).unwrap(),
            usize::MAX - 1
        );
        assert!(normalize_axis("fft", isize::MIN, 3).is_err());
    }

    #[test]
    fn axis_lane_layout_rejects_stride_overflow() {
        let err = cpu::for_axis_lane(&[usize::MAX, 2], 1, 2, |_| Ok(()))
            .expect_err("lane layout should reject stride overflow");

        assert!(err.to_string().contains("overflows usize"), "{err}");
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn fft_transpose_rule_respects_inactive_linearized_input() {
        let rule = FftAdRule;
        let op = FftOp::new(FftOperation::C2cForward, 0, None, FftNorm::Backward);
        let mut builder = computegraph::graph::GraphBuilder::<StdTensorOp>::new();
        let cotangent = builder.add_input(tenferro_ops::input_key::TensorInputKey::User { id: 0 });
        let result = rule
            .linear_transpose(
                &op,
                &mut builder,
                &[Some(cotangent)],
                &[],
                &[false],
                &mut ShapeGuardContext::default(),
            )
            .unwrap();

        assert_eq!(result, vec![None]);
        assert!(builder.build().operations().is_empty());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn fft_transpose_rule_uses_metadata_for_linear_only_matching_length() {
        let rule = FftAdRule;
        let op = FftOp::new(FftOperation::C2cForward, 0, Some(4), FftNorm::Backward);
        let active_key = ValueKey::Input(tenferro_ops::input_key::TensorInputKey::User { id: 1 });
        let mut ctx = ShapeGuardContext::default();
        ctx.insert_metadata(
            active_key.clone(),
            tenferro_ops::TensorMeta::exact(DType::C64, vec![SymDim::from(4usize)]),
        );

        let mut builder = computegraph::graph::GraphBuilder::<StdTensorOp>::new();
        let cotangent = builder.add_input(tenferro_ops::input_key::TensorInputKey::User { id: 0 });
        let result = rule
            .linear_transpose(
                &op,
                &mut builder,
                &[Some(cotangent)],
                &[tidu::PrimitiveTransposeInput::Linear {
                    key: active_key.clone(),
                    primal: None,
                }],
                &[true],
                &mut ctx,
            )
            .unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());
        let active_ref = ValueRef::External(active_key);
        let graph = builder.build();
        assert!(graph
            .operations()
            .iter()
            .all(|node| !node.inputs.iter().any(|input| input == &active_ref)));
        assert!(graph.operations().iter().all(|node| {
            !matches!(
                node.operation,
                StdTensorOp::ShapeOf { .. }
                    | StdTensorOp::DynamicTruncate { .. }
                    | StdTensorOp::PadToMatch { .. }
            )
        }));
    }
}
