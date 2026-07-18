//! FFT extension operations for tenferro.
//!
//! This crate is an out-of-tree `ExtensionOp` package. The initial
//! implementation executes on host tensors through `rustfft`; it does not add
//! FFT to the core `tenferro` backend trait surface. Concrete non-AD execution
//! uses [`TensorFftExt`] and [`TensorReadFftExt`]. Traced graph construction
//! uses [`TracedTensorFftExt`].
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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use lru::LruCache;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive, Zero};
use rustfft::{Fft, FftNum, FftPlanner};
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
use tenferro_runtime::{
    Error, ErrorPhase, ExtensionCacheKey, ExtensionCacheSelector, ExtensionCacheStore, Result,
    TracedTensor,
};
use tenferro_tensor::{
    CacheStats, DType, DeviceKind, ErrorKind, MemoryKind, Placement, RuntimeCacheControl, Tensor,
    TensorBackend, TensorRead, TypedTensor, ValidationError,
};
#[cfg(feature = "autodiff")]
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

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

/// Runtime cache namespace used for RustFFT plans.
pub const FFT_PLAN_CACHE_NAME: &str = "rustfft-plans";

/// Default number of plans retained by a caller-owned [`FftPlanCache`].
pub const DEFAULT_FFT_PLAN_CACHE_CAPACITY: usize = 64;

/// Select the FFT plan entries in an extension runtime cache.
pub const fn fft_plan_cache_selector() -> ExtensionCacheSelector {
    ExtensionCacheSelector::Cache {
        family_id: FFT_EXTENSION_FAMILY_ID,
        cache_name: FFT_PLAN_CACHE_NAME,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum FftPlanDType {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FftPlanKey {
    len: usize,
    forward: bool,
    dtype: FftPlanDType,
}

enum CachedFftPlan {
    F32(Arc<dyn Fft<f32>>),
    F64(Arc<dyn Fft<f64>>),
}

/// Bounded, caller-owned LRU cache of RustFFT plans.
///
/// Retained-byte statistics include the cache-owned key and `Arc` handle for
/// each entry. RustFFT does not expose the allocations owned by an opaque plan,
/// so those allocations are intentionally excluded from the estimate.
pub struct FftPlanCache {
    entries: LruCache<FftPlanKey, CachedFftPlan>,
}

impl FftPlanCache {
    /// Create an empty plan cache with an explicit maximum entry count.
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            entries: LruCache::new(capacity),
        }
    }

    /// Maximum number of retained plans.
    pub fn capacity(&self) -> NonZeroUsize {
        self.entries.cap()
    }

    /// Resize the cache, evicting least-recently-used plans when necessary.
    pub fn set_capacity(&mut self, capacity: NonZeroUsize) {
        self.entries.resize(capacity);
    }

    /// Remove every retained plan.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Snapshot the number of plans and known cache-owned bytes retained.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.entries.len().saturating_mul(fft_plan_retained_bytes()),
        }
    }

    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F32,
        };
        if let Some(CachedFftPlan::F32(plan)) = self.entries.get(&key) {
            return Arc::clone(plan);
        }
        let plan = build_fft_plan::<f32>(len, forward);
        self.entries.put(key, CachedFftPlan::F32(Arc::clone(&plan)));
        plan
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        };
        if let Some(CachedFftPlan::F64(plan)) = self.entries.get(&key) {
            return Arc::clone(plan);
        }
        let plan = build_fft_plan::<f64>(len, forward);
        self.entries.put(key, CachedFftPlan::F64(Arc::clone(&plan)));
        plan
    }

    #[cfg(test)]
    fn contains_f64(&self, len: usize, forward: bool) -> bool {
        self.entries.contains(&FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        })
    }
}

impl Default for FftPlanCache {
    fn default() -> Self {
        Self::with_capacity(
            NonZeroUsize::new(DEFAULT_FFT_PLAN_CACHE_CAPACITY).unwrap_or(NonZeroUsize::MIN),
        )
    }
}

impl RuntimeCacheControl for FftPlanCache {
    fn clear(&mut self) {
        Self::clear(self);
    }

    fn stats(&self) -> CacheStats {
        Self::stats(self)
    }
}

/// Reusable concrete FFT executor with explicitly owned plan state.
#[derive(Default)]
pub struct FftExecutor {
    plans: FftPlanCache,
}

impl FftExecutor {
    /// Create an executor from a caller-configured plan cache.
    pub fn new(plans: FftPlanCache) -> Self {
        Self { plans }
    }

    /// Inspect the owned plan cache.
    pub const fn plan_cache(&self) -> &FftPlanCache {
        &self.plans
    }

    /// Mutably inspect or configure the owned plan cache.
    pub fn plan_cache_mut(&mut self) -> &mut FftPlanCache {
        &mut self.plans
    }

    /// Snapshot the owned plan cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.plans.stats()
    }

    /// Remove every retained plan from this executor.
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
    pub fn fft<B: TensorBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_fft_kind("FftExecutor::fft", input.dtype())?,
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
    pub fn ifft<B: TensorBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_ifft_kind("FftExecutor::ifft", input.dtype())?,
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
    pub fn rfft<B: TensorBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_rfft_kind("FftExecutor::rfft", input.dtype())?,
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
    pub fn irfft<B: TensorBackend>(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_irfft_kind("FftExecutor::irfft", input.dtype())?,
            "FftExecutor::irfft",
            n,
            axis,
            norm,
            backend,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute<B: TensorBackend>(
        &mut self,
        input: &Tensor,
        kind: FftKind,
        op_name: &'static str,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let op = concrete_fft_op(op_name, kind, input.shape(), n, axis, norm)?;
        execute_concrete_fft_op_with_plans(input, &op, backend, &mut self.plans)
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
    fn fft<B: TensorBackend>(
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
    fn ifft<B: TensorBackend>(
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
    fn rfft<B: TensorBackend>(
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
    fn irfft<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorFftExt for Tensor {
    fn fft<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let op = concrete_fft_op(
            "TensorFftExt::fft",
            concrete_fft_kind("TensorFftExt::fft", self.dtype())?,
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &op, backend)
    }

    fn ifft<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let op = concrete_fft_op(
            "TensorFftExt::ifft",
            concrete_ifft_kind("TensorFftExt::ifft", self.dtype())?,
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &op, backend)
    }

    fn rfft<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let op = concrete_fft_op(
            "TensorFftExt::rfft",
            concrete_rfft_kind("TensorFftExt::rfft", self.dtype())?,
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &op, backend)
    }

    fn irfft<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        let op = concrete_fft_op(
            "TensorFftExt::irfft",
            concrete_irfft_kind("TensorFftExt::irfft", self.dtype())?,
            self.shape(),
            n,
            axis,
            norm,
        )?;
        execute_concrete_fft_op(self, &op, backend)
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
    fn fft_read<B: TensorBackend>(
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
    fn ifft_read<B: TensorBackend>(
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
    fn rfft_read<B: TensorBackend>(
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
    fn irfft_read<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorReadFftExt for TensorRead<'_> {
    fn fft_read<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_fft_kind("TensorReadFftExt::fft_read", self.dtype())?,
            "TensorReadFftExt::fft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn ifft_read<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_ifft_kind("TensorReadFftExt::ifft_read", self.dtype())?,
            "TensorReadFftExt::ifft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn rfft_read<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_rfft_kind("TensorReadFftExt::rfft_read", self.dtype())?,
            "TensorReadFftExt::rfft_read",
            n,
            axis,
            norm,
            backend,
        )
    }

    fn irfft_read<B: TensorBackend>(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor> {
        execute_concrete_fft_read_op(
            self,
            concrete_irfft_kind("TensorReadFftExt::irfft_read", self.dtype())?,
            "TensorReadFftExt::irfft_read",
            n,
            axis,
            norm,
            backend,
        )
    }
}

/// FFT normalization convention.
///
/// `Backward` matches NumPy, JAX, and PyTorch defaults: the forward transform
/// is unscaled and the inverse transform is scaled by `1 / n`.
///
/// # Examples
///
/// ```
/// use tenferro_fft::FftNorm;
///
/// assert_eq!(FftNorm::default(), FftNorm::Backward);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FftNorm {
    /// Scale inverse transforms by `1 / n`.
    #[default]
    Backward,
    /// Scale forward transforms by `1 / n`.
    Forward,
    /// Scale both forward and inverse transforms by `1 / sqrt(n)`.
    Ortho,
}

#[cfg(feature = "autodiff")]
impl FftNorm {
    fn c2c_adjoint(self) -> Self {
        match self {
            Self::Backward => Self::Forward,
            Self::Forward => Self::Backward,
            Self::Ortho => Self::Ortho,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FftKind {
    C2C { forward: bool },
    R2C { onesided: bool },
    C2R,
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
    kind: FftKind,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
}

impl FftOp {
    fn new(kind: FftKind, axis: usize, n: Option<usize>, norm: FftNorm) -> Self {
        Self {
            kind,
            axis,
            n,
            norm,
        }
    }

    #[cfg(feature = "autodiff")]
    fn c2c_adjoint(&self) -> Option<Self> {
        match self.kind {
            FftKind::C2C { forward } => Some(Self {
                kind: FftKind::C2C { forward: !forward },
                axis: self.axis,
                n: self.n,
                norm: self.norm.c2c_adjoint(),
            }),
            FftKind::R2C { .. } | FftKind::C2R => None,
        }
    }
}

impl ExtensionOp for FftOp {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        let kind = match self.kind {
            FftKind::C2C { forward: true } => 0,
            FftKind::C2C { forward: false } => 1,
            FftKind::R2C { onesided: true } => 2,
            FftKind::R2C { onesided: false } => 3,
            FftKind::C2R => 4,
        };
        hasher.write_u8(kind);
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
        let output_dtype = match self.kind {
            FftKind::C2C { .. } => {
                if !matches!(input_dtype, DType::C32 | DType::C64) {
                    return Err(tensor_unsupported_dtype(
                        "tenferro-fft",
                        input_dtype,
                        "C32 or C64",
                    ));
                }
                input_dtype
            }
            FftKind::R2C { onesided } => {
                let len = transform_len_dim(self.n, &input_shape[self.axis]);
                out_shape[self.axis] = if onesided { len / 2usize + 1usize } else { len };
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
            FftKind::C2R => {
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

        if matches!(self.kind, FftKind::C2C { .. }) {
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
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    execute_host_fft_op_with_plans(op, inputs, &mut plans)
}

fn execute_host_fft_op_with_plans<P: FftPlanProvider + ?Sized>(
    op: &FftOp,
    inputs: &[&Tensor],
    plans: &mut P,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "inputs",
            format!("expected 1 input, got {}", inputs.len()),
        ));
    }
    validate_host_fft_input(fft_op_name(op.kind), inputs[0])?;

    let output = match (op.kind, inputs[0]) {
        (FftKind::C2C { forward }, Tensor::C64(input)) => {
            Tensor::C64(TypedTensor::from_vec_col_major(
                output_shape_c2c(input.shape(), op.axis, op.n)?,
                execute_c2c(input, op.axis, op.n, forward, op.norm, plans)?,
            )?)
        }
        (FftKind::C2C { forward }, Tensor::C32(input)) => {
            Tensor::C32(TypedTensor::from_vec_col_major(
                output_shape_c2c(input.shape(), op.axis, op.n)?,
                execute_c2c(input, op.axis, op.n, forward, op.norm, plans)?,
            )?)
        }
        (FftKind::R2C { onesided }, Tensor::F64(input)) => {
            Tensor::C64(TypedTensor::from_vec_col_major(
                output_shape_r2c(input.shape(), op.axis, op.n, onesided)?,
                execute_r2c(input, op.axis, op.n, onesided, op.norm, plans)?,
            )?)
        }
        (FftKind::R2C { onesided }, Tensor::F32(input)) => {
            Tensor::C32(TypedTensor::from_vec_col_major(
                output_shape_r2c(input.shape(), op.axis, op.n, onesided)?,
                execute_r2c(input, op.axis, op.n, onesided, op.norm, plans)?,
            )?)
        }
        (FftKind::C2R, Tensor::C64(input)) => Tensor::F64(TypedTensor::from_vec_col_major(
            output_shape_c2r(input.shape(), op.axis, op.n)?,
            execute_c2r(input, op.axis, op.n, op.norm, plans)?,
        )?),
        (FftKind::C2R, Tensor::C32(input)) => Tensor::F32(TypedTensor::from_vec_col_major(
            output_shape_c2r(input.shape(), op.axis, op.n)?,
            execute_c2r(input, op.axis, op.n, op.norm, plans)?,
        )?),
        (kind, other) => {
            return Err(tensor_unsupported_dtype(
                fft_op_name(kind),
                other.dtype(),
                expected_dtype_description(kind),
            ));
        }
    };
    Ok(vec![output])
}

fn execute_concrete_fft_op<B: TensorBackend>(
    input: &Tensor,
    op: &FftOp,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|_exec| single_fft_output(execute_host_fft_op(op, &[input])?))
}

fn execute_concrete_fft_op_with_plans<B: TensorBackend, P: FftPlanProvider + ?Sized>(
    input: &Tensor,
    op: &FftOp,
    backend: &mut B,
    plans: &mut P,
) -> tenferro_tensor::Result<Tensor> {
    backend.with_backend_session(|_exec| {
        single_fft_output(execute_host_fft_op_with_plans(op, &[input], plans)?)
    })
}

#[allow(clippy::too_many_arguments)]
fn execute_concrete_fft_read_op<B: TensorBackend>(
    input: &TensorRead<'_>,
    kind: FftKind,
    op_name: &'static str,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    backend: &mut B,
) -> tenferro_tensor::Result<Tensor> {
    let op = concrete_fft_op(op_name, kind, input.shape(), n, axis, norm)?;
    backend.with_backend_session(|exec| {
        let materialized = exec.to_contiguous_read(input.clone())?;
        single_fft_output(execute_host_fft_op(&op, &[&materialized])?)
    })
}

fn single_fft_output(mut outputs: Vec<Tensor>) -> tenferro_tensor::Result<Tensor> {
    if outputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "outputs",
            format!("expected 1 FFT output, got {}", outputs.len()),
        ));
    }
    Ok(outputs.remove(0))
}

fn concrete_fft_op(
    op: &'static str,
    kind: FftKind,
    input_shape: &[usize],
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> tenferro_tensor::Result<FftOp> {
    validate_concrete_n(op, n)?;
    let axis = normalize_concrete_axis(op, axis, input_shape.len())?;
    validate_concrete_transform_len(op, input_shape, n, axis)?;
    if matches!(kind, FftKind::C2R) {
        output_shape_c2r(input_shape, axis, n)?;
    }
    Ok(FftOp::new(kind, axis, n, norm))
}

fn concrete_fft_kind(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftKind> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftKind::C2C { forward: true }),
        DType::F32 | DType::F64 => Ok(FftKind::R2C { onesided: false }),
        DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "F32, F64, C32, or C64"))
        }
    }
}

fn concrete_ifft_kind(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftKind> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftKind::C2C { forward: false }),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "C32 or C64"))
        }
    }
}

fn concrete_rfft_kind(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftKind> {
    match dtype {
        DType::F32 | DType::F64 => Ok(FftKind::R2C { onesided: true }),
        DType::C32 | DType::C64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "F32 or F64"))
        }
    }
}

fn concrete_irfft_kind(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftKind> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftKind::C2R),
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => {
            Err(tensor_unsupported_dtype(op, dtype, "C32 or C64"))
        }
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

fn tensor_placement(input: &Tensor) -> &Placement {
    input.placement()
}

fn tensor_has_backend_buffer(input: &Tensor) -> bool {
    input.is_backend_buffer()
}

fn validate_host_fft_input(op: &'static str, input: &Tensor) -> tenferro_tensor::Result<()> {
    let placement = tensor_placement(input);
    let is_device = matches!(placement.memory_kind, MemoryKind::Device);
    if !is_device && !tensor_has_backend_buffer(input) {
        return Ok(());
    }

    let location = match placement.device.as_ref().map(|device| &device.kind) {
        Some(DeviceKind::Gpu(kind)) => format!("GPU backend {kind:?}"),
        Some(kind) => format!("device kind {kind:?}"),
        None if is_device => "device tensor without device metadata".to_string(),
        None => "backend buffer".to_string(),
    };
    Err(tenferro_tensor::Error::unsupported(
        op,
        format!(
            "tenferro-fft supports host tensors only; unsupported {location} input; \
             download the tensor to CPU before FFT"
        ),
    ))
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
        if !matches!(fft_op.kind, FftKind::C2C { .. }) {
            return Err(ADRuleError::unsupported(
                fft_ad_family_id(fft_op.kind),
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
    if !matches!(fft_op.kind, FftKind::C2C { .. }) {
        return Err(ADRuleError::unsupported(
            fft_ad_family_id(fft_op.kind),
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

fn execute_fft_extension<B: TensorBackend + 'static>(
    op: &FftOp,
    inputs: &[&Tensor],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let mut plans = ExtensionFftPlanCache::new(ctx.caches_mut());
    execute_host_fft_op_with_plans(op, inputs, &mut plans)
}

fn execute_fft_extension_reads<B: TensorBackend + 'static>(
    op: &FftOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    // rustfft consumes compact host tensors; materialization is explicit so
    // backend-backed views produce a normal error instead of an implicit path.
    let materialized_inputs = ctx.backend_mut().with_backend_session(|exec| {
        inputs
            .iter()
            .cloned()
            .map(|input| exec.to_contiguous_read(input))
            .collect::<tenferro_tensor::Result<Vec<_>>>()
    })?;
    let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
    let mut plans = ExtensionFftPlanCache::new(ctx.caches_mut());
    execute_host_fft_op_with_plans(op, &input_refs, &mut plans)
}

define_extension_runtime! {
    runtime = FftRuntime,
    family_id = FFT_EXTENSION_FAMILY_ID,
    op_type = FftOp,
    execute = execute_fft_extension,
    execute_reads = execute_fft_extension_reads,
    register_fn = register_runtime,
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
    let kind = match input.dtype {
        DType::C32 | DType::C64 => FftKind::C2C { forward: true },
        DType::F32 | DType::F64 => FftKind::R2C { onesided: false },
        DType::I32 | DType::I64 | DType::Bool => {
            return Err(runtime_unsupported_dtype(
                "fft",
                input.dtype,
                "F32, F64, C32, or C64",
            ))
        }
    };
    apply_unary_fft("fft", input, kind, n, axis, norm)
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
    if !matches!(input.dtype, DType::C32 | DType::C64) {
        return Err(runtime_unsupported_dtype("ifft", input.dtype, "C32 or C64"));
    }
    apply_unary_fft(
        "ifft",
        input,
        FftKind::C2C { forward: false },
        n,
        axis,
        norm,
    )
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
    if !matches!(input.dtype, DType::F32 | DType::F64) {
        return Err(runtime_unsupported_dtype("rfft", input.dtype, "F32 or F64"));
    }
    apply_unary_fft(
        "rfft",
        input,
        FftKind::R2C { onesided: true },
        n,
        axis,
        norm,
    )
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
    if !matches!(input.dtype, DType::C32 | DType::C64) {
        return Err(runtime_unsupported_dtype(
            "irfft",
            input.dtype,
            "C32 or C64",
        ));
    }
    apply_unary_fft("irfft", input, FftKind::C2R, n, axis, norm)
}

fn apply_unary_fft(
    op_name: &'static str,
    input: &TracedTensor,
    kind: FftKind,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    validate_n(op_name, n)?;
    let axis = normalize_axis(op_name, axis, input.rank)?;
    validate_resolved_transform_len(op_name, input, n, axis)?;
    if matches!(kind, FftKind::C2R) {
        if let Some(shape) = input.try_concrete_shape() {
            output_shape_c2r(&shape, axis, n)?;
        }
    }
    let op = Arc::new(FftOp::new(kind, axis, n, norm));
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

fn validate_resolved_transform_len(
    op: &'static str,
    input: &TracedTensor,
    n: Option<usize>,
    axis: usize,
) -> Result<()> {
    if n.is_some() {
        return Ok(());
    }
    if input
        .try_concrete_shape()
        .and_then(|shape| shape.get(axis).copied())
        == Some(0)
    {
        return Err(runtime_invalid_argument(
            op,
            "n",
            "transform length must be positive",
        ));
    }
    Ok(())
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

fn expected_dtype_description(kind: FftKind) -> &'static str {
    match kind {
        FftKind::C2C { .. } | FftKind::C2R => "C32 or C64",
        FftKind::R2C { .. } => "F32 or F64",
    }
}

fn fft_op_name(kind: FftKind) -> &'static str {
    match kind {
        FftKind::C2C { forward: true } => "fft",
        FftKind::C2C { forward: false } => "ifft",
        FftKind::R2C { .. } => "rfft",
        FftKind::C2R => "irfft",
    }
}

#[cfg(feature = "autodiff")]
fn fft_ad_family_id(kind: FftKind) -> &'static str {
    match kind {
        FftKind::C2C { .. } => FFT_EXTENSION_FAMILY_ID,
        FftKind::R2C { .. } => "tenferro-fft.rfft.v1",
        FftKind::C2R => "tenferro-fft.irfft.v1",
    }
}

#[cfg(feature = "autodiff")]
fn fft_payload<'a>(op: &'a dyn ExtensionOp, rule: ADRuleKind) -> ADRuleResult<&'a FftOp> {
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

fn checked_shape_product(
    op: &'static str,
    role: &'static str,
    shape: &[usize],
) -> tenferro_tensor::Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            tenferro_tensor::Error::invalid_argument(
                op,
                "shape product",
                format!("{role} shape product overflows usize"),
            )
        })
}

fn checked_mul(
    op: &'static str,
    role: &'static str,
    lhs: usize,
    rhs: usize,
) -> tenferro_tensor::Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "arithmetic",
            format!("{role} overflows usize"),
        )
    })
}

fn checked_add(
    op: &'static str,
    role: &'static str,
    lhs: usize,
    rhs: usize,
) -> tenferro_tensor::Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            op,
            "arithmetic",
            format!("{role} overflows usize"),
        )
    })
}

fn uninit_output_vec<T>(len: usize) -> Vec<MaybeUninit<T>> {
    let mut output = Vec::with_capacity(len);
    // SAFETY: Uninitialized bytes are valid for `MaybeUninit<T>` slots. The
    // slots are converted to `T` only after all output positions are written.
    unsafe { output.set_len(len) };
    output
}

unsafe fn assume_init_output_vec<T>(mut output: Vec<MaybeUninit<T>>) -> Vec<T> {
    let len = output.len();
    let capacity = output.capacity();
    let ptr = output.as_mut_ptr().cast::<T>();
    std::mem::forget(output);
    // SAFETY: `MaybeUninit<T>` has the same layout as `T`; the caller
    // guarantees every slot has been initialized exactly once.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

fn build_fft_plan<T: FftNum + 'static>(len: usize, forward: bool) -> Arc<dyn Fft<T>> {
    let mut planner = FftPlanner::<T>::new();
    if forward {
        planner.plan_fft_forward(len)
    } else {
        planner.plan_fft_inverse(len)
    }
}

const fn fft_plan_retained_bytes() -> usize {
    std::mem::size_of::<FftPlanKey>() + std::mem::size_of::<CachedFftPlan>()
}

trait FftPlanProvider: Send {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>>;
    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>>;
}

impl FftPlanProvider for FftPlanCache {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        Self::plan_f32(self, len, forward)
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        Self::plan_f64(self, len, forward)
    }
}

trait CachedFftPlanScalar: FftNum + Float + FromPrimitive + 'static {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>>;
}

impl CachedFftPlanScalar for f32 {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>> {
        plans.plan_f32(len, forward)
    }
}

impl CachedFftPlanScalar for f64 {
    fn plan<P: FftPlanProvider + ?Sized>(
        plans: &mut P,
        len: usize,
        forward: bool,
    ) -> Arc<dyn Fft<Self>> {
        plans.plan_f64(len, forward)
    }
}

fn cached_fft_plan<T: CachedFftPlanScalar, P: FftPlanProvider + ?Sized>(
    plans: &mut P,
    len: usize,
    forward: bool,
) -> Arc<dyn Fft<T>> {
    T::plan(plans, len, forward)
}

#[derive(Clone)]
struct ExtensionF32Plan {
    key: FftPlanKey,
    plan: Arc<dyn Fft<f32>>,
}

#[derive(Clone)]
struct ExtensionF64Plan {
    key: FftPlanKey,
    plan: Arc<dyn Fft<f64>>,
}

struct ExtensionFftPlanCache<'a> {
    entries: &'a mut ExtensionCacheStore,
}

impl<'a> ExtensionFftPlanCache<'a> {
    fn new(entries: &'a mut ExtensionCacheStore) -> Self {
        Self { entries }
    }
}

fn extension_plan_key(key: FftPlanKey) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    ExtensionCacheKey::new(
        FFT_EXTENSION_FAMILY_ID,
        FFT_PLAN_CACHE_NAME,
        hasher.finish(),
    )
}

impl FftPlanProvider for ExtensionFftPlanCache<'_> {
    fn plan_f32(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f32>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F32,
        };
        let cache_key = extension_plan_key(key);
        if let Some(cached) = self.entries.get::<ExtensionF32Plan>(&cache_key) {
            if cached.key == key {
                return Arc::clone(&cached.plan);
            }
        }
        let plan = build_fft_plan::<f32>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionF32Plan {
                key,
                plan: Arc::clone(&plan),
            },
            fft_plan_retained_bytes(),
        );
        plan
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> Arc<dyn Fft<f64>> {
        let key = FftPlanKey {
            len,
            forward,
            dtype: FftPlanDType::F64,
        };
        let cache_key = extension_plan_key(key);
        if let Some(cached) = self.entries.get::<ExtensionF64Plan>(&cache_key) {
            if cached.key == key {
                return Arc::clone(&cached.plan);
            }
        }
        let plan = build_fft_plan::<f64>(len, forward);
        self.entries.put(
            cache_key,
            ExtensionF64Plan {
                key,
                plan: Arc::clone(&plan),
            },
            fft_plan_retained_bytes(),
        );
        plan
    }
}

fn execute_c2c<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    forward: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar,
{
    let in_shape = input.shape();
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_c2c(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let input_data = input.host_data()?;
    let output_len = checked_shape_product("fft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, fft_len, forward);
    let scale: T = scale_for(norm, forward, fft_len)?;
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill is transform padding semantics when the input
        // lane is shorter than `fft_len`; it is not redundant initialization.
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (slot, offset) in lane
            .iter_mut()
            .take(copy_len)
            .zip(lane_ctx.input_offsets(copy_len))
        {
            *slot = input_data[offset];
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .copied()
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn execute_r2c<T>(
    input: &TypedTensor<T>,
    axis: usize,
    n: Option<usize>,
    onesided: bool,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<Complex<T>>>
where
    T: CachedFftPlanScalar,
{
    let in_shape = input.shape();
    let fft_len = transform_len(in_shape, axis, n)?;
    let out_shape = output_shape_r2c(in_shape, axis, n, onesided)?;
    let out_axis_len = out_shape[axis];
    let input_data = input.host_data()?;
    let output_len = checked_shape_product("rfft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, fft_len, true);
    let scale: T = scale_for(norm, true, fft_len)?;
    let mut lane = vec![Complex::zero(); fft_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill is rfft padding semantics when the real input
        // lane is shorter than `fft_len`; later writes cover only `copy_len`.
        lane.fill(Complex::zero());
        let copy_len = lane_ctx.in_axis_len.min(fft_len);
        for (slot, offset) in lane
            .iter_mut()
            .take(copy_len)
            .zip(lane_ctx.input_offsets(copy_len))
        {
            *slot = Complex::new(input_data[offset], T::zero());
        }
        fft_plan.process(&mut lane);
        if scale != T::one() {
            for value in &mut lane {
                *value = *value * scale;
            }
        }
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .copied()
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn execute_c2r<T>(
    input: &TypedTensor<Complex<T>>,
    axis: usize,
    n: Option<usize>,
    norm: FftNorm,
    plans: &mut (impl FftPlanProvider + ?Sized),
) -> tenferro_tensor::Result<Vec<T>>
where
    T: CachedFftPlanScalar,
{
    let in_shape = input.shape();
    let out_shape = output_shape_c2r(in_shape, axis, n)?;
    let out_axis_len = out_shape[axis];
    let expected_half = validate_c2r_spectrum_len(in_shape[axis], out_axis_len)?;
    let input_data = input.host_data()?;
    let output_len = checked_shape_product("irfft", "output", &out_shape)?;
    let mut output = uninit_output_vec(output_len);
    let fft_plan = cached_fft_plan::<T, _>(plans, out_axis_len, false);
    let scale: T = scale_for(norm, false, out_axis_len)?;
    let mut lane = vec![Complex::zero(); out_axis_len];

    for_axis_lane(in_shape, axis, out_axis_len, |lane_ctx| {
        // INVARIANT: zero-fill clears the inverse lane before writing the
        // one-sided spectrum and mirrored tail for this lane.
        lane.fill(Complex::zero());
        for (slot, offset) in lane
            .iter_mut()
            .take(expected_half)
            .zip(lane_ctx.input_offsets(expected_half))
        {
            *slot = input_data[offset];
        }
        for k in expected_half..out_axis_len {
            let mirror = out_axis_len - k;
            if mirror < lane.len() {
                lane[k] = lane[mirror].conj();
            }
        }
        fft_plan.process(&mut lane);
        for (value, offset) in lane
            .iter()
            .take(out_axis_len)
            .zip(lane_ctx.output_offsets(out_axis_len))
        {
            output[offset].write(value.re * scale);
        }
        Ok(())
    })?;

    // SAFETY: `for_axis_lane` covers every element in the compact column-major
    // output exactly once, and each lane writes all `out_axis_len` positions.
    Ok(unsafe { assume_init_output_vec(output) })
}

fn scale_for<T>(norm: FftNorm, forward: bool, n: usize) -> tenferro_tensor::Result<T>
where
    T: Float + FromPrimitive,
{
    let len = T::from_usize(n).ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "tenferro_fft::scale_for",
            "FFT length",
            format!("{n} cannot be represented as scalar"),
        )
    })?;
    Ok(match (norm, forward) {
        (FftNorm::Backward, true) | (FftNorm::Forward, false) => T::one(),
        (FftNorm::Backward, false) | (FftNorm::Forward, true) => T::one() / len,
        (FftNorm::Ortho, _) => T::one() / len.sqrt(),
    })
}

#[derive(Clone, Copy)]
struct LaneContext {
    input_base: usize,
    output_base: usize,
    axis_stride: usize,
    in_axis_len: usize,
    out_axis_len: usize,
}

impl LaneContext {
    fn input_offsets(self, count: usize) -> impl Iterator<Item = usize> {
        debug_assert!(count <= self.in_axis_len);
        lane_offsets(self.input_base, self.axis_stride, count)
    }

    fn output_offsets(self, count: usize) -> impl Iterator<Item = usize> {
        debug_assert!(count <= self.out_axis_len);
        lane_offsets(self.output_base, self.axis_stride, count)
    }
}

fn lane_offsets(base: usize, stride: usize, count: usize) -> impl Iterator<Item = usize> {
    // INVARIANT: `for_axis_lane` checks input/output lane coverage before it
    // constructs any `LaneContext`, so every `base + k * stride` for
    // `k < count` stays within the compact column-major buffer.
    (0..count).map(move |k| base + k * stride)
}

fn for_axis_lane(
    in_shape: &[usize],
    axis: usize,
    out_axis_len: usize,
    mut f: impl FnMut(LaneContext) -> tenferro_tensor::Result<()>,
) -> tenferro_tensor::Result<()> {
    let in_axis_len = in_shape[axis];
    let axis_stride = checked_shape_product("fft", "axis stride", &in_shape[..axis])?;
    let outer = checked_shape_product("fft", "outer lane count", &in_shape[axis + 1..])?;
    let in_block = checked_mul("fft", "input lane block", axis_stride, in_axis_len)?;
    let out_block = checked_mul("fft", "output lane block", axis_stride, out_axis_len)?;
    let _input_len = checked_mul("fft", "input lane coverage", outer, in_block)?;
    let _output_len = checked_mul("fft", "output lane coverage", outer, out_block)?;

    // INVARIANT: lanes are processed sequentially so one scratch lane can be
    // reused while writing into a single `MaybeUninit` output buffer. Parallel
    // lane execution needs disjoint output splitting plus per-worker scratch.
    for outer_idx in 0..outer {
        let in_outer_base = checked_mul("fft", "input outer base", outer_idx, in_block)?;
        let out_outer_base = checked_mul("fft", "output outer base", outer_idx, out_block)?;
        for inner in 0..axis_stride {
            let input_base = checked_add("fft", "input lane base", in_outer_base, inner)?;
            let output_base = checked_add("fft", "output lane base", out_outer_base, inner)?;
            f(LaneContext {
                input_base,
                output_base,
                axis_stride,
                in_axis_len,
                out_axis_len,
            })?;
        }
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
        let op = FftOp::new(FftKind::R2C { onesided: true }, 0, None, FftNorm::Backward);
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

        let bad_axis = FftOp::new(FftKind::C2C { forward: true }, 2, None, FftNorm::Backward);
        assert!(tenferro_ops::ext_op::invoke_extension_shape_inference(
            &bad_axis,
            &[DType::C64],
            &[&shape]
        )
        .is_err());
    }

    #[test]
    fn checked_shape_product_rejects_overflow_before_allocation() {
        let err = checked_shape_product("fft", "output", &[usize::MAX, 2])
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
        let err = for_axis_lane(&[usize::MAX, 2], 1, 2, |_| Ok(()))
            .expect_err("lane layout should reject stride overflow");

        assert!(err.to_string().contains("overflows usize"), "{err}");
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn fft_transpose_rule_respects_inactive_linearized_input() {
        let rule = FftAdRule;
        let op = FftOp::new(FftKind::C2C { forward: true }, 0, None, FftNorm::Backward);
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
        let op = FftOp::new(
            FftKind::C2C { forward: true },
            0,
            Some(4),
            FftNorm::Backward,
        );
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
