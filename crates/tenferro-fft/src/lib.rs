//! FFT extension operations for tenferro.
//!
//! This crate is an out-of-tree `ExtensionOp` package with an explicit
//! [`FftBackend`] capability. [`tenferro_cpu::CpuEngine`] implements the
//! capability through RustFFT. With the `cuda` feature,
//! `tenferro_gpu::cuda::CudaEngine` executes the
//! supported one-dimensional F32/F64/C32/C64 operations through dynamically
//! loaded cuFFT without implicit transfers or CPU fallback. The vendor call
//! synchronizes at the cuFFT FFI boundary; subsequent CUDA postprocessing and
//! explicit download remain stream-managed. Backend choice remains explicit.
//! Concrete non-AD execution uses
//! [`TensorFftExt`] and [`TensorReadFftExt`]. Eager FFTs use
//! `EagerSessionFftExt` on a borrowed session when `autodiff` is enabled;
//! consuming in-place transforms retain `EagerTensorFftExt`. Traced graph
//! construction uses [`TracedTensorFftExt`].
//!
//! # Cargo features
//!
//! | Feature | Enables |
//! |---|---|
//! | `native` (default) and the other CPU provider features | Forwarded to `tenferro-cpu`; see its documentation. |
//! | `autodiff` | The eager surface (`EagerSessionFftExt`, `EagerTensorFftExt`) and AD rules. Adds the `tenferro-ad` dependency. |
//! | `cuda` | CUDA execution through `tenferro-gpu`. |
//! | `rocm` | Placeholder; HIP/ROCm is not implemented. |
//!
//! For transforms without AD, use [`TensorFftExt`] / [`TensorReadFftExt`] on
//! concrete tensors inside a backend session; they need no `autodiff`.
//!
//! # N-dimensional transforms
//!
//! [`FftExecutor::fftn`], [`FftExecutor::ifftn`], [`FftExecutor::rfftn`], and
//! [`FftExecutor::irfftn`] transform every axis named in `axes`, and
//! [`TracedTensorFftExt`] exposes the same four operations on traced graphs.
//! One declared axis is the ordinary one-axis transform: it runs the same
//! single backend call, so `fftn(.., &[axis], ..)` equals [`FftExecutor::fft`].
//!
//! The normalization of [`FftNorm`] is applied once to the whole transform, not
//! once per axis: the transform over axes with `P = prod(shape over axes)` is
//! scaled by `1 / P` in the direction that `Backward`/`Forward` selects, and by
//! `1 / sqrt(P)` on both sides for `Ortho`. That is also the composition of the
//! one-axis transforms, whose per-axis factors multiply to the same single
//! factor.
//!
//! With the `cuda` feature, each axis is one `cufftPlanMany` transform of that
//! one-dimensional length, run over the resident tensor; an n-dimensional
//! request therefore uses one cuFFT plan per axis and never transfers the
//! intermediate spectra. A single multi-dimensional cuFFT plan is not used yet.
//!
//! # Examples
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_cpu::CpuEngine;
//! use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
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
//! let mut engine = CpuEngine::serial();
//! let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
//! let mut builder = Runtime::builder();
//! builder
//!     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
//!     .unwrap();
//! builder
//!     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
//!     .unwrap();
//! let runtime = builder.build().unwrap();
//! let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
//! let out = runtime
//!     .run(&mut [&mut engine], &prepared, &[])
//!     .unwrap()
//!     .pop()
//!     .unwrap()
//!     .into_tensor(&mut engine)
//!     .unwrap();
//! assert_eq!(out.shape(), &[4]);
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! ```
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_cpu::CpuEngine;
//! use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
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
//! let mut engine = CpuEngine::serial();
//! let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
//! let mut builder = Runtime::builder();
//! builder
//!     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
//!     .unwrap();
//! builder
//!     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
//!     .unwrap();
//! let runtime = builder.build().unwrap();
//! let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
//! let out = runtime
//!     .run(&mut [&mut engine], &prepared, &[])
//!     .unwrap()
//!     .pop()
//!     .unwrap()
//!     .into_tensor(&mut engine)
//!     .unwrap();
//! assert_eq!(out.shape(), &[4]);
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! ```
//!
//! ```
//! use num_complex::Complex64;
//! use tenferro_cpu::CpuEngine;
//! use tenferro_fft::{FftNorm, TensorFftExt};
//! use tenferro_tensor::Tensor;
//!
//! let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
//! let mut engine = CpuEngine::serial();
//! let out = x.fft(None, -1, FftNorm::Backward, &mut engine).unwrap();
//!
//! assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(10.0, 0.0));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![cfg_attr(docsrs, feature(doc_cfg))]

use std::any::Any;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_cpu::with_cpu_engine;
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::raw::with_cuda_engine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaEngine;
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{
    apply, ExtensionCacheStore, ExtensionExecutionContext, ExtensionOp,
};
#[cfg(feature = "autodiff")]
use tenferro_runtime::program::{CoreSemanticOp, ProgramValue, SemanticProgramBuilder};
use tenferro_runtime::{Error, ErrorPhase, Result, TracedTensor};
use tenferro_tensor::{CacheStats, DType, Engine, ErrorKind, Tensor, TensorRead};

mod backend;
mod cache;
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "autodiff")]
mod eager_ext;
#[cfg(feature = "autodiff")]
mod eager_in_place;
pub mod prelude;
mod spec;

pub use backend::{FftBackend, FftExecutionCache};
pub use cache::{
    fft_plan_cache_selector, FftPlanCache, DEFAULT_FFT_PLAN_CACHE_CAPACITY, FFT_PLAN_CACHE_NAME,
};
#[cfg(feature = "autodiff")]
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub use eager_ext::{EagerSessionFftExt, EagerTensorFftExt};
#[cfg(feature = "autodiff")]
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub use eager_in_place::EagerFftInPlaceError;
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
///
/// Use this executor for repeated concrete FFT calls that should reuse backend
/// plans. The immediate [`TensorFftExt`] and [`TensorReadFftExt`] methods stay
/// one-shot and do not retain hidden process-global, thread-local, or
/// backend-owned plan state between calls.
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
    /// for unsupported dtypes, a typed capability error when the session does
    /// not expose an FFT execution capability, or a typed backend source for
    /// execution.
    pub fn fft(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_fft_operation("FftExecutor::fft", input.dtype())?,
            "FftExecutor::fft",
            n,
            axis,
            norm,
            session,
        )
    }

    /// Execute an inverse complex FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-complex input, a typed capability error when the session does
    /// not expose an FFT execution capability, or a typed backend source for
    /// execution.
    pub fn ifft(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_ifft_operation("FftExecutor::ifft", input.dtype())?,
            "FftExecutor::ifft",
            n,
            axis,
            norm,
            session,
        )
    }

    /// Execute a real FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds` or
    /// `InvalidArgument` for invalid `axis`/`n`,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-real input, a typed capability error when the session does not
    /// expose an FFT execution capability, or a typed backend source for
    /// execution.
    pub fn rfft(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_rfft_operation("FftExecutor::rfft", input.dtype())?,
            "FftExecutor::rfft",
            n,
            axis,
            norm,
            session,
        )
    }

    /// Execute an inverse real FFT while reusing owned plans.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`,
    /// `InvalidArgument`, or spectrum-length details,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`]
    /// for a non-complex input, a typed capability error when the session does
    /// not expose an FFT execution capability, or a typed backend source for
    /// execution.
    pub fn irfft(
        &mut self,
        input: &Tensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute(
            input,
            concrete_irfft_operation("FftExecutor::irfft", input.dtype())?,
            "FftExecutor::irfft",
            n,
            axis,
            norm,
            session,
        )
    }

    /// Execute a forward n-dimensional FFT over `axes`.
    ///
    /// `axes` names the transformed axes in transform order. `shape` optionally
    /// supplies one transform length per axis (default: the input extent),
    /// matching NumPy's `fftn(a, s, axes)`. Axes may be negative, must be in
    /// bounds, and must not repeat. A real input returns the full complex
    /// spectrum like [`Self::fft`]; use [`Self::rfftn`] for the one-sided one.
    ///
    /// One axis is not a special case: `fftn(.., None, &[axis], ..)` is exactly
    /// [`Self::fft`] and runs the same single-axis backend call.
    ///
    /// # Normalization
    ///
    /// [`FftNorm`] is applied **once** to the whole n-dimensional transform.
    /// With `P = prod(shape over axes)`: `Backward` leaves the forward transform
    /// unscaled and scales the inverse by `1 / P`, `Forward` scales the forward
    /// transform by `1 / P`, and `Ortho` scales both by `1 / sqrt(P)`. The
    /// per-axis factors of that one factor also make successive one-axis calls
    /// with the same `norm` agree with this entry point. Applying the whole
    /// transform's factor once *per axis* instead is a different transform: for
    /// `k = axes.len()` it leaves an extra `P^(k - 1)` on the scaled side
    /// (`P^((k - 1) / 2)` for `Ortho`).
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] for an empty, repeated,
    /// out-of-bounds, or negative-beyond-rank `axes` entry, a `shape` whose
    /// length disagrees with `axes`, a zero transform length,
    /// [`tenferro_tensor::Error::Extension`] with [`ErrorKind::Unsupported`] for
    /// an unsupported dtype, a typed capability error when the session does not
    /// expose an FFT execution capability, or a typed backend source.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftNorm};
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut executor = FftExecutor::default();
    /// let mut engine = CpuEngine::serial();
    /// let input = Tensor::from_vec_col_major([2, 2], vec![Complex64::new(1.0, 0.0); 4])?;
    /// let output = executor.fftn(&input, None, &[0, 1], FftNorm::Backward, &mut engine)?;
    /// assert_eq!(output.as_slice::<Complex64>()?[0], Complex64::new(4.0, 0.0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn fftn(
        &mut self,
        input: &Tensor,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute_nd(
            "FftExecutor::fftn",
            concrete_fft_operation("FftExecutor::fftn", input.dtype())?,
            input,
            shape,
            axes,
            norm,
            session,
        )
    }

    /// Execute an inverse n-dimensional FFT over `axes`.
    ///
    /// Normalization is the one stated on [`Self::fftn`]: the inverse transform
    /// is scaled once by `1 / prod(shape over axes)` under [`FftNorm::Backward`].
    ///
    /// # Errors
    ///
    /// Returns the same validation, capability, and backend errors as
    /// [`Self::fftn`], plus [`ErrorKind::Unsupported`] for a non-complex input.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftNorm};
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut executor = FftExecutor::default();
    /// let mut engine = CpuEngine::serial();
    /// let input = Tensor::from_vec_col_major([2, 3], vec![Complex64::new(1.0, 0.0); 6])?;
    /// let output = executor.ifftn(&input, None, &[0, 1], FftNorm::Backward, &mut engine)?;
    /// assert_eq!(output.as_slice::<Complex64>()?[0], Complex64::new(1.0, 0.0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn ifftn(
        &mut self,
        input: &Tensor,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute_nd(
            "FftExecutor::ifftn",
            concrete_ifft_operation("FftExecutor::ifftn", input.dtype())?,
            input,
            shape,
            axes,
            norm,
            session,
        )
    }

    /// Execute a real-to-complex n-dimensional FFT over `axes`.
    ///
    /// The last requested axis carries the real transform of length `n`, so the
    /// output keeps `n / 2 + 1` points there and the full spectrum on the other
    /// axes, as in NumPy's `rfftn`. The real transform runs first, which is why
    /// the per-axis loop starts from the last declared axis.
    ///
    /// # Errors
    ///
    /// Returns the same validation, capability, and backend errors as
    /// [`Self::fftn`], plus [`ErrorKind::Unsupported`] for a non-`F32`/`F64`
    /// input.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftNorm};
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut executor = FftExecutor::default();
    /// let mut engine = CpuEngine::serial();
    /// let input = Tensor::from_vec_col_major([2, 4], vec![1.0_f64; 8])?;
    /// let output = executor.rfftn(&input, None, &[0, 1], FftNorm::Backward, &mut engine)?;
    /// assert_eq!(output.shape(), &[2, 3]);
    /// assert_eq!(output.as_slice::<Complex64>()?[0], Complex64::new(8.0, 0.0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rfftn(
        &mut self,
        input: &Tensor,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute_nd(
            "FftExecutor::rfftn",
            concrete_rfft_operation("FftExecutor::rfftn", input.dtype())?,
            input,
            shape,
            axes,
            norm,
            session,
        )
    }

    /// Execute a complex-to-real n-dimensional FFT over `axes`.
    ///
    /// This is the inverse of [`Self::rfftn`]: the last requested axis carries
    /// the one-sided Hermitian spectrum and is reconstructed last, so
    /// `irfftn(rfftn(x)) == x` for a real `x` under [`FftNorm::Backward`].
    ///
    /// # Errors
    ///
    /// Returns the same validation, capability, and backend errors as
    /// [`Self::fftn`], plus [`ErrorKind::Unsupported`] for a non-complex input
    /// and a spectrum-length mismatch for the real axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{FftExecutor, FftNorm};
    /// use tenferro_tensor::Tensor;
    ///
    /// let mut executor = FftExecutor::default();
    /// let mut engine = CpuEngine::serial();
    /// let input = Tensor::from_vec_col_major([2, 3], vec![Complex64::new(1.0, 0.0); 6])?;
    /// let output = executor.irfftn(&input, Some(&[2, 4]), &[0, 1], FftNorm::Backward, &mut engine)?;
    /// assert_eq!(output.shape(), &[2, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn irfftn(
        &mut self,
        input: &Tensor,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        self.execute_nd(
            "FftExecutor::irfftn",
            concrete_irfft_operation("FftExecutor::irfftn", input.dtype())?,
            input,
            shape,
            axes,
            norm,
            session,
        )
    }

    /// Validate an n-dimensional request and run it through the shared
    /// extension execution path, which owns the axis loop and the plan cache.
    #[allow(clippy::too_many_arguments)]
    fn execute_nd(
        &mut self,
        op_name: &'static str,
        operation: FftOperation,
        input: &Tensor,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        let fft_op = validated_fft_op(
            op_name,
            operation,
            Some(input.dtype()),
            input.shape().len(),
            Some(input.shape()),
            shape,
            "shape",
            axes,
            norm,
        )?;
        with_fft_exec_session(session, op_name, |backend| {
            execute_fft_op(
                &fft_op,
                &TensorRead::Tensor(input),
                backend,
                FftExecutionCache::caller_owned(&mut self.plans),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        input: &Tensor,
        operation: FftOperation,
        op_name: &'static str,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
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
        // The executor calls the concrete backend directly (no internal
        // session entry); the built-in dispatch only bridges the borrowed
        // session to its FFT execution capability.
        with_fft_exec_session(session, op_name, |backend| {
            backend.execute_fft(
                input,
                &spec,
                FftExecutionCache::caller_owned(&mut self.plans),
            )
        })
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

    /// Build a traced n-dimensional forward FFT over `axes`.
    ///
    /// `shape` supplies one transform length per axis and defaults to the input
    /// extents. Normalization is the single per-transform factor documented on
    /// [`FftExecutor::fftn`], and a one-element `axes` is exactly [`Self::fft`].
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` for an empty, repeated, or out-of-bounds
    /// `axes` entry or a `shape` that does not have one length per axis, and
    /// `Error::Extension` with `ErrorKind::Unsupported` for integer, boolean, or
    /// otherwise unsupported dtypes.
    ///
    /// # Deferred errors
    ///
    /// Symbolic axis extents and extension execution failures are checked at
    /// compile or execution time after concrete inputs are bound.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
    /// use num_complex::Complex64;
    /// use tenferro_fft::{FftNorm, TracedTensorFftExt};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0); 4])?;
    /// let y = x.fftn(None, &[0, 1], FftNorm::Backward)?;
    /// # let mut compiler = GraphCompiler::new();
    /// # let program = compiler.compile(&y)?;
    /// # let mut engine = CpuEngine::serial();
    /// # let engine_id = tenferro_cpu::runtime_engine_id()?;
    /// # let mut builder = Runtime::builder();
    /// # builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    /// # builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id)?)?;
    /// # let runtime = builder.build()?;
    /// # let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    /// # let out = runtime
    /// #     .run(&mut [&mut engine], &prepared, &[])?
    /// #     .pop()
    /// #     .unwrap()
    /// #     .into_tensor(&mut engine)?;
    /// assert_eq!(out.as_slice::<Complex64>()?[0], Complex64::new(4.0, 0.0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn fftn(&self, shape: Option<&[usize]>, axes: &[isize], norm: FftNorm) -> Result<TracedTensor>;

    /// Build a traced inverse n-dimensional FFT over `axes`.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`Self::fftn`] and
    /// `Error::Extension` with `ErrorKind::Unsupported` when the input is not
    /// `C32`/`C64`.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape and extension execution failures may be deferred to
    /// compile or execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
    /// use num_complex::Complex64;
    /// use tenferro_fft::{FftNorm, TracedTensorFftExt};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![Complex64::new(1.0, 0.0); 6])?;
    /// let y = x.ifftn(None, &[0, 1], FftNorm::Backward)?;
    /// # let mut compiler = GraphCompiler::new();
    /// # let program = compiler.compile(&y)?;
    /// # let mut engine = CpuEngine::serial();
    /// # let engine_id = tenferro_cpu::runtime_engine_id()?;
    /// # let mut builder = Runtime::builder();
    /// # builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    /// # builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id)?)?;
    /// # let runtime = builder.build()?;
    /// # let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    /// # let out = runtime
    /// #     .run(&mut [&mut engine], &prepared, &[])?
    /// #     .pop()
    /// #     .unwrap()
    /// #     .into_tensor(&mut engine)?;
    /// assert_eq!(out.as_slice::<Complex64>()?[0], Complex64::new(1.0, 0.0));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn ifftn(&self, shape: Option<&[usize]>, axes: &[isize], norm: FftNorm)
        -> Result<TracedTensor>;

    /// Build a traced real-to-complex n-dimensional FFT over `axes`.
    ///
    /// The last requested axis carries the real transform and keeps `n / 2 + 1`
    /// points of the returned one-sided Hermitian spectrum.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`Self::fftn`] and
    /// `Error::Extension` with `ErrorKind::Unsupported` when the input is not
    /// `F32`/`F64`.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape and extension execution failures may be deferred to
    /// compile or execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
    /// use num_complex::Complex64;
    /// use tenferro_fft::{FftNorm, TracedTensorFftExt};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 4], vec![1.0_f64; 8])?;
    /// let y = x.rfftn(None, &[0, 1], FftNorm::Backward)?;
    /// # let mut compiler = GraphCompiler::new();
    /// # let program = compiler.compile(&y)?;
    /// # let mut engine = CpuEngine::serial();
    /// # let engine_id = tenferro_cpu::runtime_engine_id()?;
    /// # let mut builder = Runtime::builder();
    /// # builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    /// # builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id)?)?;
    /// # let runtime = builder.build()?;
    /// # let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    /// # let out = runtime
    /// #     .run(&mut [&mut engine], &prepared, &[])?
    /// #     .pop()
    /// #     .unwrap()
    /// #     .into_tensor(&mut engine)?;
    /// assert_eq!(out.shape(), &[2, 3]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn rfftn(&self, shape: Option<&[usize]>, axes: &[isize], norm: FftNorm)
        -> Result<TracedTensor>;

    /// Build a traced complex-to-real n-dimensional FFT over `axes`.
    ///
    /// This is the inverse of [`Self::rfftn`]: the last requested axis carries
    /// the one-sided Hermitian spectrum that is reconstructed last.
    ///
    /// # Errors
    ///
    /// Returns the validation errors of [`Self::fftn`],
    /// `Error::Extension` with `ErrorKind::Unsupported` when the input is not
    /// `C32`/`C64`, and a spectrum-length mismatch for the real axis.
    ///
    /// # Deferred errors
    ///
    /// Symbolic spectrum lengths and extension execution failures may be
    /// deferred to compile or execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tenferro_cpu::CpuEngine;
    /// # use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
    /// use num_complex::Complex64;
    /// use tenferro_fft::{FftNorm, TracedTensorFftExt};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![Complex64::new(1.0, 0.0); 6])?;
    /// let y = x.irfftn(Some(&[2, 4]), &[0, 1], FftNorm::Backward)?;
    /// # let mut compiler = GraphCompiler::new();
    /// # let program = compiler.compile(&y)?;
    /// # let mut engine = CpuEngine::serial();
    /// # let engine_id = tenferro_cpu::runtime_engine_id()?;
    /// # let mut builder = Runtime::builder();
    /// # builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    /// # builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id)?)?;
    /// # let runtime = builder.build()?;
    /// # let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    /// # let out = runtime
    /// #     .run(&mut [&mut engine], &prepared, &[])?
    /// #     .pop()
    /// #     .unwrap()
    /// #     .into_tensor(&mut engine)?;
    /// assert_eq!(out.shape(), &[2, 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn irfftn(
        &self,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
    ) -> Result<TracedTensor>;
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

    fn fftn(&self, shape: Option<&[usize]>, axes: &[isize], norm: FftNorm) -> Result<TracedTensor> {
        fftn(self, shape, axes, norm)
    }

    fn ifftn(
        &self,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
    ) -> Result<TracedTensor> {
        ifftn(self, shape, axes, norm)
    }

    fn rfftn(
        &self,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
    ) -> Result<TracedTensor> {
        rfftn(self, shape, axes, norm)
    }

    fn irfftn(
        &self,
        shape: Option<&[usize]>,
        axes: &[isize],
        norm: FftNorm,
    ) -> Result<TracedTensor> {
        irfftn(self, shape, axes, norm)
    }
}

/// Backend-explicit FFT methods for concrete [`Tensor`] values.
///
/// This is the non-AD immediate execution surface. It uses unsuffixed method
/// names because the receiver is an owned compact tensor value. Use
/// [`TensorReadFftExt`] when the input is a borrowed view or other
/// [`TensorRead`] value.
///
/// Direct calls intentionally use a call-local one-shot FFT plan cache. Use
/// [`FftExecutor`] for repeated concrete calls with stable transform lengths,
/// or traced/runtime execution when the runtime should own the extension cache.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::{FftNorm, TensorFftExt};
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
/// let mut engine = CpuEngine::serial();
///
/// let spectrum = input.fft(None, -1, FftNorm::Backward, &mut engine)?;
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
    /// integer or boolean input, a typed capability error when the session
    /// does not expose an FFT execution capability, or a typed backend source
    /// for execution.
    fn fft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-complex input, a typed capability error when the session does not
    /// expose an FFT execution capability, or a typed backend source for
    /// execution.
    fn ifft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-`F32`/`F64` input, a typed capability error when the session does
    /// not expose an FFT execution capability, or a typed backend source for
    /// execution.
    fn rfft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds`, `InvalidArgument`,
    /// or spectrum-length details, `Error::Extension` with
    /// `ErrorKind::Unsupported` for a non-complex input, a typed capability
    /// error when the session does not expose an FFT execution capability, or
    /// a typed backend source for execution.
    fn irfft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorFftExt for Tensor {
    fn fft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
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
        with_fft_exec_session(session, "TensorFftExt::fft", |backend| {
            execute_concrete_fft_op(self, &spec, backend)
        })
    }

    fn ifft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
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
        with_fft_exec_session(session, "TensorFftExt::ifft", |backend| {
            execute_concrete_fft_op(self, &spec, backend)
        })
    }

    fn rfft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
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
        with_fft_exec_session(session, "TensorFftExt::rfft", |backend| {
            execute_concrete_fft_op(self, &spec, backend)
        })
    }

    fn irfft(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
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
        with_fft_exec_session(session, "TensorFftExt::irfft", |backend| {
            execute_concrete_fft_op(self, &spec, backend)
        })
    }
}

/// Backend-explicit FFT methods for read-only tensor inputs.
///
/// The `_read` suffix follows the repository convention for APIs that
/// explicitly accept [`TensorRead`] values such as borrowed views.
///
/// Direct read calls intentionally materialize through a call-local one-shot
/// FFT plan cache. Use [`FftExecutor`] on compact owned tensors when repeated
/// concrete calls should retain backend plans across calls.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::{FftNorm, TensorReadFftExt};
/// use tenferro_tensor::{TensorRead, TensorView};
///
/// let shape = [4usize];
/// let data = [1.0_f64, 2.0, 3.0, 4.0];
/// let input = TensorRead::from_view(TensorView::f64(&shape, &data)?);
/// let mut engine = CpuEngine::serial();
///
/// let spectrum = input.fft_read(None, -1, FftNorm::Backward, &mut engine)?;
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
    /// integer or boolean input, a typed capability error when the session
    /// does not expose an FFT execution capability, or a typed backend source
    /// for materialization or execution.
    fn fft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-complex input, a typed capability error when the session does not
    /// expose an FFT execution capability, or a typed backend source for
    /// materialization.
    fn ifft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds` or `InvalidArgument`
    /// for `axis`/`n`, `Error::Extension` with `ErrorKind::Unsupported` for a
    /// non-`F32`/`F64` input, a typed capability error when the session does
    /// not expose an FFT execution capability, or a typed backend source for
    /// materialization.
    fn rfft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;

    /// Execute a one-dimensional inverse real FFT along `axis`.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` with `AxisOutOfBounds`, `InvalidArgument`,
    /// or spectrum-length details, `Error::Extension` with
    /// `ErrorKind::Unsupported` for a non-complex input, a typed capability
    /// error when the session does not expose an FFT execution capability, or
    /// a typed backend source for materialization.
    fn irfft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor>;
}

impl TensorReadFftExt for TensorRead<'_> {
    fn fft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_fft_exec_session(session, "TensorReadFftExt::fft_read", |backend| {
            execute_concrete_fft_read_op(
                self,
                concrete_fft_operation("TensorReadFftExt::fft_read", self.dtype())?,
                "TensorReadFftExt::fft_read",
                n,
                axis,
                norm,
                backend,
            )
        })
    }

    fn ifft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_fft_exec_session(session, "TensorReadFftExt::ifft_read", |backend| {
            execute_concrete_fft_read_op(
                self,
                concrete_ifft_operation("TensorReadFftExt::ifft_read", self.dtype())?,
                "TensorReadFftExt::ifft_read",
                n,
                axis,
                norm,
                backend,
            )
        })
    }

    fn rfft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_fft_exec_session(session, "TensorReadFftExt::rfft_read", |backend| {
            execute_concrete_fft_read_op(
                self,
                concrete_rfft_operation("TensorReadFftExt::rfft_read", self.dtype())?,
                "TensorReadFftExt::rfft_read",
                n,
                axis,
                norm,
                backend,
            )
        })
    }

    fn irfft_read(
        &self,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
        session: &mut dyn Engine,
    ) -> tenferro_tensor::Result<Tensor> {
        with_fft_exec_session(session, "TensorReadFftExt::irfft_read", |backend| {
            execute_concrete_fft_read_op(
                self,
                concrete_irfft_operation("TensorReadFftExt::irfft_read", self.dtype())?,
                "TensorReadFftExt::irfft_read",
                n,
                axis,
                norm,
                backend,
            )
        })
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
    /// Validated, non-empty, duplicate-free axes. The last axis is the real
    /// transform axis of `R2cFull`/`R2cOnesided`/`C2r`, as in NumPy `rfftn`.
    axes: Vec<usize>,
    /// One requested length per axis, when the caller supplied them.
    lengths: Option<Vec<usize>>,
    norm: FftNorm,
}

impl FftOp {
    fn new(
        operation: FftOperation,
        axes: Vec<usize>,
        lengths: Option<Vec<usize>>,
        norm: FftNorm,
    ) -> Self {
        Self {
            operation,
            axes,
            lengths,
            norm,
        }
    }

    /// Single spec for the whole request, used by concrete single-axis backends.
    fn spec(&self, dtype: DType, shape: &[usize]) -> FftPlanSpec {
        FftPlanSpec::new(
            self.operation,
            self.axes.clone(),
            self.lengths.clone(),
            self.norm,
            dtype,
            shape.to_vec(),
        )
    }

    /// Single-axis spec for the `index`-th declared axis against `shape`.
    fn step_spec(&self, index: usize, dtype: DType, shape: &[usize]) -> FftPlanSpec {
        FftPlanSpec::new(
            self.operation.step_operation(index, self.axes.len()),
            vec![self.axes[index]],
            self.lengths.as_ref().map(|lengths| vec![lengths[index]]),
            self.norm,
            dtype,
            shape.to_vec(),
        )
    }

    /// Axis application order for the successive one-dimensional transforms.
    ///
    /// Complex transforms commute, so they keep the declared order. A real
    /// forward transform must consume the real input first, and a complex-to-real
    /// inverse must produce real output last; both name their real axis last, so
    /// only the forward case needs the rotation.
    fn execution_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.axes.len()).collect();
        if matches!(
            self.operation,
            FftOperation::R2cFull | FftOperation::R2cOnesided
        ) {
            order.rotate_right(1);
        }
        order
    }

    #[cfg(feature = "autodiff")]
    fn c2c_adjoint(&self) -> Option<Self> {
        let operation = match self.operation {
            FftOperation::C2cForward => FftOperation::C2cInverse,
            FftOperation::C2cInverse => FftOperation::C2cForward,
            FftOperation::R2cFull | FftOperation::R2cOnesided | FftOperation::C2r => return None,
        };
        Some(Self {
            operation,
            axes: self.axes.clone(),
            lengths: self.lengths.clone(),
            norm: self.norm.c2c_adjoint(),
        })
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
        hasher.write_usize(self.axes.len());
        for axis in &self.axes {
            hasher.write_usize(*axis);
        }
        match &self.lengths {
            Some(lengths) => {
                hasher.write_u8(1);
                for len in lengths {
                    hasher.write_usize(*len);
                }
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

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtype = ctx.input_dtype(0)?;
        let input_shape = ctx.input_shape(0)?;
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
            FftOperation::R2cFull | FftOperation::R2cOnesided => match input_dtype {
                DType::F32 => DType::C32,
                DType::F64 => DType::C64,
                _ => {
                    return Err(tensor_unsupported_dtype(
                        "tenferro-fft",
                        input_dtype,
                        "F32 or F64",
                    ));
                }
            },
            FftOperation::C2r => match input_dtype {
                DType::C32 => DType::F32,
                DType::C64 => DType::F64,
                _ => {
                    return Err(tensor_unsupported_dtype(
                        "tenferro-fft",
                        input_dtype,
                        "C32 or C64",
                    ));
                }
            },
        };

        let count = self.axes.len();
        let mut out_shape = input_shape.to_vec();
        for (index, &axis) in self.axes.iter().enumerate() {
            if axis >= input_shape.len() {
                return Err(tenferro_tensor::Error::axis_out_of_bounds(
                    "tenferro-fft",
                    axis,
                    input_shape.len(),
                ));
            }
            let len = self.lengths.as_ref().map(|lengths| lengths[index]);
            out_shape[axis] = match self.operation.step_operation(index, count) {
                FftOperation::C2cForward | FftOperation::C2cInverse => {
                    transform_len_dim(len, &input_shape[axis])
                }
                FftOperation::R2cFull | FftOperation::R2cOnesided => {
                    let len = transform_len_dim(len, &input_shape[axis]);
                    if self.operation.is_onesided() {
                        len / 2usize + 1usize
                    } else {
                        len
                    }
                }
                FftOperation::C2r => output_dim_c2r(&input_shape[axis], len)?,
            };
        }

        Ok(vec![(output_dtype, out_shape)])
    }
}

/// Run a concrete FFT body against the built-in FFT execution sessions
/// carried by `session` (CPU/CUDA/WebGPU), returning a typed capability error
/// when the session does not expose an FFT execution capability.
///
/// This is the built-in dispatch shared by the concrete FFT surface; callers
/// never downcast themselves (issue #1680 Phase 3). Third-party
/// [`FftBackend`] implementations remain supported through the SPI trait, but
/// the concrete op path is built-in-session only.
fn with_fft_exec_session<X>(
    session: &mut dyn Engine,
    op: &'static str,
    f: impl FnOnce(&mut dyn FftBackend) -> tenferro_tensor::Result<X>,
) -> tenferro_tensor::Result<X> {
    // The capability branches are mutually exclusive, so `f` runs exactly
    // once. Probe the marker first, then re-extract the same exec session and
    // run the concrete body on it (FnOnce cannot be captured by several
    // branch closures).
    if with_cpu_engine(session, |_| ()).is_some() {
        return with_cpu_engine(session, |exec| f(exec as &mut dyn FftBackend))
            .expect("marker probe matched a CPU execution session");
    }
    #[cfg(feature = "cuda")]
    if with_cuda_engine(session, |_| ()).is_some() {
        return with_cuda_engine(session, |exec| f(exec as &mut dyn FftBackend))
            .expect("marker probe matched a CUDA execution session");
    }
    Err(tenferro_tensor::Error::unsupported(
        op,
        "selected backend session does not expose an FFT execution capability",
    ))
}

fn execute_concrete_fft_op(
    input: &Tensor,
    spec: &FftPlanSpec,
    backend: &mut dyn FftBackend,
) -> tenferro_tensor::Result<Tensor> {
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    backend.execute_fft(input, spec, FftExecutionCache::caller_owned(&mut plans))
}

#[allow(clippy::too_many_arguments)]
fn execute_concrete_fft_read_op(
    input: &TensorRead<'_>,
    operation: FftOperation,
    op_name: &'static str,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
    backend: &mut dyn FftBackend,
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
    let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
    backend.execute_fft_read(
        input.clone(),
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
    validated_fft_plan_spec(
        op,
        operation,
        input_dtype,
        input_shape,
        n.map(|n| [n]).as_ref().map(<[usize; 1]>::as_slice),
        &[axis],
        norm,
    )
}

/// Validate one FFT request over one or more axes and build its descriptor.
///
/// This is the only place that normalizes and de-duplicates axes, checks that
/// exactly one requested length is supplied per axis, and validates the dtype
/// and the concrete per-axis transform shapes. The eager executor, the traced
/// builder, and the extension execution path all reach it, so 1-D and n-D
/// requests cannot drift apart.
#[allow(clippy::too_many_arguments)]
fn validated_fft_op(
    op: &'static str,
    operation: FftOperation,
    dtype: Option<DType>,
    input_rank: usize,
    input_shape: Option<&[usize]>,
    lengths: Option<&[usize]>,
    length_argument: &'static str,
    axes: &[isize],
    norm: FftNorm,
) -> tenferro_tensor::Result<FftOp> {
    if input_rank == 0 {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "rank",
            "FFT requires rank >= 1",
        ));
    }
    if axes.is_empty() {
        return Err(tenferro_tensor::Error::invalid_argument(
            op,
            "axes",
            "FFT requires at least one axis",
        ));
    }
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = normalize_axis(op, axis, input_rank)?;
        if normalized.contains(&axis) {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                "axes",
                format!("axis {axis} is repeated"),
            ));
        }
        normalized.push(axis);
    }
    if let Some(lengths) = lengths {
        if lengths.len() != normalized.len() {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                length_argument,
                format!(
                    "expected {} requested lengths for {} axes, got {}",
                    normalized.len(),
                    normalized.len(),
                    lengths.len()
                ),
            ));
        }
        if lengths.contains(&0) {
            return Err(tenferro_tensor::Error::invalid_argument(
                op,
                length_argument,
                "transform length must be positive",
            ));
        }
    }
    if let Some(dtype) = dtype {
        validate_operation_dtype(op, operation, dtype)?;
    }
    let count = normalized.len();
    if let Some(shape) = input_shape {
        for (index, &axis) in normalized.iter().enumerate() {
            let len = lengths.map(|lengths| lengths[index]);
            if len.is_none() && shape[axis] == 0 {
                return Err(tenferro_tensor::Error::invalid_argument(
                    op,
                    length_argument,
                    "transform length must be positive",
                ));
            }
            match operation.step_operation(index, count) {
                FftOperation::C2cForward | FftOperation::C2cInverse => {
                    transform_len(shape, axis, len)?;
                }
                FftOperation::R2cFull | FftOperation::R2cOnesided => {
                    transform_len(shape, axis, len)?;
                }
                FftOperation::C2r => {
                    output_shape_c2r(shape, axis, len)?;
                }
            }
        }
    }
    Ok(FftOp::new(
        operation,
        normalized,
        lengths.map(<[usize]>::to_vec),
        norm,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validated_fft_plan_spec(
    op: &'static str,
    operation: FftOperation,
    input_dtype: DType,
    input_shape: &[usize],
    lengths: Option<&[usize]>,
    axes: &[isize],
    norm: FftNorm,
) -> tenferro_tensor::Result<FftPlanSpec> {
    let fft_op = validated_fft_op(
        op,
        operation,
        Some(input_dtype),
        input_shape.len(),
        Some(input_shape),
        lengths,
        "n",
        axes,
        norm,
    )?;
    Ok(fft_op.spec(input_dtype, input_shape))
}

fn concrete_fft_operation(op: &'static str, dtype: DType) -> tenferro_tensor::Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2cForward),
        DType::F32 | DType::F64 => Ok(FftOperation::R2cFull),
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
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
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
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
        DType::C32 | DType::C64 | DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
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
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool | DType::External(_) => {
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
impl SemanticLinearizeRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<SemanticLinearizeResult, SemanticAdError> {
        let fft_op = semantic_fft_payload(request.op(), SemanticAdRuleKind::Linearize)?;
        if !fft_op.operation.is_c2c() {
            return Err(semantic_fft_unsupported(
                fft_op.operation,
                SemanticAdRuleKind::Linearize,
            ));
        }
        let tangent = match request.tangent_inputs()[0] {
            AdValue::Absent => AdValue::Absent,
            AdValue::Value(tangent) => {
                AdValue::Value(builder.add_extension(Arc::new(fft_op.clone()), &[tangent])?[0])
            }
        };
        Ok(SemanticLinearizeResult::new([tangent], []))
    }
}

#[cfg(feature = "autodiff")]
impl SemanticLinearTransposeRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn residual_mask(&self) -> ResidualSpec {
        // The variable-length adjoint path reads primal input 0 as a tensor
        // (ShapeOf + PadToMatch against the original input).
        ResidualSpec::input(0)
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        Ok([semantic_fft_adjoint(
            request.op(),
            request.cotangent_outputs()[0],
            request.active_inputs()[0],
            request.primal_input_value(0)?,
            request.residual_mask(),
            builder,
        )?]
        .into())
    }
}

#[cfg(feature = "autodiff")]
impl SemanticPrimalVjpRule for FftAdRule {
    fn family_id(&self) -> &'static str {
        FFT_EXTENSION_FAMILY_ID
    }

    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0)
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        Ok([semantic_fft_adjoint(
            request.op(),
            request.cotangent_outputs()[0],
            request.active_inputs()[0],
            request.primal_input_value(0)?,
            request.residual_mask(),
            builder,
        )?]
        .into())
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Copy)]
enum SemanticAdRuleKind {
    Linearize,
    Transpose,
}

#[cfg(feature = "autodiff")]
fn semantic_fft_payload(
    op: &dyn ExtensionOp,
    role: SemanticAdRuleKind,
) -> std::result::Result<&FftOp, SemanticAdError> {
    op.as_any().downcast_ref::<FftOp>().ok_or_else(|| {
        semantic_fft_unsupported_family(
            FFT_EXTENSION_FAMILY_ID,
            role,
            "FFT semantic AD received an incompatible extension payload",
        )
    })
}

#[cfg(feature = "autodiff")]
fn semantic_fft_adjoint(
    op: &dyn ExtensionOp,
    cotangent: AdValue,
    active: bool,
    primal_input: ProgramValue,
    residual_mask: ResidualSpec,
    builder: &mut SemanticProgramBuilder,
) -> std::result::Result<AdValue, SemanticAdError> {
    if !active {
        return Ok(AdValue::Absent);
    }
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(AdValue::Absent);
    };
    let fft_op = semantic_fft_payload(op, SemanticAdRuleKind::Transpose)?;
    if !fft_op.operation.is_c2c() {
        return Err(semantic_fft_unsupported(
            fft_op.operation,
            SemanticAdRuleKind::Transpose,
        ));
    }
    let adjoint_op = fft_op
        .c2c_adjoint()
        .ok_or_else(|| semantic_fft_unsupported(fft_op.operation, SemanticAdRuleKind::Transpose))?;
    let adjoint = builder.add_extension(Arc::new(adjoint_op), &[cotangent])?[0];
    restore_semantic_c2c_adjoint_input_length(builder, adjoint, primal_input, residual_mask, fft_op)
        .map(AdValue::Value)
}

#[cfg(feature = "autodiff")]
fn restore_semantic_c2c_adjoint_input_length(
    builder: &mut SemanticProgramBuilder,
    adjoint: ProgramValue,
    primal_input: ProgramValue,
    residual_mask: ResidualSpec,
    fft_op: &FftOp,
) -> std::result::Result<ProgramValue, SemanticAdError> {
    let Some(lengths) = fft_op.lengths.as_ref() else {
        return Ok(adjoint);
    };
    debug_assert!(
        residual_mask.declares_input(0),
        "fft transpose read primal input 0 as a tensor operand but the residual mask does not \
         declare it; declare it in the fft rule's residual mask"
    );
    let mut adjoint = adjoint;
    for (&axis, &transform_len) in fft_op.axes.iter().zip(lengths) {
        let input_len = builder
            .value_metadata(primal_input)?
            .shape()
            .get(axis)
            .and_then(|extent| extent.as_exact())
            .and_then(|dim| match dim {
                tenferro_ops::dim_expr::DimExpr::Const(value) => Some(*value),
                _ => None,
            });
        if input_len == Some(transform_len) {
            continue;
        }

        let size = builder.add_op(CoreSemanticOp::ShapeOf { axis }, &[primal_input])?[0];
        let truncated =
            builder.add_op(CoreSemanticOp::DynamicTruncate { axis }, &[adjoint, size])?[0];
        adjoint = builder.add_op(
            CoreSemanticOp::PadToMatch { axis },
            &[truncated, primal_input],
        )?[0];
    }
    Ok(adjoint)
}

#[cfg(feature = "autodiff")]
fn semantic_fft_unsupported(operation: FftOperation, role: SemanticAdRuleKind) -> SemanticAdError {
    semantic_fft_unsupported_family(
        fft_ad_family_id(operation),
        role,
        "FFT operation has no semantic AD rule",
    )
}

#[cfg(feature = "autodiff")]
fn semantic_fft_unsupported_family(
    family_id: &'static str,
    role: SemanticAdRuleKind,
    message: impl Into<String>,
) -> SemanticAdError {
    SemanticAdError::Unsupported {
        family_id,
        role: match role {
            SemanticAdRuleKind::Linearize => {
                tenferro_ad::semantic_extension::SemanticAdRuleRole::Linearize
            }
            SemanticAdRuleKind::Transpose => {
                tenferro_ad::semantic_extension::SemanticAdRuleRole::LinearTranspose
            }
        },
        message: message.into(),
    }
}

/// Return the semantic-program FFT extension AD rule set.
#[cfg(feature = "autodiff")]
///
/// # Errors
///
/// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] if the FFT
/// family identifier is invalid, or
/// [`SemanticExtensionRegistryError::DuplicateRule`] if a rule for the family
/// and role is already registered.
pub fn semantic_ad_rules(
) -> std::result::Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(FftAdRule))?
        .with_linear_transpose(Arc::new(FftAdRule))?
        .with_primal_vjp(Arc::new(FftAdRule))
}

pub(crate) fn execute_fft_extension_reads_session(
    op: &FftOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, dyn Engine + '_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let (session, caches) = ctx.parts_mut();
    execute_fft_extension_reads_on_session(op, inputs, session, caches)
}

fn execute_fft_extension_reads_for_capability<B: FftBackend + ?Sized>(
    op: &FftOp,
    inputs: &[TensorRead<'_>],
    session: &mut B,
    caches: &mut ExtensionCacheStore,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.len() != 1 {
        return Err(tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "inputs",
            format!("expected 1 input, got {}", inputs.len()),
        ));
    }
    Ok(vec![execute_fft_op(
        op,
        &inputs[0],
        session,
        FftExecutionCache::engine_owned(caches),
    )?])
}

/// Execute one validated FFT request as successive one-axis backend calls.
///
/// This is the only axis loop for n-dimensional FFTs: the eager executor
/// ([`FftExecutor::fftn`]) and the traced/extension execution path both enter
/// here. Each iteration hands the backend a single-axis [`FftPlanSpec`], so the
/// backend runs its own plan for that length and dtype (`rustfft`/`realfft` on
/// CPU, a one-dimensional `cufftPlanMany` on CUDA) and reuses it from the shared
/// plan cache. Nothing here computes a transform: this function owns only the
/// axis order, the per-axis operation, and the intermediate shape bookkeeping.
fn execute_fft_op<B: FftBackend + ?Sized>(
    op: &FftOp,
    input: &TensorRead<'_>,
    session: &mut B,
    mut cache: FftExecutionCache<'_>,
) -> tenferro_tensor::Result<Tensor> {
    let mut current: Option<Tensor> = None;
    for index in op.execution_order() {
        let (dtype, shape) = match &current {
            Some(tensor) => (tensor.dtype(), tensor.shape().to_vec()),
            None => (input.dtype(), input.shape().to_vec()),
        };
        let spec = op.step_spec(index, dtype, &shape);
        let step_name = fft_op_name(spec.operation());
        let read = match &current {
            Some(tensor) => TensorRead::Tensor(tensor),
            None => input.clone(),
        };
        // The borrowed-view hook rejects an unusable placement before
        // canonicalization. An owned tensor keeps the backend's own placement
        // contract instead, where a managed CPU tensor is a legal input.
        if matches!(&read, TensorRead::View(_)) {
            session.validate_fft_read_input(step_name, &read)?;
        }
        let output = session.execute_fft_read(
            read,
            &spec,
            FftExecutionCache::with_owner(cache.owner(), cache.store_mut()),
        )?;
        current = Some(output);
    }
    current.ok_or_else(|| {
        tenferro_tensor::Error::invalid_argument(
            "tenferro-fft",
            "axes",
            "FFT requires at least one axis",
        )
    })
}

fn execute_fft_extension_reads_on_session(
    op: &FftOp,
    inputs: &[TensorRead<'_>],
    session: &mut dyn Engine,
    caches: &mut ExtensionCacheStore,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if let Some(result) = with_cpu_engine(session, |session| {
        execute_fft_extension_reads_for_capability(op, inputs, session, caches)
    }) {
        return result;
    }
    #[cfg(feature = "cuda")]
    if let Some(result) = with_cuda_engine(session, |session| {
        execute_fft_extension_reads_for_capability(op, inputs, session, caches)
    }) {
        return result;
    }
    Err(tenferro_tensor::Error::unsupported(
        fft_op_name(op.operation),
        "selected backend session does not expose an FFT execution capability",
    ))
}

define_extension_runtime! {
    runtime = FftRuntime,
    family_id = FFT_EXTENSION_FAMILY_ID,
    op_type = FftOp,
    execute_in_session = execute_fft_extension_reads_in_session,
    session_supported = fft_session_supported,
}

/// Adapter from the scheduler/`apply_eager` borrowed-session shape to the
/// existing FFT session executor. Reuses the same forward kernel already shared
/// by the owner and eager paths; do not reimplement it here.
fn execute_fft_extension_reads_in_session(
    op: &FftOp,
    session: &mut dyn Engine,
    caches: &mut ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let mut ctx = ExtensionExecutionContext::new(session, caches);
    execute_fft_extension_reads_session(op, inputs, &mut ctx)
}

fn fft_session_supported<B: Engine + 'static>(_op: &FftOp) -> bool {
    // The session executor routes CPU/CUDA/WebGPU through their FftBackend exec
    // sessions; keep scheduler-session admission consistent with the backends
    // that `execute_fft_extension_reads_on_session` actually handles.
    let type_id = std::any::TypeId::of::<B>();
    type_id == std::any::TypeId::of::<tenferro_cpu::CpuEngine>() || {
        #[cfg(feature = "cuda")]
        {
            type_id == std::any::TypeId::of::<CudaEngine>()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
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
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]).unwrap();
/// let y = x.fft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut engine = CpuEngine::serial();
/// let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
/// let mut builder = Runtime::builder();
/// builder
///     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
///     .unwrap();
/// builder
///     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
///     .unwrap();
/// let runtime = builder.build().unwrap();
/// let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
/// let out = runtime
///     .run(&mut [&mut engine], &prepared, &[])
///     .unwrap()
///     .pop()
///     .unwrap()
///     .into_tensor(&mut engine)
///     .unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
/// ```
fn fft(input: &TracedTensor, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<TracedTensor> {
    let operation = runtime_forward_fft_operation(input.dtype)?;
    let lengths = single_axis_lengths(n);
    apply_unary_fftn(
        "fft",
        input,
        operation,
        lengths.as_ref().map(|lengths| lengths.as_slice()),
        "n",
        &[axis],
        norm,
    )
}

/// Build a one-dimensional inverse FFT along `axis`.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let spectrum = TracedTensor::from_vec_col_major(vec![2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)]).unwrap();
/// let y = spectrum.ifft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut engine = CpuEngine::serial();
/// let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
/// let mut builder = Runtime::builder();
/// builder
///     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
///     .unwrap();
/// builder
///     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
///     .unwrap();
/// let runtime = builder.build().unwrap();
/// let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
/// let out = runtime
///     .run(&mut [&mut engine], &prepared, &[])
///     .unwrap()
///     .pop()
///     .unwrap()
///     .into_tensor(&mut engine)
///     .unwrap();
/// assert_eq!(out.as_slice::<Complex64>().unwrap()[0], Complex64::new(1.0, 0.0));
/// ```
fn ifft(
    input: &TracedTensor,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype("ifft", input.dtype, &[DType::C32, DType::C64], "C32 or C64")?;
    let lengths = single_axis_lengths(n);
    apply_unary_fftn(
        "ifft",
        input,
        FftOperation::C2cInverse,
        lengths.as_ref().map(|lengths| lengths.as_slice()),
        "n",
        &[axis],
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
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
/// use tenferro_fft::{FftNorm, TracedTensorFftExt};
///
/// let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let y = x.rfft(None, -1, FftNorm::Backward).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut engine = CpuEngine::serial();
/// let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
/// let mut builder = Runtime::builder();
/// builder
///     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
///     .unwrap();
/// builder
///     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
///     .unwrap();
/// let runtime = builder.build().unwrap();
/// let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
/// let out = runtime
///     .run(&mut [&mut engine], &prepared, &[])
///     .unwrap()
///     .pop()
///     .unwrap()
///     .into_tensor(&mut engine)
///     .unwrap();
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
    let lengths = single_axis_lengths(n);
    apply_unary_fftn(
        "rfft",
        input,
        FftOperation::R2cOnesided,
        lengths.as_ref().map(|lengths| lengths.as_slice()),
        "n",
        &[axis],
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
/// use tenferro_cpu::CpuEngine;
/// use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
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
/// let mut engine = CpuEngine::serial();
/// let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
/// let mut builder = Runtime::builder();
/// builder
///     .register_engine(tenferro_cpu::runtime_engine_registration(&engine).unwrap())
///     .unwrap();
/// builder
///     .install_extension_module(tenferro_fft::extension_module::<CpuEngine>(engine_id).unwrap())
///     .unwrap();
/// let runtime = builder.build().unwrap();
/// let prepared = runtime.prepare(&program, &[], &[engine.target()]).unwrap();
/// let out = runtime
///     .run(&mut [&mut engine], &prepared, &[])
///     .unwrap()
///     .pop()
///     .unwrap()
///     .into_tensor(&mut engine)
///     .unwrap();
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
    let lengths = single_axis_lengths(n);
    apply_unary_fftn(
        "irfft",
        input,
        FftOperation::C2r,
        lengths.as_ref().map(|lengths| lengths.as_slice()),
        "n",
        &[axis],
        norm,
    )
}

/// Build an n-dimensional forward FFT over `axes`.
///
/// Real input returns the full complex spectrum, exactly like [`fft`]; the
/// normalization is the one [`FftExecutor::fftn`] documents.
fn fftn(
    input: &TracedTensor,
    lengths: Option<&[usize]>,
    axes: &[isize],
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype(
        "fftn",
        input.dtype,
        &[DType::F32, DType::F64, DType::C32, DType::C64],
        "F32, F64, C32, or C64",
    )?;
    let operation = runtime_forward_fft_operation(input.dtype)?;
    apply_unary_fftn("fftn", input, operation, lengths, "shape", axes, norm)
}

/// Build an inverse n-dimensional FFT over `axes`.
fn ifftn(
    input: &TracedTensor,
    lengths: Option<&[usize]>,
    axes: &[isize],
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype(
        "ifftn",
        input.dtype,
        &[DType::C32, DType::C64],
        "C32 or C64",
    )?;
    apply_unary_fftn(
        "ifftn",
        input,
        FftOperation::C2cInverse,
        lengths,
        "shape",
        axes,
        norm,
    )
}

/// Build a real-to-complex n-dimensional FFT over `axes`.
///
/// The last requested axis carries the real transform.
fn rfftn(
    input: &TracedTensor,
    lengths: Option<&[usize]>,
    axes: &[isize],
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype(
        "rfftn",
        input.dtype,
        &[DType::F32, DType::F64],
        "F32 or F64",
    )?;
    apply_unary_fftn(
        "rfftn",
        input,
        FftOperation::R2cOnesided,
        lengths,
        "shape",
        axes,
        norm,
    )
}

/// Build a complex-to-real n-dimensional FFT over `axes`.
///
/// The last requested axis carries the one-sided Hermitian spectrum.
fn irfftn(
    input: &TracedTensor,
    lengths: Option<&[usize]>,
    axes: &[isize],
    norm: FftNorm,
) -> Result<TracedTensor> {
    require_runtime_dtype(
        "irfftn",
        input.dtype,
        &[DType::C32, DType::C64],
        "C32 or C64",
    )?;
    apply_unary_fftn(
        "irfftn",
        input,
        FftOperation::C2r,
        lengths,
        "shape",
        axes,
        norm,
    )
}

fn apply_unary_fftn(
    op_name: &'static str,
    input: &TracedTensor,
    operation: FftOperation,
    lengths: Option<&[usize]>,
    length_argument: &'static str,
    axes: &[isize],
    norm: FftNorm,
) -> Result<TracedTensor> {
    let concrete_shape = input.try_concrete_shape();
    let op = Arc::new(
        validated_fft_op(
            op_name,
            operation,
            Some(input.dtype),
            input.rank,
            concrete_shape.as_deref(),
            lengths,
            length_argument,
            axes,
            norm,
        )
        .map_err(graph_build_fft_error)?,
    );
    let mut outputs = apply(op, &[input])?;
    outputs
        .pop()
        .ok_or_else(|| Error::Internal("FFT extension declares exactly one output".into()))
}

/// Re-raise a rejected traced or eager request as a graph-build validation error.
///
/// The request validator is shared with the concrete path and reports
/// `tenferro_tensor` errors; the graph-building surfaces report the same
/// validation failure at graph-build time.
fn graph_build_fft_error(error: tenferro_tensor::Error) -> Error {
    match error {
        tenferro_tensor::Error::Validation { op, source } => {
            Error::validation(op, ErrorPhase::GraphBuild, source)
        }
        other => Error::from(other),
    }
}

/// Lengths of a one-axis request, so the one-dimensional entry points are the
/// ordinary n-dimensional builder with a single axis.
fn single_axis_lengths(n: Option<usize>) -> Option<[usize; 1]> {
    n.map(|n| [n])
}

fn normalize_axis(op: &'static str, axis: isize, rank: usize) -> tenferro_tensor::Result<usize> {
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

/// Build the one-axis operation descriptor used by the eager builder, which
/// validates the dtype itself.
#[cfg(feature = "autodiff")]
#[allow(clippy::too_many_arguments)]
fn prepare_runtime_fft_op(
    op_name: &'static str,
    operation: FftOperation,
    rank: usize,
    concrete_shape: Option<&[usize]>,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<FftOp> {
    let lengths = single_axis_lengths(n);
    validated_fft_op(
        op_name,
        operation,
        None,
        rank,
        concrete_shape,
        lengths.as_ref().map(|lengths| lengths.as_slice()),
        "n",
        &[axis],
        norm,
    )
    .map_err(graph_build_fft_error)
}

fn runtime_forward_fft_operation(dtype: DType) -> Result<FftOperation> {
    match dtype {
        DType::C32 | DType::C64 => Ok(FftOperation::C2cForward),
        DType::F32 | DType::F64 => Ok(FftOperation::R2cFull),
        DType::I32 | DType::I64 | DType::Bool | DType::External(_) => Err(
            runtime_unsupported_dtype("fft", dtype, "F32, F64, C32, or C64"),
        ),
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
mod nd_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_infer_output_meta_rejects_invalid_trait_inputs_without_panicking() {
        let op = FftOp::new(FftOperation::R2cOnesided, vec![0], None, FftNorm::Backward);
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

        let bad_axis = FftOp::new(FftOperation::C2cForward, vec![2], None, FftNorm::Backward);
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
        let err = cpu::LaneLayout::new(&[usize::MAX, 2], 1, 2)
            .expect_err("lane layout should reject stride overflow");

        assert!(err.to_string().contains("overflows usize"), "{err}");
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn fft_semantic_rules_emit_extension_first_jvp_and_length_restoring_transpose() {
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

        let fft_op = FftOp::new(
            FftOperation::C2cForward,
            vec![0],
            Some(vec![2]),
            FftNorm::Backward,
        );
        let mut source = SemanticProgramBuilder::new();
        let source_input = source
            .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
            .unwrap();
        let source_output = source
            .add_extension(Arc::new(fft_op), &[source_input])
            .unwrap()[0];
        let source = source.finish(&[source_output]).unwrap();
        let operation = source.program.operations().next().unwrap();

        let rules = semantic_ad_rules().unwrap();
        let mut destination = SemanticProgramBuilder::new();
        let primal = destination
            .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
            .unwrap();
        let tangent = destination
            .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
            .unwrap();
        let primal_output = destination
            .add_extension(
                Arc::new(FftOp::new(
                    FftOperation::C2cForward,
                    vec![0],
                    Some(vec![2]),
                    FftNorm::Backward,
                )),
                &[primal],
            )
            .unwrap()[0];
        let linearized = rules
            .linearize_operation(
                operation,
                &[primal],
                &[primal_output],
                &[AdValue::Value(tangent)],
                &[true],
                &mut destination,
            )
            .unwrap();
        let AdValue::Value(tangent_output) = linearized.tangent_outputs()[0] else {
            panic!("FFT tangent must be active");
        };
        let cotangent_inputs = rules
            .linear_transpose_operation(
                operation,
                &[primal],
                &[primal_output],
                &[AdValue::Value(tangent_output)],
                &[true],
                linearized.residuals(),
                &mut destination,
            )
            .unwrap();
        let AdValue::Value(cotangent_input) = cotangent_inputs[0] else {
            panic!("FFT cotangent must be active");
        };
        let frozen = destination
            .finish(&[tangent_output, cotangent_input])
            .unwrap();
        let operations: Vec<_> = frozen.program.operations().collect();
        assert!(
            operations
                .iter()
                .filter(|operation| matches!(operation.op(), SemanticOpRef::Extension(_)))
                .count()
                >= 3
        );
        assert!(operations.iter().any(|operation| matches!(
            operation.op(),
            SemanticOpRef::Core(CoreSemanticOp::DynamicTruncate { axis: 0 })
        )));
        assert!(operations.iter().any(|operation| matches!(
            operation.op(),
            SemanticOpRef::Core(CoreSemanticOp::PadToMatch { axis: 0 })
        )));
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn fft_semantic_rules_run_through_whole_program_jvp_and_vjp() {
        use tenferro_ad::AdContext;
        use tenferro_ops::dim_expr::DimExpr;
        use tenferro_runtime::program::{ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder};

        let mut builder = SemanticProgramBuilder::new();
        let input = builder
            .input(ProgramInputSpec::new(DType::C64, [DimExpr::Const(4)]))
            .unwrap();
        let output = builder
            .add_extension(
                Arc::new(FftOp::new(
                    FftOperation::C2cForward,
                    vec![0],
                    Some(vec![2]),
                    FftNorm::Backward,
                )),
                &[input],
            )
            .unwrap()[0];
        let source = builder.finish(&[output]).unwrap();
        let ad = AdContext::builder()
            .with_semantic_extension_rules(semantic_ad_rules().unwrap())
            .unwrap()
            .build()
            .unwrap();

        let jvp = ad.jvp_program(&source, &[true]).unwrap();
        assert_eq!(jvp.derivative_input_indices(), &[Some(1)]);
        assert!(matches!(
            jvp.frozen().program.operations().last().unwrap().op(),
            SemanticOpRef::Extension(op) if op.family_id() == FFT_EXTENSION_FAMILY_ID
        ));

        let vjp = ad.vjp_program(&source, &[true], &[true]).unwrap();
        assert_eq!(vjp.derivative_output_indices(), &[Some(0)]);
        assert!(vjp.frozen().program.operations().any(|operation| matches!(
            operation.op(),
            SemanticOpRef::Core(CoreSemanticOp::PadToMatch { axis: 0 })
                | SemanticOpRef::Core(CoreSemanticOp::DynamicTruncate { axis: 0 })
        )));
    }
}
