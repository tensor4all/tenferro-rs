use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::{
    apply_eager_with_targeted_extension_in_session, EagerExtensionBackendKind, EagerExtensionTarget,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaEngine;
#[cfg(feature = "webgpu")]
use tenferro_gpu::webgpu::WebGpuBackend;
use tenferro_runtime::{ErrorPhase, ExecutionEngine, ExtensionModule};
use tenferro_tensor::DType;

use crate::{
    extension_module, prepare_runtime_fft_op, require_runtime_dtype, runtime_forward_fft_operation,
    FftNorm, FftOperation,
};

/// Consuming in-place FFT methods for [`EagerTensor`].
/// Ordinary eager FFT operations use [`EagerSessionFftExt`] inside a borrowed
/// `EagerRuntime::with_eager_session` region.
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub trait EagerTensorFftExt {
    /// Consume a uniquely owned, untracked, compact column-major CPU complex
    /// tensor and overwrite it with its FFT. Shape and allocation are preserved; no implicit copy occurs.
    /// A compact slice of a larger plain host allocation keeps that allocation:
    /// only the slice's logical elements are overwritten and the elements outside
    /// it are untouched (#2023). A device-backed or managed owner must be
    /// whole-root, because the provider binding path carries no descriptor offset.
    /// Gradient-tracked inputs and active trace capture are rejected even under no_grad.
    ///
    /// # Errors
    /// Returns `EagerFftInPlaceError::Rejected` with the unchanged input for an
    /// invalid axis, non-complex/device input, noncompact layout, recording
    /// state, or shared owner.
    /// `EagerFftInPlaceError::Execution` consumes the input if execution fails
    /// after ownership acquisition; the operation does not promise rollback.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    /// let mut engine = CpuEngine::serial();
    /// let runtime = EagerRuntime::new()?;
    /// let input = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.), Complex64::new(2., 0.)])?, runtime)?;
    /// let output = input.fft_in_place(&mut engine, 0, FftNorm::Backward)?;
    /// assert_eq!(output.value()?.as_slice::<Complex64>()?, &[Complex64::new(3., 0.), Complex64::new(-1., 0.)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn fft_in_place(
        self,
        engine: &mut dyn ExecutionEngine,
        axis: isize,
        norm: FftNorm,
    ) -> std::result::Result<EagerTensor, crate::EagerFftInPlaceError>;

    /// Consume an untracked, compact column-major CPU C32/C64 tensor and perform a shape-preserving
    /// inverse FFT in its original allocation. Ownership rules match fft_in_place.
    ///
    /// # Errors
    /// Returns `EagerFftInPlaceError::Rejected` with the original input when
    /// validation or exclusive ownership fails, or `Execution` after ownership
    /// acquisition. Active capture and tracked inputs are not supported.
    ///
    /// # Examples
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    /// let mut engine = CpuEngine::serial();
    /// let runtime = EagerRuntime::new()?;
    /// let input = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major([2], vec![Complex64::new(3., 0.), Complex64::new(-1., 0.)])?, runtime)?;
    /// let output = input.ifft_in_place(&mut engine, 0, FftNorm::Backward)?;
    /// assert_eq!(output.value()?.as_slice::<Complex64>()?, &[Complex64::new(1., 0.), Complex64::new(2., 0.)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn ifft_in_place(
        self,
        engine: &mut dyn ExecutionEngine,
        axis: isize,
        norm: FftNorm,
    ) -> std::result::Result<EagerTensor, crate::EagerFftInPlaceError>;
}

impl EagerTensorFftExt for EagerTensor {
    fn fft_in_place(
        self,
        engine: &mut dyn ExecutionEngine,
        axis: isize,
        norm: FftNorm,
    ) -> std::result::Result<EagerTensor, crate::EagerFftInPlaceError> {
        crate::eager_in_place::apply(self, engine, FftOperation::C2cForward, axis, norm)
    }

    fn ifft_in_place(
        self,
        engine: &mut dyn ExecutionEngine,
        axis: isize,
        norm: FftNorm,
    ) -> std::result::Result<EagerTensor, crate::EagerFftInPlaceError> {
        crate::eager_in_place::apply(self, engine, FftOperation::C2cInverse, axis, norm)
    }
}

/// FFT extension methods on a runtime-bound borrowed eager session.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, Tensor};
/// use tenferro_fft::{EagerSessionFftExt, FftNorm};
/// let mut engine = tenferro_cpu::CpuEngine::serial();
/// let ctx = EagerRuntime::new()?;
/// let result = ctx.with_eager_session(&mut engine, |session| {
///     let input = session.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
///     session.fft(&input, None, -1, FftNorm::Backward)
/// })?;
/// assert_eq!(result.shape(), &[2]);
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub trait EagerSessionFftExt {
    /// Perform a complex or full-spectrum real FFT on this borrowed session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_fft::{EagerSessionFftExt, FftNorm};
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut engine, |session| {
    ///     let x = session.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    ///     session.fft(&x, None, -1, FftNorm::Backward)
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<Complex64>()?[0], Complex64::new(3.0, 0.0));
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns typed validation, backend, module-installation or unsupported
    /// executor errors; it never opens a nested eager session.
    fn fft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor>;

    /// Perform an inverse complex FFT on this borrowed session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_fft::{EagerSessionFftExt, FftNorm};
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut engine, |session| {
    ///     let x = session.constant_from(Tensor::from_vec_col_major(vec![2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)])?)?;
    ///     session.ifft(&x, None, -1, FftNorm::Backward)
    /// })?;
    /// assert_eq!(result.value()?.as_slice::<Complex64>()?, &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, unsupported-dtype, module or backend errors.
    fn ifft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor>;

    /// Perform a one-sided real FFT on this borrowed session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_fft::{EagerSessionFftExt, FftNorm};
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut engine, |session| {
    ///     let x = session.constant_from(Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?)?;
    ///     session.rfft(&x, None, -1, FftNorm::Backward)
    /// })?;
    /// assert_eq!(result.shape(), &[3]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, unsupported-dtype, module or backend errors.
    fn rfft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor>;

    /// Perform an inverse real FFT on this borrowed session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, Tensor};
    /// use tenferro_fft::{EagerSessionFftExt, FftNorm};
    /// let mut engine = tenferro_cpu::CpuEngine::serial();
    /// let ctx = EagerRuntime::new()?;
    /// let result = ctx.with_eager_session(&mut engine, |session| {
    ///     let x = session.constant_from(Tensor::from_vec_col_major(vec![3], vec![Complex64::new(10.0, 0.0), Complex64::new(-2.0, 2.0), Complex64::new(-2.0, 0.0)])?)?;
    ///     session.irfft(&x, Some(4), -1, FftNorm::Backward)
    /// })?;
    /// assert_eq!(result.shape(), &[4]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    /// # Errors
    /// Returns typed validation, unsupported-dtype, module or backend errors.
    fn irfft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor>;
}

impl EagerSessionFftExt for EagerSession<'_> {
    fn fft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor> {
        let operation = runtime_forward_fft_operation(input.dtype())?;
        apply_eager_fft_in_session(self, "fft", input, operation, n, axis, norm)
    }

    fn ifft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor> {
        require_runtime_dtype(
            "ifft",
            input.dtype(),
            &[DType::C32, DType::C64],
            "C32 or C64",
        )?;
        apply_eager_fft_in_session(self, "ifft", input, FftOperation::C2cInverse, n, axis, norm)
    }

    fn rfft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor> {
        require_runtime_dtype(
            "rfft",
            input.dtype(),
            &[DType::F32, DType::F64],
            "F32 or F64",
        )?;
        apply_eager_fft_in_session(
            self,
            "rfft",
            input,
            FftOperation::R2cOnesided,
            n,
            axis,
            norm,
        )
    }

    fn irfft(
        &mut self,
        input: &EagerTensor,
        n: Option<usize>,
        axis: isize,
        norm: FftNorm,
    ) -> Result<EagerTensor> {
        require_runtime_dtype(
            "irfft",
            input.dtype(),
            &[DType::C32, DType::C64],
            "C32 or C64",
        )?;
        apply_eager_fft_in_session(self, "irfft", input, FftOperation::C2r, n, axis, norm)
    }
}

fn apply_eager_fft_in_session(
    session: &mut EagerSession<'_>,
    op_name: &'static str,
    input: &EagerTensor,
    operation: FftOperation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<EagerTensor> {
    let op = Arc::new(prepare_runtime_fft_op(
        op_name,
        operation,
        input.shape().len(),
        Some(input.shape()),
        n,
        axis,
        norm,
    )?);
    let mut outputs = apply_eager_with_targeted_extension_in_session(
        session,
        op,
        &[input],
        eager_extension_module,
    )?
    .into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(
            "FFT eager extension returned an unexpected number of outputs".into(),
        )),
    }
}

pub(crate) fn eager_extension_module(
    target: EagerExtensionTarget,
) -> Result<Arc<dyn ExtensionModule>> {
    let EagerExtensionTarget {
        engine_id,
        backend_kind,
    } = target;
    match backend_kind {
        EagerExtensionBackendKind::Cpu => {
            extension_module::<CpuEngine>(engine_id).map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "cuda")]
        EagerExtensionBackendKind::Cuda => {
            extension_module::<CudaEngine>(engine_id).map_err(eager_runtime_config_error)
        }
        // TODO(#2206 webgpu): the WebGPU engine is migrated in a later unit.
        #[cfg(feature = "webgpu")]
        EagerExtensionBackendKind::WebGpu => {
            extension_module::<WebGpuBackend>(engine_id).map_err(eager_runtime_config_error)
        }
    }
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_fft::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}
