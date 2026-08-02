use std::sync::{Arc, OnceLock};

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::apply_eager_with_extension_session;
use tenferro_ad::EagerTensor;
use tenferro_runtime::{ErrorPhase, ExtensionModule};
use tenferro_tensor::DType;

use crate::{
    execute_fft_extension_reads_session, extension_module, prepare_runtime_fft_op,
    require_runtime_dtype, runtime_forward_fft_operation, FftNorm, FftOperation,
};

/// FFT extension methods for [`EagerTensor`].
pub trait EagerTensorFftExt {
    /// Execute a complex FFT, or a full-spectrum FFT for real input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    ///
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ///     EagerRuntime::new().unwrap(),
    /// )?;
    /// let y = x.fft(None, -1, FftNorm::Backward)?;
    /// assert_eq!(y.materialized()?.as_slice::<Complex64>().unwrap()[0], Complex64::new(3.0, 0.0));
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid axis or transform length, an
    /// extension error for an unsupported dtype or backend, and a runtime-state
    /// error when the eager runtime is unavailable.
    fn fft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor>;

    /// Execute an inverse complex FFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    ///
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![2], vec![Complex64::new(3.0, 0.0), Complex64::new(-1.0, 0.0)]).unwrap(),
    ///     EagerRuntime::new().unwrap(),
    /// )?;
    /// let y = x.ifft(None, -1, FftNorm::Backward)?;
    /// assert_eq!(y.shape(), &[2]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid axis or transform length, an
    /// extension error for non-complex input or an unsupported backend, and a
    /// runtime-state error when the eager runtime is unavailable.
    fn ifft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor>;

    /// Execute a one-sided real FFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    ///
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ///     EagerRuntime::new().unwrap(),
    /// )?;
    /// let y = x.rfft(None, -1, FftNorm::Backward)?;
    /// assert_eq!(y.shape(), &[3]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid axis or transform length, an
    /// extension error for non-real input or an unsupported backend, and a
    /// runtime-state error when the eager runtime is unavailable.
    fn rfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor>;

    /// Execute an inverse real FFT from a one-sided complex spectrum.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_fft::{EagerTensorFftExt, FftNorm};
    ///
    /// let x = EagerTensor::from_tensor_in(
    ///     Tensor::from_vec_col_major(vec![3], vec![Complex64::new(10.0, 0.0), Complex64::new(-2.0, 2.0), Complex64::new(-2.0, 0.0)]).unwrap(),
    ///     EagerRuntime::new().unwrap(),
    /// )?;
    /// let y = x.irfft(Some(4), -1, FftNorm::Backward)?;
    /// assert_eq!(y.shape(), &[4]);
    /// # Ok::<(), tenferro_ad::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid axis, transform length, or
    /// spectrum length, an extension error for non-complex input or an
    /// unsupported backend, and a runtime-state error when the eager runtime is
    /// unavailable.
    fn irfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor>;
}

impl EagerTensorFftExt for EagerTensor {
    fn fft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor> {
        let operation = runtime_forward_fft_operation(self.dtype())?;
        apply_eager_fft("fft", self, operation, n, axis, norm)
    }

    fn ifft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor> {
        require_runtime_dtype(
            "ifft",
            self.dtype(),
            &[DType::C32, DType::C64],
            "C32 or C64",
        )?;
        apply_eager_fft("ifft", self, FftOperation::C2cInverse, n, axis, norm)
    }

    fn rfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor> {
        require_runtime_dtype(
            "rfft",
            self.dtype(),
            &[DType::F32, DType::F64],
            "F32 or F64",
        )?;
        apply_eager_fft("rfft", self, FftOperation::R2cOnesided, n, axis, norm)
    }

    fn irfft(&self, n: Option<usize>, axis: isize, norm: FftNorm) -> Result<EagerTensor> {
        require_runtime_dtype(
            "irfft",
            self.dtype(),
            &[DType::C32, DType::C64],
            "C32 or C64",
        )?;
        apply_eager_fft("irfft", self, FftOperation::C2r, n, axis, norm)
    }
}

fn apply_eager_fft(
    op_name: &'static str,
    input: &EagerTensor,
    operation: FftOperation,
    n: Option<usize>,
    axis: isize,
    norm: FftNorm,
) -> Result<EagerTensor> {
    let op = prepare_runtime_fft_op(
        op_name,
        operation,
        input.shape().len(),
        Some(input.shape()),
        n,
        axis,
        norm,
    )?;

    let op = Arc::new(op);
    let execute_op = Arc::clone(&op);
    let module = eager_cpu_extension_module()?;
    let mut outputs =
        apply_eager_with_extension_session(op, &[input], module, move |_op, input_reads, ctx| {
            execute_fft_extension_reads_session(&execute_op, input_reads, ctx)
        })?
        .into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(Error::Internal(
            "FFT eager extension returned an unexpected number of outputs".into(),
        )),
    }
}

fn eager_cpu_extension_module() -> Result<Arc<dyn ExtensionModule>> {
    static MODULE: OnceLock<Arc<dyn ExtensionModule>> = OnceLock::new();
    if let Some(module) = MODULE.get() {
        return Ok(Arc::clone(module));
    }

    let engine_id = tenferro_cpu::runtime_engine_id().map_err(eager_runtime_config_error)?;
    let module = extension_module::<tenferro_cpu::CpuBackend>(engine_id)
        .map_err(eager_runtime_config_error)?;
    let _ = MODULE.set(Arc::clone(&module));
    Ok(MODULE.get().cloned().unwrap_or(module))
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_fft::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}
