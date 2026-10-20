use crate::{FftExecutionCache, FftNorm, FftOperation};
use std::sync::Arc;
use tenferro_ad::extension::{
    adopt_untracked_eager_value, prepare_eager_in_place_input, EagerExtensionBackendKind,
};
use tenferro_ad::{EagerTensor, IntoValueError};
use tenferro_runtime::{Error, ErrorPhase, ExecutionEngine};
use tenferro_tensor::{DType, TensorValue};

/// Failure of a consuming eager FFT.
///
/// Rejected inputs are returned unchanged. After successful ownership acquisition,
/// an execution failure consumes the input; no rollback or hidden copy is made.
///
/// # Examples
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::{EagerFftInPlaceError, EagerTensorFftExt, FftNorm};
/// let mut engine = CpuEngine::serial();
/// let runtime = EagerRuntime::new()?;
/// let input = EagerTensor::from_tensor_in(Tensor::from_vec_col_major([1], vec![1.0_f64])?, runtime)?;
/// match input.fft_in_place(&mut engine, 0, FftNorm::Backward) {
///     Err(EagerFftInPlaceError::Rejected { input, .. }) => assert_eq!(input.shape(), &[1]),
///     other => panic!("real input must be rejected: {other:?}"),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, thiserror::Error)]
pub enum EagerFftInPlaceError {
    /// Validation or ownership acquisition failed before any mutation.
    #[error("in-place FFT input rejected: {source}")]
    Rejected {
        input: Box<EagerTensor>,
        #[source]
        source: Error,
    },
    /// Failure after ownership acquisition; the input has been consumed.
    #[error(transparent)]
    Execution(#[from] Error),
}

/// Drop the rejected input and keep the error, so `?` works in functions that
/// return `tenferro_ad::Result`.
///
/// Match on [`EagerFftInPlaceError::Rejected`] instead when the unchanged input
/// must be recovered.
///
/// # Examples
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::{EagerTensorFftExt, FftNorm};
///
/// fn transform(input: EagerTensor, engine: &mut CpuEngine) -> tenferro_ad::Result<EagerTensor> {
///     Ok(input.fft_in_place(engine, 0, FftNorm::Backward)?)
/// }
///
/// let mut engine = CpuEngine::serial();
/// let runtime = EagerRuntime::new()?;
/// let real = EagerTensor::from_tensor_in(Tensor::from_vec_col_major([1], vec![1.0_f64])?, runtime)?;
/// let error = transform(real, &mut engine).unwrap_err();
/// assert!(error.to_string().contains("C32 or C64"), "{error}");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
impl From<EagerFftInPlaceError> for Error {
    fn from(error: EagerFftInPlaceError) -> Self {
        match error {
            EagerFftInPlaceError::Rejected { source, .. } => source,
            EagerFftInPlaceError::Execution(source) => source,
        }
    }
}

pub(crate) fn apply(
    input: EagerTensor,
    engine: &mut dyn ExecutionEngine,
    operation: FftOperation,
    axis: isize,
    norm: FftNorm,
) -> Result<EagerTensor, EagerFftInPlaceError> {
    let runtime = Arc::clone(input.runtime());
    let mut validate = || -> Result<_, Error> {
        crate::require_runtime_dtype(
            "fft_in_place",
            input.dtype(),
            &[DType::C32, DType::C64],
            "C32 or C64",
        )?;
        let spec = crate::concrete_fft_spec(
            "fft_in_place",
            operation,
            input.dtype(),
            input.shape(),
            None,
            axis,
            norm,
        )?;
        let read = input.tensor_read();
        crate::cpu::validate_host_fft_read_input("fft_in_place", &read)?;
        // Structural ownership extraction preserves layout; it does not turn
        // a lazy transpose into compact storage for the dense lane kernel.
        if !read.is_col_major_contiguous()? {
            return Err(tenferro_tensor::Error::unsupported(
                "fft_in_place", "in-place FFT requires compact column-major input; use the borrowed FFT for strided views",
            ).into());
        }
        prepare_eager_in_place_input(
            &mut *engine,
            &input,
            crate::FFT_EXTENSION_FAMILY_ID,
            |target| {
                if !matches!(target.backend_kind, EagerExtensionBackendKind::Cpu) {
                    return Err(Error::runtime_state(
                        "fft_in_place",
                        ErrorPhase::Execution,
                        "in-place FFT requires a CPU eager runtime",
                    ));
                }
                crate::eager_ext::eager_extension_module(target)
            },
        )?;
        Ok(spec)
    };
    let spec = match validate() {
        Ok(spec) => spec,
        Err(source) => {
            return Err(EagerFftInPlaceError::Rejected {
                input: Box::new(input),
                source,
            })
        }
    };
    let mut tensor = match input.into_value() {
        Ok(tensor) => tensor,
        Err(IntoValueError::NotUnique(input)) => return Err(EagerFftInPlaceError::Rejected {
            input: Box::new(input), source: Error::runtime_state("fft_in_place", ErrorPhase::Execution,
                "the input is retained by another handle or saved value; release the other handle or explicitly duplicate the input"),
        }),
        Err(IntoValueError::Extract { value: input, error }) => return Err(EagerFftInPlaceError::Rejected {
            input: Box::new(input), source: Error::runtime_state_source("fft_in_place", ErrorPhase::Execution, error),
        }),
    };
    // The plans live in the engine's extension caches: take them for the call
    // and give them back whatever the outcome.
    let mut caches = engine.take_extension_caches();
    let outcome = tenferro_cpu::with_cpu_engine(&mut *engine, |cpu| {
        crate::cpu::execute_in_place(
            cpu,
            &mut tensor,
            &spec,
            FftExecutionCache::engine_owned(&mut caches),
        )
    });
    engine.restore_extension_caches(caches);
    outcome
        .ok_or_else(|| {
            Error::runtime_state(
                "fft_in_place",
                ErrorPhase::Execution,
                "in-place FFT requires a CPU engine",
            )
        })?
        .map_err(Error::from)?;
    adopt_untracked_eager_value(runtime, TensorValue::from_tensor(tensor)).map_err(Into::into)
}
