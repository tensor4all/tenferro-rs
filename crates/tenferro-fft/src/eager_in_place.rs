use crate::{FftExecutionCache, FftNorm, FftOperation};
use std::sync::Arc;
use tenferro_ad::extension::{
    adopt_untracked_eager_value, prepare_eager_in_place_input, EagerExtensionBackendKind,
};
use tenferro_ad::{EagerTensor, IntoValueError};
use tenferro_runtime::{Error, ErrorPhase};
use tenferro_tensor::{DType, TensorValue};

/// Failure of a consuming eager FFT.
///
/// Rejected inputs are returned unchanged. After successful ownership acquisition,
/// an execution failure consumes the input; no rollback or hidden copy is made.
///
/// # Examples
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_fft::{EagerFftInPlaceError, EagerTensorFftExt, FftNorm};
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1)?)?;
/// let input = EagerTensor::from_tensor_in(Tensor::from_vec_col_major([1], vec![1.0_f64])?, runtime)?;
/// match input.fft_in_place(0, FftNorm::Backward) {
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

pub(crate) fn apply(
    input: EagerTensor,
    operation: FftOperation,
    axis: isize,
    norm: FftNorm,
) -> Result<EagerTensor, EagerFftInPlaceError> {
    let runtime = Arc::clone(input.runtime());
    let validate = || -> Result<_, Error> {
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
        prepare_eager_in_place_input(&input, crate::FFT_EXTENSION_FAMILY_ID, |target| {
            if !matches!(target.backend_kind, EagerExtensionBackendKind::Cpu) {
                return Err(Error::runtime_state(
                    "fft_in_place",
                    ErrorPhase::Execution,
                    "in-place FFT requires a CPU eager runtime",
                ));
            }
            crate::eager_ext::eager_extension_module(target)
        })?;
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
    runtime.with_extension_execution_context(|context| {
        let (session, caches) = context.parts_mut();
        tenferro_cpu::with_cpu_exec_session(session, |session| {
            crate::cpu::execute_in_place(
                session,
                &mut tensor,
                &spec,
                FftExecutionCache::runtime_owned(caches),
            )
        })
        .ok_or_else(|| {
            Error::runtime_state(
                "fft_in_place",
                ErrorPhase::Execution,
                "the runtime does not expose a CPU execution session",
            )
        })?
        .map_err(Error::from)
    })??;
    adopt_untracked_eager_value(runtime, TensorValue::from_tensor(tensor)).map_err(Into::into)
}
