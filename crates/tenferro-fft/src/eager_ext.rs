use std::sync::Arc;

use tenferro_ad::error::{Error, Result};
use tenferro_ad::extension::{
    apply_eager_with_targeted_extension_in_session, EagerExtensionBackendKind, EagerExtensionTarget,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaEngine;
use tenferro_runtime::{ErrorPhase, ExtensionModule};

use crate::{extension_module, prepare_runtime_fft_op, FftNorm, FftOperation};

pub(crate) fn apply_eager_fft_in_session(
    session: &mut EagerSession<'_>,
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
    apply_eager_fft_op_in_session(session, op, input)
}

/// Run one validated FFT extension op on `input` in the borrowed session.
pub(crate) fn apply_eager_fft_op_in_session(
    session: &mut EagerSession<'_>,
    op: crate::FftOp,
    input: &EagerTensor,
) -> Result<EagerTensor> {
    let mut outputs = apply_eager_with_targeted_extension_in_session(
        session,
        Arc::new(op),
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
    }
}

fn eager_runtime_config_error(source: tenferro_runtime::RuntimeConfigError) -> Error {
    Error::runtime_state_source(
        "tenferro_fft::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}
