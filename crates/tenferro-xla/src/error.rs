use std::error::Error as StdError;
use std::path::PathBuf;

use tenferro_tensor::{DType, ErrorKind, ValidationKind};

/// Erased typed source used only at the XLA plugin boundary.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// Error type for StableHLO lowering and runtime PJRT plugin loading.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
/// use tenferro_xla::{lower_compiled_to_stablehlo, Error};
///
/// let x = TracedTensor::input_symbolic_shape(DType::I64, 1).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler
///     .compile_with_input_specs(&y, &[(&x, DType::I64, &[2])])
///     .unwrap();
/// let err = lower_compiled_to_stablehlo(&program).unwrap_err();
/// assert!(matches!(err, Error::UnsupportedDType { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("XLA lowering does not support dtype {dtype:?} in {context}")]
    UnsupportedDType { dtype: DType, context: &'static str },
    #[error("XLA lowering does not support ExecOp::{op}: {reason}")]
    UnsupportedOp {
        op: &'static str,
        reason: &'static str,
    },
    #[error(
        "XLA lowering supports only exact static shapes; ExecOp::{op} output {output_index} axis {axis} is {kind}"
    )]
    NonStaticShape {
        op: &'static str,
        output_index: usize,
        axis: usize,
        kind: &'static str,
    },
    #[error("invalid XLA program: {message}")]
    InvalidProgram { message: String },
    #[error("XLA tensor input/output error: {0}")]
    Tensor(#[from] tenferro_tensor::Error),
    #[error("XLA extension standard-op lowering failed: {source}")]
    ExtensionLowering {
        #[source]
        source: tenferro_ops::ext_op::ExtensionLoweringError,
    },
    #[error("PJRT support requires enabling the tenferro-xla `pjrt` feature")]
    PjrtFeatureDisabled,
    #[error("PJRT execution requires an executor created from a loaded plugin")]
    PjrtPluginNotLoaded,
    #[error("PJRT call {call} failed: {message}")]
    PjrtCall { call: &'static str, message: String },
    #[error("environment variable {var} is not set; set it to a PJRT plugin .so path")]
    MissingEnv { var: &'static str },
    #[error("failed to load PJRT plugin from {path}: {source}")]
    PluginLoad {
        path: PathBuf,
        #[source]
        source: BoxError,
    },
}

impl Error {
    /// Return the stable coarse classification for this XLA failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::ErrorKind;
    /// use tenferro_xla::Error;
    ///
    /// assert_eq!(
    ///     Error::UnsupportedOp { op: "Maximum", reason: "not lowered" }.kind(),
    ///     ErrorKind::Unsupported,
    /// );
    /// ```
    #[must_use]
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::UnsupportedDType { .. } | Self::UnsupportedOp { .. } => ErrorKind::Unsupported,
            Self::NonStaticShape { .. } => ErrorKind::Validation(ValidationKind::ShapeMismatch),
            Self::InvalidProgram { .. } => ErrorKind::Validation(ValidationKind::InvalidArgument),
            Self::Tensor(source) => source.kind(),
            Self::ExtensionLowering { source } => source.kind(),
            Self::PjrtFeatureDisabled | Self::PjrtPluginNotLoaded | Self::MissingEnv { .. } => {
                ErrorKind::RuntimeState
            }
            Self::PluginLoad { .. } => ErrorKind::Io,
            Self::PjrtCall { .. } => ErrorKind::BackendFailure,
        }
    }
}

/// Result alias for `tenferro-xla`.
///
/// # Examples
///
/// ```
/// use tenferro_xla::Result;
///
/// fn ok() -> Result<()> {
///     Ok(())
/// }
///
/// ok().unwrap();
/// ```
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests;
