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
/// use tenferro_xla::{lower_to_stablehlo, Error};
///
/// let x = TracedTensor::input_symbolic_shape(DType::I64, 1).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler
///     .compile_with_input_specs(&y, &[(&x, DType::I64, &[2])])
///     .unwrap();
/// let err = lower_to_stablehlo(program.semantic_program()).unwrap_err();
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
mod tests {
    use std::error::Error as StdError;
    use std::path::PathBuf;

    use tenferro_ops::ext_op::ExtensionLoweringError;
    use tenferro_tensor::{Error as TensorError, ErrorKind, ValidationKind};

    use super::Error;

    #[test]
    fn xla_error_kind_distinguishes_capability_state_io_and_backend_failures() {
        let cases = [
            (
                Error::UnsupportedDType {
                    dtype: tenferro_tensor::DType::I64,
                    context: "constant",
                },
                ErrorKind::Unsupported,
            ),
            (
                Error::UnsupportedOp {
                    op: "Custom",
                    reason: "not lowered",
                },
                ErrorKind::Unsupported,
            ),
            (
                Error::NonStaticShape {
                    op: "Reshape",
                    output_index: 0,
                    axis: 1,
                    kind: "symbolic",
                },
                ErrorKind::Validation(ValidationKind::ShapeMismatch),
            ),
            (
                Error::InvalidProgram {
                    message: "missing output".into(),
                },
                ErrorKind::Validation(ValidationKind::InvalidArgument),
            ),
            (
                Error::Tensor(TensorError::invalid_argument("xla", "input", "invalid")),
                ErrorKind::Validation(ValidationKind::InvalidArgument),
            ),
            (
                Error::ExtensionLowering {
                    source: ExtensionLoweringError::new_with_kind(
                        ErrorKind::Unsupported,
                        "cannot lower",
                    ),
                },
                ErrorKind::Unsupported,
            ),
            (Error::PjrtFeatureDisabled, ErrorKind::RuntimeState),
            (Error::PjrtPluginNotLoaded, ErrorKind::RuntimeState),
            (
                Error::MissingEnv { var: "PJRT_PLUGIN" },
                ErrorKind::RuntimeState,
            ),
            (
                Error::PluginLoad {
                    path: PathBuf::from("plugin.so"),
                    source: Box::new(std::io::Error::other("not found")),
                },
                ErrorKind::Io,
            ),
            (
                Error::PjrtCall {
                    call: "pjrt_execute",
                    message: "invalid status".into(),
                },
                ErrorKind::BackendFailure,
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.kind(), expected, "classified {error:?}");
        }
    }

    #[test]
    fn xla_error_sources_remain_typed_at_boundary() {
        let tensor = Error::Tensor(TensorError::backend_source(
            "xla_input",
            std::io::Error::other("device read failed"),
        ));
        assert!(StdError::source(&tensor).is_some());

        let lowering = Error::ExtensionLowering {
            source: ExtensionLoweringError::from_source_with_kind(
                ErrorKind::BackendFailure,
                std::io::Error::other("shape source"),
            ),
        };
        assert_eq!(lowering.kind(), ErrorKind::BackendFailure);
        let source = StdError::source(&lowering).expect("lowering source should be retained");
        let typed_source = source
            .source()
            .expect("typed lowering source should remain in the chain");
        assert!(typed_source.downcast_ref::<std::io::Error>().is_some());

        let plugin = Error::PluginLoad {
            path: PathBuf::from("plugin.so"),
            source: Box::new(std::io::Error::other("dlopen failed")),
        };
        assert!(StdError::source(&plugin).is_some());
    }

    #[test]
    fn xla_extension_lowering_preserves_non_validation_kinds() {
        for expected in [
            ErrorKind::NumericalFailure,
            ErrorKind::BackendFailure,
            ErrorKind::RuntimeState,
        ] {
            let error = Error::ExtensionLowering {
                source: ExtensionLoweringError::new_with_kind(expected, "typed failure"),
            };
            assert_eq!(error.kind(), expected);
        }
    }
}
