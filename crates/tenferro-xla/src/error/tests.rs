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
