use tenferro_runtime::{Error, ErrorPhase};
use tidu::ADRuleError;

pub(crate) fn ad_rule_error(transform: &'static str, err: ADRuleError) -> Error {
    match err {
        ADRuleError::Unsupported { op, .. } => Error::UnsupportedAdRule { transform, op },
        ADRuleError::InvalidInput { op, message, .. } => Error::invalid_argument(
            transform,
            ErrorPhase::GraphBuild,
            "ad_input",
            format!("{op}: {message}"),
        ),
    }
}
