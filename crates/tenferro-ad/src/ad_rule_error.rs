use tenferro_runtime::Error;
use tidu::ADRuleError;

pub(crate) fn ad_rule_error(transform: &'static str, err: ADRuleError) -> Error {
    match err {
        ADRuleError::Unsupported { op, .. } => Error::UnsupportedAdRule { transform, op },
        ADRuleError::InvalidInput { op, message, .. } => Error::InvalidGraphBuild {
            op: transform,
            message: format!("{op}: {message}"),
        },
    }
}
