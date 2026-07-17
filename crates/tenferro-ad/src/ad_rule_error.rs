use tenferro_runtime::{Error, ErrorPhase};
use tenferro_tensor::ErrorKind;
use tidu::ADRuleError;

/// State used while implementing tidu's message-only callback traits.
///
/// `tidu::ADRuleError` predates tenferro's structured error model and cannot
/// carry a source. Keep the typed runtime error alongside the compatibility
/// value while crossing that external dependency boundary; public tenferro
/// APIs consume `typed` first and therefore never expose the rendered message
/// as the primary error.
#[derive(Default)]
pub(crate) struct DeferredErrors {
    ad_rule: Option<ADRuleError>,
    typed: Option<Error>,
}

impl DeferredErrors {
    pub(crate) fn take_ad_rule(&mut self) -> Option<ADRuleError> {
        self.ad_rule.take()
    }

    pub(crate) fn take_typed(&mut self) -> Option<Error> {
        self.typed.take()
    }

    pub(crate) fn record_ad_rule(&mut self, err: ADRuleError) {
        if self.ad_rule.is_none() {
            self.ad_rule = Some(err);
        }
    }

    pub(crate) fn has_error(&self) -> bool {
        self.ad_rule.is_some() || self.typed.is_some()
    }

    pub(crate) fn runtime(&mut self, transform: &'static str, err: Error) -> ADRuleError {
        let fallback = runtime_error_to_ad_rule(transform, &err);
        if self.typed.is_none() {
            self.typed = Some(err);
        }
        self.record_ad_rule(fallback.clone());
        fallback
    }
}

/// Convert only at the external tidu callback boundary.
fn runtime_error_to_ad_rule(transform: &'static str, err: &Error) -> ADRuleError {
    match err.kind() {
        ErrorKind::Unsupported => ADRuleError::unsupported(transform, tidu::ADRuleKind::Transpose),
        _ => ADRuleError::invalid_input(transform, tidu::ADRuleKind::Transpose, err.to_string()),
    }
}

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
