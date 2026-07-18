use tenferro_ops::ShapeGuardContext;
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

pub(crate) fn ad_rule_error_with_context(
    transform: &'static str,
    err: ADRuleError,
    ctx: &mut ShapeGuardContext,
) -> Error {
    match ctx.take_deferred_shape_error() {
        Some(source) => Error::ad_rule_source(transform, source),
        None => ad_rule_error(transform, err),
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use computegraph::types::{ValueKey, ValueRef};
    use tenferro_ops::input_key::TensorInputKey;
    use tenferro_ops::std_tensor_op::StdTensorOp;
    use tenferro_ops::{ShapeGuardContext, ShapeGuardError};
    use tenferro_tensor::{ErrorKind, ValidationKind};

    use super::ad_rule_error_with_context;
    use crate::error::Error;

    #[test]
    fn shape_guard_source_survives_the_message_only_ad_boundary() {
        let key = ValueKey::<StdTensorOp>::Input(TensorInputKey::User { id: 901 });
        let value = ValueRef::External(key);
        let mut ctx = ShapeGuardContext::default();
        let callback_error: tidu::ADRuleError = ctx.shape_of(&value).unwrap_err().into();

        let error = ad_rule_error_with_context("jvp", callback_error, &mut ctx);

        assert_eq!(
            error.kind(),
            ErrorKind::Validation(ValidationKind::InvalidArgument)
        );
        assert_eq!(
            error.phase(),
            Some(tenferro_runtime::ErrorPhase::GraphBuild)
        );
        let source = StdError::source(&error).expect("typed AD source should be retained");
        assert!(source
            .downcast_ref::<ShapeGuardError>()
            .is_some_and(|source| matches!(source, ShapeGuardError::MissingMetadata { .. })));
        assert!(matches!(error, Error::AdRuleSource { .. }));
    }

    #[test]
    fn unrelated_ad_rule_errors_keep_their_existing_mapping() {
        let mut ctx = ShapeGuardContext::default();
        let error = ad_rule_error_with_context(
            "jvp",
            tidu::ADRuleError::invalid_input(
                "rule",
                tidu::ADRuleKind::Jvp,
                "invalid configuration",
            ),
            &mut ctx,
        );

        assert!(matches!(error, Error::Validation { .. }));
    }
}
