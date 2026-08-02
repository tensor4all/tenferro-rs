use std::sync::Arc;

use tenferro_runtime::runtime::{
    EventDomainError, EventDomainId, EventDomainOperation, EventToken,
};
use tenferro_runtime::{Error as RuntimeError, Result};

pub(crate) fn admit_event_token<'a, T: 'static>(
    dependency: &'a dyn EventToken,
    expected: EventDomainId,
    token_type: &'static str,
) -> Result<&'a T> {
    let actual = dependency.origin();
    if actual != expected {
        return Err(RuntimeError::from(
            EventDomainError::DependencyDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                node_index: None,
                expected,
                actual,
            },
        ));
    }

    dependency.as_any().downcast_ref::<T>().ok_or_else(|| {
        RuntimeError::from(EventDomainError::IncompatibleTokenType {
            operation: EventDomainOperation::Enqueue,
            node_index: None,
            expected,
            actual,
            token_type,
        })
    })
}

pub(crate) fn admit_event_tokens<T: 'static, R>(
    dependencies: &[Arc<dyn EventToken>],
    expected: EventDomainId,
    token_type: &'static str,
    launch: impl FnOnce() -> Result<R>,
) -> Result<R> {
    for dependency in dependencies {
        admit_event_token::<T>(dependency.as_ref(), expected, token_type)?;
    }
    launch()
}
