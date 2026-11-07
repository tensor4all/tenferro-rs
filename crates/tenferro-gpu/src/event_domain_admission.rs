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

pub(crate) fn admit_event_tokens<'a, T: 'static, R>(
    dependencies: &'a [Arc<dyn EventToken>],
    expected: EventDomainId,
    token_type: &'static str,
    launch: impl FnOnce(&[&'a T]) -> Result<R>,
) -> Result<R> {
    let mut admitted = dependencies
        .iter()
        .map(|dependency| admit_event_token::<T>(dependency.as_ref(), expected, token_type));
    let Some(first) = admitted.next().transpose()? else {
        return launch(&[]);
    };
    // `&T` has no `Default`, so `tinyvec::Array` cannot hold it. It is `Copy`,
    // so up to `INLINE` tokens go into a stack array seeded with the first
    // token; every slot below `dependencies.len()` is overwritten before
    // `launch` reads it. Longer lists spill to a `Vec`, as an inline-capacity
    // vector would.
    const INLINE: usize = 4;
    if dependencies.len() <= INLINE {
        let mut inline = [first; INLINE];
        for (slot, token) in inline[1..].iter_mut().zip(admitted) {
            *slot = token?;
        }
        launch(&inline[..dependencies.len()])
    } else {
        let mut spilled = Vec::with_capacity(dependencies.len());
        spilled.push(first);
        for token in admitted {
            spilled.push(token?);
        }
        launch(&spilled)
    }
}
