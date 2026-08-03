use std::num::NonZeroU64;

use super::*;
use crate::{RegistrationIdentity, RuntimeEpoch, RuntimeId};

fn test_domain() -> EventDomainId {
    EventDomainId::runtime_created_for_test(
        RuntimeId::from_nonzero(NonZeroU64::MIN),
        RuntimeEpoch::from_nonzero(NonZeroU64::MIN),
        RegistrationIdentity::new(NonZeroU64::MIN, NonZeroU64::MIN),
    )
}

#[test]
fn immediate_completion_as_any_exposes_the_completion_object() -> crate::Result<()> {
    let domain = test_domain();
    let mut run = ImmediateEventDomainDriver::new().begin_run(domain)?;
    let mut launch = || Ok(());
    let completion = run.enqueue(&[], &mut launch)?;

    assert_eq!(completion.origin(), domain);
    assert!(completion.as_any().is::<ReadyEventToken>());
    assert_eq!(
        Arc::as_ptr(&completion).cast::<()>(),
        std::ptr::from_ref(completion.as_any()).cast::<()>(),
        "as_any must expose the completion token itself"
    );
    Ok(())
}
