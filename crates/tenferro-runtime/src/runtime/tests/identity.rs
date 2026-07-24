use std::num::NonZeroU64;

use super::super::{RegistrationIdentity, RuntimeEpoch, RuntimeId};

#[test]
fn opaque_nonzero_ids_round_trip_only_inside_the_crate() {
    let runtime_id = RuntimeId::from_nonzero(NonZeroU64::new(7).unwrap());
    let epoch = RuntimeEpoch::from_nonzero(NonZeroU64::new(11).unwrap());

    assert_eq!(runtime_id.get(), NonZeroU64::new(7).unwrap());
    assert_eq!(epoch.get(), NonZeroU64::new(11).unwrap());
    assert_eq!(RuntimeEpoch::one().get(), NonZeroU64::MIN);
}

#[test]
fn runtime_epoch_checked_next_stops_at_nonzero_max() {
    let epoch = RuntimeEpoch::from_nonzero(NonZeroU64::new(41).unwrap());
    let maximum = RuntimeEpoch::from_nonzero(NonZeroU64::MAX);

    assert_eq!(
        epoch.checked_next().unwrap().get(),
        NonZeroU64::new(42).unwrap()
    );
    assert_eq!(maximum.checked_next(), None);
}

#[test]
fn registration_debug_exposes_ordinal_and_never_issuer() {
    let identity =
        RegistrationIdentity::new(NonZeroU64::new(101).unwrap(), NonZeroU64::new(202).unwrap());
    let debug = format!("{identity:?}");

    assert_eq!(identity.ordinal(), NonZeroU64::new(202).unwrap());
    assert!(debug.contains("RegistrationIdentity"));
    assert!(debug.contains("202"));
    assert!(!debug.contains("101"));
    assert!(!debug.contains("issuer"));
}
