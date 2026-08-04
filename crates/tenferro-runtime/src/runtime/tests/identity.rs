use std::fmt::Debug;
use std::hash::Hash;
use std::num::NonZeroU64;

use super::super::{
    IdentityKind, ProviderDeviceIdentity, ProviderId, RegistrationIdentity, RuntimeEpoch, RuntimeId,
};

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

#[test]
fn provider_device_identity_validates_and_is_ordered_and_hashable() {
    fn assert_runtime_control_plane_traits<T: Clone + Debug + Eq + Hash + Ord>() {}

    assert_runtime_control_plane_traits::<ProviderId>();
    assert_runtime_control_plane_traits::<ProviderDeviceIdentity>();

    let provider = ProviderId::new("tenferro.test.provider").unwrap();
    let first = ProviderDeviceIdentity::new(provider.clone(), "device-a").unwrap();
    let second = ProviderDeviceIdentity::new(provider.clone(), "device-b").unwrap();

    assert_eq!(first.provider_id(), &provider);
    assert_eq!(first.target_identity(), "device-a");
    assert!(first < second);
    assert_ne!(
        first,
        ProviderDeviceIdentity::new(provider, "device-a/other").unwrap()
    );
    assert_eq!(
        ProviderId::new("not-namespaced").unwrap_err().kind(),
        IdentityKind::Provider
    );
    assert_eq!(
        ProviderDeviceIdentity::new(ProviderId::new("tenferro.test.provider").unwrap(), "",)
            .unwrap_err()
            .kind(),
        IdentityKind::ProviderTarget
    );
    for target in [
        "device 0",
        "device\n0",
        "device\t0",
        "device\u{200b}0",
        "デバイス-0",
    ] {
        assert_eq!(
            ProviderDeviceIdentity::new(
                ProviderId::new("tenferro.test.provider").unwrap(),
                target,
            )
            .unwrap_err()
            .kind(),
            IdentityKind::ProviderTarget,
            "target {target:?} must remain diagnostic-safe"
        );
    }
}
