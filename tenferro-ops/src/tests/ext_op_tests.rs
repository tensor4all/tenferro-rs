//! Coverage tests for `ext_op` registry + validation.

use std::sync::Arc;

use crate::ext_op::{
    is_extension_registered, lookup_extension_factory, register_extension, ExtensionFactory,
    ExtensionRegistryError,
};

#[derive(Debug)]
struct CoverageFamily {
    family: &'static str,
}

impl ExtensionFactory for CoverageFamily {
    fn family_id(&self) -> &'static str {
        self.family
    }
    fn version(&self) -> u32 {
        1
    }
    // `instantiate_default` intentionally left as the default (`None`) to
    // exercise the default-impl body.
}

#[test]
fn default_instantiate_returns_none() {
    let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily {
        family: "covtest.instantiate.v1",
    });
    assert!(factory.instantiate_default().is_none());
}

#[test]
fn register_rejects_malformed_family_ids() {
    // Each case targets a different reject branch in `is_valid_family_id`.
    let cases = [
        "noversion",     // rsplitn(2, '.').next() returns None on second call
        "foo.v1",        // prefix has no '.', split_once returns None
        "foo.bar",       // version segment "bar" fails starts_with('v')
        "foo.bar.v",     // empty digit string after 'v'
        "foo.bar.vabc",  // non-digit version
        ".op.v1",        // empty crate name
        "foo..v1",       // empty op name
        "foo bar.op.v1", // whitespace in crate
        "fooあ.op.v1",   // non-ASCII in crate
    ];
    for bad in cases {
        let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family: bad });
        match register_extension(factory) {
            Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
                assert_eq!(family_id, bad);
            }
            other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
        }
    }
}

#[test]
fn register_and_lookup_roundtrips() {
    let family = "covtest.register_lookup.v1";
    let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    register_extension(factory).expect("first registration should succeed");

    assert!(is_extension_registered(family));
    let looked_up = lookup_extension_factory(family).expect("factory should be registered");
    assert_eq!(looked_up.family_id(), family);
    assert_eq!(looked_up.version(), 1);
}

#[test]
fn register_rejects_duplicate_family_id() {
    let family = "covtest.duplicate.v1";
    let first: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    register_extension(first).expect("first registration should succeed");

    let second: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    match register_extension(second) {
        Err(ExtensionRegistryError::Duplicate { family_id }) => {
            assert_eq!(family_id, family);
        }
        other => panic!("expected Duplicate for {family:?}, got {other:?}"),
    }
}

#[test]
fn lookup_unregistered_family_returns_none() {
    assert!(!is_extension_registered("covtest.absent.v999"));
    assert!(lookup_extension_factory("covtest.absent.v999").is_none());
}
