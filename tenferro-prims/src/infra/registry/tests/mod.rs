use super::*;

#[test]
fn registry_constructor_and_accessors_are_covered_in_crate_unit_tests() {
    let registry = BackendRegistry::new();
    let _cpu = registry.cpu();
    assert!(registry.cuda().is_none());
    assert!(registry.rocm().is_none());

    let default_registry = BackendRegistry::default();
    let _cpu = default_registry.cpu();
    assert!(default_registry.cuda().is_none());
    assert!(default_registry.rocm().is_none());
}
