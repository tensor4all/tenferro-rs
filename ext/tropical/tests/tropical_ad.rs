use tenferro_ext_tropical::{MaxPlus, TropicalKind};

#[test]
fn tropical_ad_test_target_exists_without_fake_registration() {
    // The Task 2 skeleton intentionally does not register tropical AD rules yet.
    // Keep this target in place so later AD work replaces it with behavioral
    // coverage instead of inventing a placeholder public API.
    assert_eq!(TropicalKind::MaxPlus, TropicalKind::MaxPlus);
    assert_eq!((MaxPlus(1.0_f64) * MaxPlus(2.0_f64)).value(), 3.0);
}
