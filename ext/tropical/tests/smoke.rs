use tenferro_ext_tropical::{MaxPlus, MinPlus, TropicalKind};

#[test]
fn tropical_crate_exports_core_types() {
    assert_eq!(TropicalKind::MaxPlus, TropicalKind::MaxPlus);
    assert_eq!(MaxPlus(2.0_f64).value(), 2.0);
    assert_eq!(MinPlus(3.0_f64).value(), 3.0);
}
