use num_complex::{Complex32, Complex64};
use tenferro_internal_frontend_core::AbsAsF64;

use super::super::ScalarType;

#[test]
fn scalar_type_abs_as_f64_covers_all_supported_runtime_scalars() {
    assert_eq!((-3.0_f32).abs_as_f64(), 3.0);
    assert_eq!((-4.0_f64).abs_as_f64(), 4.0);
    assert!((Complex32::new(3.0, 4.0).abs_as_f64() - 5.0).abs() < 1e-6);
    assert!((Complex64::new(5.0, 12.0).abs_as_f64() - 13.0).abs() < 1e-12);
}

#[test]
fn scalar_type_variants_cover_real_and_complex_families() {
    let variants = [
        ScalarType::F32,
        ScalarType::F64,
        ScalarType::C32,
        ScalarType::C64,
    ];
    assert_eq!(variants.len(), 4);
}
