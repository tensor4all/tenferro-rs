use super::*;
use std::hint::black_box;
use tenferro_algebra::Semiring;

#[test]
fn f32_semiring_paths_are_covered_in_crate_unit_tests() {
    assert_eq!(MaxPlusAlgebra::<f32>::zero(), MaxPlus(f32::NEG_INFINITY));
    assert_eq!(MaxPlusAlgebra::<f32>::one(), MaxPlus(0.0));
    assert_eq!(
        MaxPlusAlgebra::<f32>::add(MaxPlus(1.0), MaxPlus(3.0)),
        MaxPlus(3.0)
    );
    assert_eq!(
        MaxPlusAlgebra::<f32>::mul(MaxPlus(1.0), MaxPlus(3.0)),
        MaxPlus(4.0)
    );

    assert_eq!(MinPlusAlgebra::<f32>::zero(), MinPlus(f32::INFINITY));
    assert_eq!(MinPlusAlgebra::<f32>::one(), MinPlus(0.0));
    assert_eq!(
        MinPlusAlgebra::<f32>::add(MinPlus(1.0), MinPlus(3.0)),
        MinPlus(1.0)
    );
    assert_eq!(
        MinPlusAlgebra::<f32>::mul(MinPlus(1.0), MinPlus(3.0)),
        MinPlus(4.0)
    );

    assert_eq!(MaxMulAlgebra::<f32>::zero(), MaxMul(0.0));
    assert_eq!(MaxMulAlgebra::<f32>::one(), MaxMul(1.0));
    assert_eq!(
        MaxMulAlgebra::<f32>::add(MaxMul(0.25), MaxMul(0.75)),
        MaxMul(0.75)
    );
    assert_eq!(
        MaxMulAlgebra::<f32>::mul(MaxMul(0.25), MaxMul(0.75)),
        MaxMul(0.1875)
    );
}

#[test]
fn f64_semiring_paths_are_covered_in_crate_unit_tests() {
    assert_eq!(MaxPlusAlgebra::<f64>::zero(), MaxPlus(f64::NEG_INFINITY));
    assert_eq!(MaxPlusAlgebra::<f64>::one(), MaxPlus(0.0));
    assert_eq!(
        MaxPlusAlgebra::<f64>::add(black_box(MaxPlus(1.0)), black_box(MaxPlus(3.0))),
        MaxPlus(3.0)
    );
    assert_eq!(
        MaxPlusAlgebra::<f64>::mul(black_box(MaxPlus(1.0)), black_box(MaxPlus(3.0))),
        MaxPlus(4.0)
    );

    assert_eq!(MinPlusAlgebra::<f64>::zero(), MinPlus(f64::INFINITY));
    assert_eq!(MinPlusAlgebra::<f64>::one(), MinPlus(0.0));
    assert_eq!(
        MinPlusAlgebra::<f64>::add(black_box(MinPlus(1.0)), black_box(MinPlus(3.0))),
        MinPlus(1.0)
    );
    assert_eq!(
        MinPlusAlgebra::<f64>::mul(black_box(MinPlus(1.0)), black_box(MinPlus(3.0))),
        MinPlus(4.0)
    );

    assert_eq!(MaxMulAlgebra::<f64>::zero(), MaxMul(0.0));
    assert_eq!(MaxMulAlgebra::<f64>::one(), MaxMul(1.0));
    assert_eq!(
        MaxMulAlgebra::<f64>::add(black_box(MaxMul(0.25)), black_box(MaxMul(0.75))),
        MaxMul(0.75)
    );
    assert_eq!(
        MaxMulAlgebra::<f64>::mul(black_box(MaxMul(0.25)), black_box(MaxMul(0.75))),
        MaxMul(0.1875)
    );
}
