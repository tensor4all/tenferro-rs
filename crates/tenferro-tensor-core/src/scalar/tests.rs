use crate::{ad_admission, AdAdmissionError, Scalar, ScalarArithmetic, ScalarDomain};
use num_complex::{Complex32, Complex64};

#[test]
fn every_preset_scalar_declares_its_algebra() {
    assert_eq!(<f32 as Scalar>::DOMAIN, ScalarDomain::Field);
    assert_eq!(<f64 as Scalar>::DOMAIN, ScalarDomain::Field);
    assert_eq!(<i32 as Scalar>::DOMAIN, ScalarDomain::Field);
    assert_eq!(<i64 as Scalar>::DOMAIN, ScalarDomain::Field);
    assert_eq!(<bool as Scalar>::DOMAIN, ScalarDomain::NonField);
    assert_eq!(<Complex32 as Scalar>::DOMAIN, ScalarDomain::Field);
    assert_eq!(<Complex64 as Scalar>::DOMAIN, ScalarDomain::Field);
}

#[test]
fn float_arithmetic_uses_the_ordinary_rules() {
    assert_eq!(<f32 as ScalarArithmetic>::scalar_zero(), 0.0);
    assert_eq!(<f32 as ScalarArithmetic>::scalar_one(), 1.0);
    assert_eq!(<f32 as ScalarArithmetic>::scalar_add(1.0, 2.0), 3.0);
    assert_eq!(<f32 as ScalarArithmetic>::scalar_sub(1.0, 2.0), -1.0);
    assert_eq!(<f32 as ScalarArithmetic>::scalar_mul(3.0, 4.0), 12.0);
}

#[test]
fn complex_arithmetic_stays_in_the_complex_field() {
    let lhs = Complex64::new(1.0, 2.0);
    let rhs = Complex64::new(3.0, 4.0);

    assert_eq!(
        <Complex64 as ScalarArithmetic>::scalar_zero(),
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        <Complex64 as ScalarArithmetic>::scalar_one(),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(
        <Complex64 as ScalarArithmetic>::scalar_add(lhs, rhs),
        Complex64::new(4.0, 6.0)
    );
    assert_eq!(
        <Complex64 as ScalarArithmetic>::scalar_sub(lhs, rhs),
        Complex64::new(-2.0, -2.0)
    );
    assert_eq!(
        <Complex32 as ScalarArithmetic>::scalar_mul(
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0)
        ),
        Complex32::new(0.0, 1.0)
    );
}

#[test]
fn integer_arithmetic_wraps_like_the_existing_kernels() {
    assert_eq!(<i32 as ScalarArithmetic>::scalar_zero(), 0);
    assert_eq!(<i64 as ScalarArithmetic>::scalar_one(), 1);
    assert_eq!(<i32 as ScalarArithmetic>::scalar_add(i32::MAX, 1), i32::MIN);
    assert_eq!(<i64 as ScalarArithmetic>::scalar_sub(i64::MIN, 1), i64::MAX);
    assert_eq!(<i32 as ScalarArithmetic>::scalar_mul(i32::MAX, 2), -2);
}

#[test]
fn admission_rejects_every_unsupported_request_explicitly() {
    assert_eq!(
        ad_admission::<f64>(0),
        Err(AdAdmissionError::UnsupportedAdOrder { order: 0 })
    );
    assert_eq!(
        ad_admission::<f64>(3),
        Err(AdAdmissionError::UnsupportedAdOrder { order: 3 })
    );
    assert_eq!(
        ad_admission::<bool>(1),
        Err(AdAdmissionError::NonFieldScalar)
    );
    assert_eq!(
        ad_admission::<f64>(1),
        Err(AdAdmissionError::AdRuleUnavailable)
    );
    assert_eq!(
        ad_admission::<i64>(1),
        Err(AdAdmissionError::AdRuleUnavailable)
    );
}
