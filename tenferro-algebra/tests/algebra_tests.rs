//! Tests for tenferro-algebra: Scalar blanket, Conjugate, HasAlgebra,
//! Semiring axioms for Standard<T>.

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Conjugate, HasAlgebra, Scalar, Semiring, Standard};

// ============================================================================
// Scalar blanket impl
// ============================================================================

/// Compile-time contract: these types implement Scalar.
/// Also verifies Scalar's supertraits (Copy, Send, Sync, Add, Mul, Zero, One, PartialEq)
/// at runtime by exercising zero/one/arithmetic.
#[test]
fn scalar_contract_f32() {
    fn check<T: Scalar>(z: T, o: T) -> T {
        z + o
    }
    assert_eq!(check(0.0_f32, 1.0_f32), 1.0_f32);
}

#[test]
fn scalar_contract_f64() {
    fn check<T: Scalar>(a: T, b: T) -> T {
        a * b
    }
    assert_eq!(check(3.0_f64, 4.0_f64), 12.0_f64);
}

#[test]
fn scalar_contract_complex32() {
    fn check<T: Scalar>(a: T, b: T) -> T {
        a + b
    }
    let a = Complex32::new(1.0, 2.0);
    let b = Complex32::new(3.0, 4.0);
    assert_eq!(check(a, b), Complex32::new(4.0, 6.0));
}

#[test]
fn scalar_contract_complex64() {
    fn check<T: Scalar>(a: T, b: T) -> T {
        a * b
    }
    let a = Complex64::new(1.0, 2.0);
    let b = Complex64::new(3.0, 4.0);
    // (1+2i)(3+4i) = -5+10i
    assert_eq!(check(a, b), Complex64::new(-5.0, 10.0));
}

// ============================================================================
// Conjugate
// ============================================================================

#[test]
fn conjugate_f32_identity() {
    assert_eq!(1.25_f32.conj(), 1.25_f32);
}

#[test]
fn conjugate_f64_identity() {
    assert_eq!(1.75_f64.conj(), 1.75_f64);
}

#[test]
fn conjugate_complex32() {
    let z = Complex32::new(1.0, 2.0);
    assert_eq!(z.conj(), Complex32::new(1.0, -2.0));
}

#[test]
fn conjugate_complex64() {
    let z = Complex64::new(3.0, -4.0);
    assert_eq!(z.conj(), Complex64::new(3.0, 4.0));
}

#[test]
fn conjugate_real_zero() {
    assert_eq!(0.0_f64.conj(), 0.0_f64);
}

#[test]
fn conjugate_complex_zero() {
    let z = Complex64::new(0.0, 0.0);
    assert_eq!(z.conj(), Complex64::new(0.0, 0.0));
}

#[test]
fn conjugate_involution() {
    let z = Complex64::new(1.0, 2.0);
    assert_eq!(z.conj().conj(), z);
}

// ============================================================================
// HasAlgebra
// ============================================================================

/// Compile-time contract: these types implement HasAlgebra<Algebra = Standard<T>>.
/// Also verify the algebra type is correct at runtime.
#[test]
fn has_algebra_f32() {
    fn check_add<T: HasAlgebra<Algebra = Standard<T>> + Scalar>(a: T, b: T) -> T {
        <Standard<T> as Semiring>::add(a, b)
    }
    assert_eq!(check_add(1.5_f32, 2.5_f32), 4.0_f32);
}

#[test]
fn has_algebra_f64() {
    fn check_mul<T: HasAlgebra<Algebra = Standard<T>> + Scalar>(a: T, b: T) -> T {
        <Standard<T> as Semiring>::mul(a, b)
    }
    assert_eq!(check_mul(3.0_f64, 7.0_f64), 21.0_f64);
}

#[test]
fn has_algebra_complex32() {
    fn check_zero<T: HasAlgebra<Algebra = Standard<T>> + Scalar>() -> T {
        <Standard<T> as Semiring>::zero()
    }
    assert_eq!(check_zero::<Complex32>(), Complex32::new(0.0, 0.0));
}

#[test]
fn has_algebra_complex64() {
    fn check_one<T: HasAlgebra<Algebra = Standard<T>> + Scalar>() -> T {
        <Standard<T> as Semiring>::one()
    }
    assert_eq!(check_one::<Complex64>(), Complex64::new(1.0, 0.0));
}

// ============================================================================
// Semiring: Standard<f64> basic operations
// ============================================================================

#[test]
fn semiring_f64_zero() {
    assert_eq!(<Standard<f64> as Semiring>::zero(), 0.0_f64);
}

#[test]
fn semiring_f64_one() {
    assert_eq!(<Standard<f64> as Semiring>::one(), 1.0_f64);
}

#[test]
fn semiring_f64_add() {
    assert_eq!(<Standard<f64> as Semiring>::add(2.0, 3.0), 5.0);
}

#[test]
fn semiring_f64_mul() {
    assert_eq!(<Standard<f64> as Semiring>::mul(2.0, 3.0), 6.0);
}

// ============================================================================
// Semiring: Standard<f32>
// ============================================================================

#[test]
fn semiring_f32_zero() {
    assert_eq!(<Standard<f32> as Semiring>::zero(), 0.0_f32);
}

#[test]
fn semiring_f32_one() {
    assert_eq!(<Standard<f32> as Semiring>::one(), 1.0_f32);
}

#[test]
fn semiring_f32_add() {
    assert_eq!(<Standard<f32> as Semiring>::add(1.5_f32, 2.5_f32), 4.0_f32);
}

#[test]
fn semiring_f32_mul() {
    assert_eq!(<Standard<f32> as Semiring>::mul(2.0_f32, 3.0_f32), 6.0_f32);
}

// ============================================================================
// Semiring: Standard<Complex64>
// ============================================================================

#[test]
fn semiring_complex64_zero() {
    assert_eq!(
        <Standard<Complex64> as Semiring>::zero(),
        Complex64::new(0.0, 0.0)
    );
}

#[test]
fn semiring_complex64_one() {
    assert_eq!(
        <Standard<Complex64> as Semiring>::one(),
        Complex64::new(1.0, 0.0)
    );
}

#[test]
fn semiring_complex64_add() {
    let a = Complex64::new(1.0, 2.0);
    let b = Complex64::new(3.0, 4.0);
    assert_eq!(
        <Standard<Complex64> as Semiring>::add(a, b),
        Complex64::new(4.0, 6.0)
    );
}

#[test]
fn semiring_complex64_mul() {
    let a = Complex64::new(1.0, 2.0);
    let b = Complex64::new(3.0, 4.0);
    // (1+2i)(3+4i) = 3+4i+6i+8i² = 3+10i-8 = -5+10i
    assert_eq!(
        <Standard<Complex64> as Semiring>::mul(a, b),
        Complex64::new(-5.0, 10.0)
    );
}

// ============================================================================
// Semiring axioms: additive identity
// ============================================================================

#[test]
fn semiring_additive_identity_f64() {
    let z = <Standard<f64> as Semiring>::zero();
    assert_eq!(<Standard<f64> as Semiring>::add(5.0, z), 5.0);
    assert_eq!(<Standard<f64> as Semiring>::add(z, 5.0), 5.0);
}

// ============================================================================
// Semiring axioms: multiplicative identity
// ============================================================================

#[test]
fn semiring_multiplicative_identity_f64() {
    let o = <Standard<f64> as Semiring>::one();
    assert_eq!(<Standard<f64> as Semiring>::mul(7.0, o), 7.0);
    assert_eq!(<Standard<f64> as Semiring>::mul(o, 7.0), 7.0);
}

// ============================================================================
// Semiring axioms: associativity of add
// ============================================================================

#[test]
fn semiring_add_associativity_f64() {
    type S = Standard<f64>;
    let (a, b, c) = (1.0, 2.0, 3.0);
    assert_eq!(S::add(S::add(a, b), c), S::add(a, S::add(b, c)));
}

// ============================================================================
// Semiring axioms: associativity of mul
// ============================================================================

#[test]
fn semiring_mul_associativity_f64() {
    type S = Standard<f64>;
    let (a, b, c) = (2.0, 3.0, 4.0);
    assert_eq!(S::mul(S::mul(a, b), c), S::mul(a, S::mul(b, c)));
}

// ============================================================================
// Semiring axioms: distributivity
// ============================================================================

#[test]
fn semiring_left_distributivity_f64() {
    type S = Standard<f64>;
    let (a, b, c) = (2.0, 3.0, 4.0);
    // a * (b + c) == a*b + a*c
    assert_eq!(S::mul(a, S::add(b, c)), S::add(S::mul(a, b), S::mul(a, c)));
}

#[test]
fn semiring_right_distributivity_f64() {
    type S = Standard<f64>;
    let (a, b, c) = (2.0, 3.0, 4.0);
    // (a + b) * c == a*c + b*c
    assert_eq!(S::mul(S::add(a, b), c), S::add(S::mul(a, c), S::mul(b, c)));
}

// ============================================================================
// Semiring axioms: zero annihilates mul
// ============================================================================

#[test]
fn semiring_zero_annihilates_f64() {
    type S = Standard<f64>;
    let z = S::zero();
    assert_eq!(S::mul(5.0, z), z);
    assert_eq!(S::mul(z, 5.0), z);
}
