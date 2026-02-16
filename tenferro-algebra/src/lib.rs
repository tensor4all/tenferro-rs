//! Algebra traits for the tenferro workspace.
//!
//! This crate provides the minimal algebra foundation:
//!
//! - [`Scalar`]: Minimum requirements for tensor element types
//!   (`Copy + Send + Sync + Add + Mul + Zero + One + PartialEq`).
//! - [`Conjugate`]: Complex conjugation (identity for real types).
//! - [`HasAlgebra`]: Maps a scalar type `T` to its default algebra `A`.
//!   Enables automatic inference: `Tensor<f64>` → `Standard`,
//!   `Tensor<MaxPlus<f64>>` → `MaxPlus` (in external crate).
//! - [`Semiring`]: Defines zero, one, add, mul for algebra-generic operations.
//! - [`Standard`]: Standard arithmetic algebra (add = `+`, mul = `*`).
//!
//! # Extensibility
//!
//! External crates define new algebras by implementing `HasAlgebra` for their
//! scalar types and `TensorPrims<MyAlgebra>` for `CpuBackend` (orphan rule
//! compatible). For example, `tenferro-tropical` defines `MaxPlus<T>`.
//!
//! # Examples
//!
//! ```
//! use tenferro_algebra::{HasAlgebra, Scalar, Standard};
//!
//! // f64 maps to Standard algebra automatically
//! fn check_algebra<T: HasAlgebra<Algebra = Standard>>() {}
//! check_algebra::<f64>();
//! check_algebra::<f32>();
//!
//! // Scalar is automatically implemented for numeric types
//! fn needs_scalar<T: Scalar>() {}
//! needs_scalar::<f64>();
//! needs_scalar::<f32>();
//! ```

use num_complex::{Complex32, Complex64};

/// Scalar element type for tensors.
///
/// Minimum requirements for a type to be stored in a `Tensor<T>`.
/// All standard numeric types (`f32`, `f64`, `Complex32`, `Complex64`)
/// satisfy this trait automatically via the blanket implementation.
///
/// # Examples
///
/// ```
/// use tenferro_algebra::Scalar;
///
/// fn needs_scalar<T: Scalar>() {}
/// needs_scalar::<f64>();
/// needs_scalar::<f32>();
/// ```
pub trait Scalar:
    Copy
    + Send
    + Sync
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + num_traits::Zero
    + num_traits::One
    + PartialEq
{
}

impl<T> Scalar for T where
    T: Copy
        + Send
        + Sync
        + std::ops::Add<Output = Self>
        + std::ops::Mul<Output = Self>
        + num_traits::Zero
        + num_traits::One
        + PartialEq
{
}

/// Complex conjugation for tensor element types.
///
/// Default implementation returns `self` unchanged, which is correct
/// for real-valued types. Complex types override with actual conjugation.
///
/// # Examples
///
/// ```
/// use tenferro_algebra::Conjugate;
///
/// // Real types: conj is identity
/// assert_eq!(3.14_f64.conj(), 3.14_f64);
///
/// // Complex types: conj negates imaginary part
/// use num_complex::Complex64;
/// let z = Complex64::new(1.0, 2.0);
/// assert_eq!(z.conj(), Complex64::new(1.0, -2.0));
/// ```
pub trait Conjugate: Copy {
    /// Return the complex conjugate of this value.
    fn conj(self) -> Self {
        self
    }
}

impl Conjugate for f32 {}
impl Conjugate for f64 {}

impl Conjugate for Complex32 {
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }
}

impl Conjugate for Complex64 {
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
}

/// Maps a scalar type `T` to its default algebra `A`.
///
/// Enables automatic algebra inference: `Tensor<f64>` → `Standard`,
/// `Tensor<MaxPlus<f64>>` → `MaxPlus` (in external crate).
///
/// # Implementing for custom types
///
/// ```ignore
/// struct MyScalar(f64);
/// struct MyAlgebra;
///
/// impl HasAlgebra for MyScalar {
///     type Algebra = MyAlgebra;
/// }
/// ```
pub trait HasAlgebra {
    /// The algebra associated with this scalar type.
    type Algebra;
}

/// Standard arithmetic algebra (add = `+`, mul = `*`).
///
/// This is the default algebra for built-in numeric types (`f32`, `f64`,
/// `Complex32`, `Complex64`).
pub struct Standard;

impl HasAlgebra for f32 {
    type Algebra = Standard;
}

impl HasAlgebra for f64 {
    type Algebra = Standard;
}

impl HasAlgebra for Complex32 {
    type Algebra = Standard;
}

impl HasAlgebra for Complex64 {
    type Algebra = Standard;
}

/// Semiring trait for algebra-generic operations.
///
/// Defines the four fundamental operations needed for tensor contractions
/// under a given algebra:
///
/// - `zero()`: Additive identity
/// - `one()`: Multiplicative identity
/// - `add(a, b)`: Semiring addition (e.g., `+` for Standard, `max` for MaxPlus)
/// - `mul(a, b)`: Semiring multiplication (e.g., `*` for Standard, `+` for MaxPlus)
///
/// # Examples
///
/// Standard arithmetic:
/// - `zero() = 0`, `one() = 1`, `add = +`, `mul = *`
///
/// Tropical (MaxPlus) semiring (in external crate):
/// - `zero() = -∞`, `one() = 0`, `add = max`, `mul = +`
pub trait Semiring {
    /// The scalar type for this semiring.
    type Scalar: Scalar;

    /// Additive identity element.
    fn zero() -> Self::Scalar;

    /// Multiplicative identity element.
    fn one() -> Self::Scalar;

    /// Semiring addition.
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Semiring multiplication.
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
