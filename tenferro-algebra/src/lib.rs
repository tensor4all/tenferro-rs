//! Algebra traits for the tenferro workspace.
//!
//! This crate provides the minimal algebra foundation for
//! [`TensorPrims<A>`](tenferro_prims::TensorPrims):
//!
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
//! use tenferro_algebra::{HasAlgebra, Standard};
//!
//! // f64 maps to Standard algebra automatically
//! fn check_algebra<T: HasAlgebra<Algebra = Standard>>() {}
//! check_algebra::<f64>();
//! check_algebra::<f32>();
//! ```

use num_complex::{Complex32, Complex64};
use strided_traits::ScalarBase;

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
    type Scalar: ScalarBase;

    /// Additive identity element.
    fn zero() -> Self::Scalar;

    /// Multiplicative identity element.
    fn one() -> Self::Scalar;

    /// Semiring addition.
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Semiring multiplication.
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
