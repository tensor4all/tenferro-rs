//! Algebra traits for the tenferro workspace.
//!
//! This crate provides the minimal algebra foundation:
//!
//! - [`Scalar`]: Minimum requirements for tensor element types
//!   (equivalent to [`ScalarBase`](strided_traits::ScalarBase) from strided-traits).
//! - [`Conjugate`]: Complex conjugation (identity for real types).
//! - [`HasAlgebra`]: Maps a scalar type `T` to its default algebra `Alg`.
//!   Enables automatic inference: `Tensor<f64>` → `Standard<f64>`,
//!   `Tensor<MaxPlus<f64>>` → `MaxPlusAlgebra<f64>` (in external crate).
//!   This is UX sugar — the core model is `Alg::Scalar`-centric.
//! - [`Algebra`]: Associates an algebra marker with its scalar type (`Alg::Scalar`).
//! - [`Semiring`]: Extends `Algebra` with zero, one, add, mul for algebra-generic operations.
//! - [`Standard<T>`](Standard): Typed standard arithmetic algebra (add = `+`, mul = `*`).
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
//! // f64 maps to Standard<f64> algebra automatically
//! fn check_algebra<T: HasAlgebra<Algebra = Standard<T>>>() {}
//! check_algebra::<f64>();
//! check_algebra::<f32>();
//!
//! // Scalar is automatically implemented for numeric types
//! fn needs_scalar<T: Scalar>() {}
//! needs_scalar::<f64>();
//! needs_scalar::<f32>();
//! ```

use std::marker::PhantomData;

use num_complex::{Complex32, Complex64};

/// Scalar element type for tensors.
///
/// Minimum requirements for a type to be stored in a `Tensor<T>`.
/// All standard numeric types (`f32`, `f64`, `Complex32`, `Complex64`)
/// satisfy this trait automatically via the blanket implementation.
///
/// `Scalar` is a supertrait of [`ScalarBase`](strided_traits::ScalarBase),
/// ensuring that any `Scalar` type can be used with strided-rs operations.
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
pub trait Scalar: strided_traits::ScalarBase {}

impl<T: strided_traits::ScalarBase> Scalar for T {}

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

/// Maps a scalar type `T` to its default algebra `Alg`.
///
/// Enables automatic algebra inference: `Tensor<f64>` → `Standard<f64>`,
/// `Tensor<MaxPlus<f64>>` → `MaxPlusAlgebra<f64>` (in external crate).
///
/// This trait is **UX sugar** for default algebra inference. The core
/// algebra model is `Alg::Scalar`-centric (see [`Semiring`]).
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

/// Typed standard arithmetic algebra (add = `+`, mul = `*`).
///
/// The type parameter `T` carries the scalar type, making the algebra
/// `Alg::Scalar`-centric. This is the canonical core model — `HasAlgebra`
/// provides UX sugar for automatic inference (e.g., `f64` → `Standard<f64>`).
///
/// This is the default algebra for built-in numeric types (`f32`, `f64`,
/// `Complex32`, `Complex64`).
///
/// # Examples
///
/// ```
/// use tenferro_algebra::{HasAlgebra, Standard};
///
/// // f64 maps to Standard<f64> algebra automatically
/// fn check_algebra<T: HasAlgebra<Algebra = Standard<T>>>() {}
/// check_algebra::<f64>();
/// check_algebra::<f32>();
/// ```
pub struct Standard<T>(PhantomData<T>);

impl HasAlgebra for f32 {
    type Algebra = Standard<f32>;
}

impl HasAlgebra for f64 {
    type Algebra = Standard<f64>;
}

impl HasAlgebra for Complex32 {
    type Algebra = Standard<Complex32>;
}

impl HasAlgebra for Complex64 {
    type Algebra = Standard<Complex64>;
}

/// Associates an algebra marker with its scalar type.
///
/// This is the minimal trait required by [`TensorPrims`] — it provides
/// the scalar type without requiring semiring operations.
///
/// [`Semiring`] extends `Algebra` with zero/one/add/mul.
///
/// # Examples
///
/// ```
/// use tenferro_algebra::{Algebra, Standard};
///
/// fn needs_algebra<A: Algebra>() {}
/// needs_algebra::<Standard<f64>>();
/// needs_algebra::<Standard<f32>>();
/// ```
pub trait Algebra {
    /// The scalar element type for tensors under this algebra.
    type Scalar: Scalar;
}

/// Semiring trait for algebra-generic operations.
///
/// The algebra type `Alg` carries its scalar type via `Alg::Scalar`. This
/// is the **core algebra model** — `TensorPrims<Alg>` and einsum/linalg
/// trait bounds are centered on `Alg::Scalar`.
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
/// Standard arithmetic (`Standard<f64>`):
/// - `zero() = 0`, `one() = 1`, `add = +`, `mul = *`
///
/// Tropical (MaxPlus) semiring (in external crate):
/// - `zero() = -∞`, `one() = 0`, `add = max`, `mul = +`
pub trait Semiring: Algebra {
    /// Additive identity element.
    fn zero() -> Self::Scalar;

    /// Multiplicative identity element.
    fn one() -> Self::Scalar;

    /// Semiring addition.
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Semiring multiplication.
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}

impl<T: Scalar> Algebra for Standard<T> {
    type Scalar = T;
}

/// Standard arithmetic implements `Semiring` with `+` and `*`.
///
/// # Examples
///
/// ```
/// use tenferro_algebra::{Semiring, Standard};
///
/// let z = <Standard<f64> as Semiring>::zero();
/// let o = <Standard<f64> as Semiring>::one();
/// assert_eq!(z, 0.0);
/// assert_eq!(o, 1.0);
/// ```
impl<T: Scalar + num_traits::Zero + num_traits::One> Semiring for Standard<T> {
    fn zero() -> T {
        T::zero()
    }

    fn one() -> T {
        T::one()
    }

    fn add(a: T, b: T) -> T {
        a + b
    }

    fn mul(a: T, b: T) -> T {
        a * b
    }
}
