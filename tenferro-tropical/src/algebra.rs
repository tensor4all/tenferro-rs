//! Tropical algebra markers, [`HasAlgebra`], and [`Semiring`] implementations.
//!
//! Each zero-sized struct identifies a tropical algebra for use with
//! [`TensorPrims<A>`](tenferro_prims::TensorPrims). The orphan rule is
//! satisfied because the algebra markers are defined in this crate.
//!
//! | Algebra marker | Scalar wrapper | ⊕ | ⊗ |
//! |----------------|---------------|---|---|
//! | [`MaxPlusAlgebra`] | [`MaxPlus<T>`](crate::MaxPlus) | max | + |
//! | [`MinPlusAlgebra`] | [`MinPlus<T>`](crate::MinPlus) | min | + |
//! | [`MaxMulAlgebra`] | [`MaxMul<T>`](crate::MaxMul) | max | × |

use tenferro_algebra::{HasAlgebra, Semiring};

use crate::scalar::{MaxMul, MaxPlus, MinPlus};

/// Algebra marker for the max-plus tropical semiring (⊕ = max, ⊗ = +).
///
/// Used as the algebra parameter `A` in
/// [`TensorPrims<MaxPlusAlgebra>`](tenferro_prims::TensorPrims).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlusAlgebra;
/// use tenferro_prims::{CpuBackend, TensorPrims};
///
/// // Check extension support
/// let has_contract = CpuBackend::has_extension_for::<f64>(
///     tenferro_prims::Extension::Contract,
/// );
/// ```
pub struct MaxPlusAlgebra;

/// Algebra marker for the min-plus tropical semiring (⊕ = min, ⊗ = +).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MinPlusAlgebra;
/// use tenferro_prims::{CpuBackend, TensorPrims};
///
/// let has_contract = CpuBackend::has_extension_for::<f64>(
///     tenferro_prims::Extension::Contract,
/// );
/// ```
pub struct MinPlusAlgebra;

/// Algebra marker for the max-times tropical semiring (⊕ = max, ⊗ = ×).
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxMulAlgebra;
/// use tenferro_prims::{CpuBackend, TensorPrims};
///
/// let has_contract = CpuBackend::has_extension_for::<f64>(
///     tenferro_prims::Extension::Contract,
/// );
/// ```
pub struct MaxMulAlgebra;

// ---------------------------------------------------------------------------
// HasAlgebra: scalar → algebra mapping
// ---------------------------------------------------------------------------

impl HasAlgebra for MaxPlus<f32> {
    type Algebra = MaxPlusAlgebra;
}

impl HasAlgebra for MaxPlus<f64> {
    type Algebra = MaxPlusAlgebra;
}

impl HasAlgebra for MinPlus<f32> {
    type Algebra = MinPlusAlgebra;
}

impl HasAlgebra for MinPlus<f64> {
    type Algebra = MinPlusAlgebra;
}

impl HasAlgebra for MaxMul<f32> {
    type Algebra = MaxMulAlgebra;
}

impl HasAlgebra for MaxMul<f64> {
    type Algebra = MaxMulAlgebra;
}

// ---------------------------------------------------------------------------
// Semiring implementations (f64 only for POC)
// ---------------------------------------------------------------------------

/// Max-plus semiring over `MaxPlus<f64>`.
///
/// - `zero()` = MaxPlus(−∞)
/// - `one()` = MaxPlus(0.0)
/// - `add(a, b)` = max(a, b)
/// - `mul(a, b)` = a + b (ordinary addition)
impl Semiring for MaxPlusAlgebra {
    type Scalar = MaxPlus<f64>;

    fn zero() -> Self::Scalar {
        todo!()
    }

    fn one() -> Self::Scalar {
        todo!()
    }

    fn add(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }

    fn mul(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }
}

/// Min-plus semiring over `MinPlus<f64>`.
///
/// - `zero()` = MinPlus(+∞)
/// - `one()` = MinPlus(0.0)
/// - `add(a, b)` = min(a, b)
/// - `mul(a, b)` = a + b (ordinary addition)
impl Semiring for MinPlusAlgebra {
    type Scalar = MinPlus<f64>;

    fn zero() -> Self::Scalar {
        todo!()
    }

    fn one() -> Self::Scalar {
        todo!()
    }

    fn add(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }

    fn mul(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }
}

/// Max-times semiring over `MaxMul<f64>`.
///
/// - `zero()` = MaxMul(0.0)
/// - `one()` = MaxMul(1.0)
/// - `add(a, b)` = max(a, b)
/// - `mul(a, b)` = a × b (ordinary multiplication)
impl Semiring for MaxMulAlgebra {
    type Scalar = MaxMul<f64>;

    fn zero() -> Self::Scalar {
        todo!()
    }

    fn one() -> Self::Scalar {
        todo!()
    }

    fn add(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }

    fn mul(_a: Self::Scalar, _b: Self::Scalar) -> Self::Scalar {
        todo!()
    }
}
