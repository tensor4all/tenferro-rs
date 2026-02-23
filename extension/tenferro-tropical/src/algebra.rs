//! Tropical algebra markers, [`HasAlgebra`], and [`Semiring`] implementations.
//!
//! Each zero-sized struct identifies a tropical algebra for use with
//! [`TensorPrims<Alg>`](tenferro_prims::TensorPrims). The orphan rule is
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
/// Used as the algebra parameter `Alg` in
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
// Semiring implementations
//
// The Semiring trait uses an associated type `type Scalar`, so each algebra
// marker can only implement Semiring once (Rust does not allow two impls with
// different associated types for the same struct). We choose f64 as the
// canonical Semiring scalar for constant queries (zero/one/add/mul).
//
// f32 is fully supported at the TensorPrims level: plan() and execute() are
// generic over `T: Scalar`, so MaxPlus<f32>, MinPlus<f32>, and MaxMul<f32>
// work correctly through the standard operator overloads (Add/Mul/Zero/One
// on the scalar wrappers). The Semiring trait is primarily used for querying
// algebraic constants and is not in the critical TensorPrims execution path.
//
// If f32-specific Semiring constants are needed in the future, the options
// are: (a) create separate algebra markers (MaxPlusF32Algebra, etc.), or
// (b) make the Semiring trait generic (Semiring<T>). Both require changes
// to tenferro-algebra and are deferred until a concrete use case arises.
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
        MaxPlus(f64::NEG_INFINITY)
    }

    fn one() -> Self::Scalar {
        MaxPlus(0.0)
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b // max
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a * b // ordinary +
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
        MinPlus(f64::INFINITY)
    }

    fn one() -> Self::Scalar {
        MinPlus(0.0)
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b // min
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a * b // ordinary +
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
        MaxMul(0.0)
    }

    fn one() -> Self::Scalar {
        MaxMul(1.0)
    }

    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a + b // max
    }

    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a * b // ordinary *
    }
}
