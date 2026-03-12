use tenferro_algebra::{HasAlgebra, Scalar, Standard};

/// Trait for extracting the inner float type from a tropical scalar wrapper.
///
/// This enables generic code that operates on the inner values for backward
/// pass computations.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::TropicalScalar;
///
/// let x = MaxPlus(3.0_f64);
/// assert_eq!(x.inner(), 3.0);
/// ```
pub trait TropicalScalar: Scalar {
    /// The inner floating-point type.
    type Inner: Scalar
        + num_traits::Float
        + std::ops::AddAssign
        + HasAlgebra<Algebra = Standard<Self::Inner>>;

    /// Extract the inner value.
    fn inner(&self) -> Self::Inner;

    /// Wrap an inner value into the tropical type.
    fn from_inner(v: Self::Inner) -> Self;

    /// Backward contribution for tropical multiplication w.r.t. the first operand.
    fn mul_backward_a(a_inner: Self::Inner, b_inner: Self::Inner, dout: Self::Inner)
        -> Self::Inner;

    /// Backward contribution for tropical multiplication w.r.t. the second operand.
    fn mul_backward_b(a_inner: Self::Inner, b_inner: Self::Inner, dout: Self::Inner)
        -> Self::Inner;
}

macro_rules! impl_tropical_scalar_additive {
    ($wrapper:ident, $float:ty) => {
        impl TropicalScalar for crate::$wrapper<$float> {
            type Inner = $float;

            fn inner(&self) -> $float {
                self.0
            }

            fn from_inner(v: $float) -> Self {
                crate::$wrapper(v)
            }

            fn mul_backward_a(_a: $float, _b: $float, dout: $float) -> $float {
                dout
            }

            fn mul_backward_b(_a: $float, _b: $float, dout: $float) -> $float {
                dout
            }
        }
    };
}

macro_rules! impl_tropical_scalar_multiplicative {
    ($wrapper:ident, $float:ty) => {
        impl TropicalScalar for crate::$wrapper<$float> {
            type Inner = $float;

            fn inner(&self) -> $float {
                self.0
            }

            fn from_inner(v: $float) -> Self {
                crate::$wrapper(v)
            }

            fn mul_backward_a(_a: $float, b: $float, dout: $float) -> $float {
                dout * b
            }

            fn mul_backward_b(a: $float, _b: $float, dout: $float) -> $float {
                dout * a
            }
        }
    };
}

impl_tropical_scalar_additive!(MaxPlus, f32);
impl_tropical_scalar_additive!(MaxPlus, f64);
impl_tropical_scalar_additive!(MinPlus, f32);
impl_tropical_scalar_additive!(MinPlus, f64);
impl_tropical_scalar_multiplicative!(MaxMul, f32);
impl_tropical_scalar_multiplicative!(MaxMul, f64);
