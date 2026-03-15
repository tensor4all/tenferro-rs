use num_complex::{Complex32, Complex64};

/// Dynamic scalar value extracted from a rank-0 [`crate::Tensor`].
///
/// This is the canonical scalar-extraction boundary for the dynamic frontend.
/// It preserves the original dtype and never performs implicit casting.
///
/// # Examples
///
/// ```rust
/// use tenferro::{ScalarValue, Tensor};
///
/// let x = Tensor::from_slice(&[2.0_f64], &[]).unwrap();
/// assert_eq!(x.try_scalar_value().unwrap(), ScalarValue::F64(2.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}
