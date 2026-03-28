use num_complex::{Complex32, Complex64};

/// Dynamic scalar value extracted from a rank-0 dynamic tensor.
///
/// This preserves the original dtype and never performs implicit casting.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_frontend_core::ScalarValue;
///
/// assert_eq!(ScalarValue::F64(2.0), ScalarValue::F64(2.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}
