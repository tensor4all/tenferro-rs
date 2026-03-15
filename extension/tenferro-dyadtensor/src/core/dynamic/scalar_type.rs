use num_complex::{Complex32, Complex64};

/// Runtime scalar type tag used by dynamic tensor wrappers.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::ScalarType;
///
/// assert_eq!(ScalarType::F64, ScalarType::F64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `Complex32`
    C32,
    /// `Complex64`
    C64,
}

pub(super) trait AbsAsF64 {
    fn abs_as_f64(self) -> f64;
}

impl AbsAsF64 for f32 {
    fn abs_as_f64(self) -> f64 {
        self.abs() as f64
    }
}

impl AbsAsF64 for f64 {
    fn abs_as_f64(self) -> f64 {
        self.abs()
    }
}

impl AbsAsF64 for Complex32 {
    fn abs_as_f64(self) -> f64 {
        self.norm() as f64
    }
}

impl AbsAsF64 for Complex64 {
    fn abs_as_f64(self) -> f64 {
        self.norm()
    }
}
