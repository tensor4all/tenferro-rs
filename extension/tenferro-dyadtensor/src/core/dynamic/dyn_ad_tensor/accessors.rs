use num_complex::{Complex32, Complex64};

use super::Tensor;
use crate::AdTensor;

impl Tensor {
    pub(crate) fn mode(&self) -> crate::core::AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns typed AD tensor ref when dtype is `f32`.
    pub fn as_f32(&self) -> Option<&AdTensor<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `f64`.
    pub fn as_f64(&self) -> Option<&AdTensor<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<&AdTensor<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD tensor ref when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<&AdTensor<Complex64>> {
        if let Self::C64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns true when scalar dtype is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    /// Returns true when scalar dtype is real.
    pub fn is_real(&self) -> bool {
        matches!(self, Self::F32(_) | Self::F64(_))
    }
}
