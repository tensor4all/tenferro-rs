use num_complex::{Complex32, Complex64};
use num_traits::Zero;

use super::super::{DynScalar, ScalarType};
use super::DynAdScalar;
use crate::{AdMode, AdValue, NodeId, TapeId};

impl DynAdScalar {
    /// Creates a real scalar (`f64`) in primal mode.
    pub fn new_real(x: f64) -> Self {
        Self::from(x)
    }

    /// Creates a complex scalar (`Complex64`) in primal mode.
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from(Complex64::new(re, im))
    }

    /// Returns runtime scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns AD mode.
    pub fn mode(&self) -> AdMode {
        match self {
            Self::F32(v) => v.mode(),
            Self::F64(v) => v.mode(),
            Self::C32(v) => v.mode(),
            Self::C64(v) => v.mode(),
        }
    }

    /// Returns reverse-mode node id when available.
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::F32(v) => v.node_id(),
            Self::F64(v) => v.node_id(),
            Self::C32(v) => v.node_id(),
            Self::C64(v) => v.node_id(),
        }
    }

    /// Returns reverse-mode tape id when available.
    pub fn tape_id(&self) -> Option<TapeId> {
        match self {
            Self::F32(v) => v.tape_id(),
            Self::F64(v) => v.tape_id(),
            Self::C32(v) => v.tape_id(),
            Self::C64(v) => v.tape_id(),
        }
    }

    /// Returns primal part as dynamic scalar.
    pub fn primal(&self) -> DynScalar {
        match self {
            Self::F32(v) => DynScalar::F32(*v.primal_ref()),
            Self::F64(v) => DynScalar::F64(*v.primal_ref()),
            Self::C32(v) => DynScalar::C32(*v.primal_ref()),
            Self::C64(v) => DynScalar::C64(*v.primal_ref()),
        }
    }

    /// Consumes this scalar and returns the primal value, explicitly dropping AD metadata.
    pub fn primal_into(self) -> DynScalar {
        match self {
            Self::F32(v) => DynScalar::F32(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::F64(v) => DynScalar::F64(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::C32(v) => DynScalar::C32(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
            Self::C64(v) => DynScalar::C64(match v {
                AdValue::Primal(primal) => primal,
                AdValue::Forward { primal, .. } => primal,
                AdValue::Reverse { primal, .. } => primal,
            }),
        }
    }

    /// Returns tangent part as dynamic scalar when available.
    pub fn tangent(&self) -> Option<DynScalar> {
        match self {
            Self::F32(v) => v.tangent_ref().copied().map(DynScalar::F32),
            Self::F64(v) => v.tangent_ref().copied().map(DynScalar::F64),
            Self::C32(v) => v.tangent_ref().copied().map(DynScalar::C32),
            Self::C64(v) => v.tangent_ref().copied().map(DynScalar::C64),
        }
    }

    /// Returns the primal value while intentionally dropping AD metadata.
    pub fn detach(&self) -> DynScalar {
        self.primal()
    }

    /// Returns typed AD value ref when dtype is `f32`.
    pub fn as_f32(&self) -> Option<&AdValue<f32>> {
        if let Self::F32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `f64`.
    pub fn as_f64(&self) -> Option<&AdValue<f64>> {
        if let Self::F64(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<&AdValue<Complex32>> {
        if let Self::C32(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns typed AD value ref when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<&AdValue<Complex64>> {
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

    /// Extracts the real part as `f64`, intentionally dropping AD metadata.
    pub fn real(&self) -> f64 {
        match self {
            Self::F32(v) => *v.primal_ref() as f64,
            Self::F64(v) => *v.primal_ref(),
            Self::C32(v) => v.primal_ref().re as f64,
            Self::C64(v) => v.primal_ref().re,
        }
    }

    /// Extracts the imaginary part as `f64`, intentionally dropping AD metadata.
    pub fn imag(&self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) => 0.0,
            Self::C32(v) => v.primal_ref().im as f64,
            Self::C64(v) => v.primal_ref().im,
        }
    }

    /// Extracts the magnitude as `f64`, intentionally dropping AD metadata.
    pub fn abs(&self) -> f64 {
        match self {
            Self::F32(v) => v.primal_ref().abs() as f64,
            Self::F64(v) => v.primal_ref().abs(),
            Self::C32(v) => v.primal_ref().norm() as f64,
            Self::C64(v) => v.primal_ref().norm(),
        }
    }

    /// Returns true when the primal scalar value is zero.
    pub fn is_zero(&self) -> bool {
        match self {
            Self::F32(v) => v.primal_ref().is_zero(),
            Self::F64(v) => v.primal_ref().is_zero(),
            Self::C32(v) => v.primal_ref().is_zero(),
            Self::C64(v) => v.primal_ref().is_zero(),
        }
    }
}
