use chainrules_core::Differentiable;

use crate::StructuredTensor;

use super::DynTensor;

impl Differentiable for DynTensor {
    type Tangent = DynTensor;

    fn zero_tangent(&self) -> Self::Tangent {
        match self {
            Self::F32(value) => Self::F32(value.zero_tangent()),
            Self::F64(value) => Self::F64(value.zero_tangent()),
            Self::C32(value) => Self::C32(value.zero_tangent()),
            Self::C64(value) => Self::C64(value.zero_tangent()),
        }
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        match (a, b) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                Self::F32(StructuredTensor::<f32>::accumulate_tangent(lhs, rhs))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                Self::F64(StructuredTensor::<f64>::accumulate_tangent(lhs, rhs))
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                Self::C32(StructuredTensor::<num_complex::Complex32>::accumulate_tangent(lhs, rhs))
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                Self::C64(StructuredTensor::<num_complex::Complex64>::accumulate_tangent(lhs, rhs))
            }
            (lhs, rhs) => unreachable!(
                "DynTensor::accumulate_tangent requires matching dtypes, got lhs={:?}, rhs={:?}",
                lhs.scalar_type(),
                rhs.scalar_type()
            ),
        }
    }

    fn num_elements(&self) -> usize {
        match self {
            Self::F32(value) => value.num_elements(),
            Self::F64(value) => value.num_elements(),
            Self::C32(value) => value.num_elements(),
            Self::C64(value) => value.num_elements(),
        }
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        match self {
            Self::F32(value) => Self::F32(value.seed_cotangent()),
            Self::F64(value) => Self::F64(value.seed_cotangent()),
            Self::C32(value) => Self::C32(value.seed_cotangent()),
            Self::C64(value) => Self::C64(value.seed_cotangent()),
        }
    }
}
