use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use num_complex::{Complex32, Complex64};

use super::super::dyn_ad_tensor::DynAdTensor;
use super::DynAdScalar;
use crate::{AdScalar, AdValue};

macro_rules! impl_dyn_ad_scalar_from_value {
    ($variant:ident, $ty:ty) => {
        impl From<AdValue<$ty>> for DynAdScalar {
            fn from(value: AdValue<$ty>) -> Self {
                Self::$variant(value)
            }
        }

        impl From<$ty> for DynAdScalar {
            fn from(value: $ty) -> Self {
                Self::$variant(AdValue::primal(value))
            }
        }
    };
}

impl_dyn_ad_scalar_from_value!(F32, f32);
impl_dyn_ad_scalar_from_value!(F64, f64);
impl_dyn_ad_scalar_from_value!(C32, Complex32);
impl_dyn_ad_scalar_from_value!(C64, Complex64);

impl Add for DynAdScalar {
    type Output = crate::Result<DynAdScalar>;

    fn add(self, rhs: Self) -> Self::Output {
        self.try_add(rhs)
    }
}

impl Sub for DynAdScalar {
    type Output = crate::Result<DynAdScalar>;

    fn sub(self, rhs: Self) -> Self::Output {
        self.try_sub(rhs)
    }
}

impl Mul for DynAdScalar {
    type Output = crate::Result<DynAdScalar>;

    fn mul(self, rhs: Self) -> Self::Output {
        self.try_mul(rhs)
    }
}

impl Div for DynAdScalar {
    type Output = crate::Result<DynAdScalar>;

    fn div(self, rhs: Self) -> Self::Output {
        self.try_div(rhs)
    }
}

impl Neg for DynAdScalar {
    type Output = DynAdScalar;

    fn neg(self) -> Self::Output {
        match self {
            DynAdScalar::F32(v) => DynAdScalar::F32((-AdScalar::from(v)).into_value()),
            DynAdScalar::F64(v) => DynAdScalar::F64((-AdScalar::from(v)).into_value()),
            DynAdScalar::C32(v) => DynAdScalar::C32((-AdScalar::from(v)).into_value()),
            DynAdScalar::C64(v) => DynAdScalar::C64((-AdScalar::from(v)).into_value()),
        }
    }
}

macro_rules! impl_dynadvalue_scalar_binop {
    ($trait:ident, $method:ident, $scalar:ty) => {
        impl $trait<$scalar> for DynAdScalar {
            type Output = crate::Result<DynAdScalar>;

            fn $method(self, rhs: $scalar) -> Self::Output {
                $trait::$method(self, DynAdScalar::from(rhs))
            }
        }

        impl $trait<DynAdScalar> for $scalar {
            type Output = crate::Result<DynAdScalar>;

            fn $method(self, rhs: DynAdScalar) -> Self::Output {
                $trait::$method(DynAdScalar::from(self), rhs)
            }
        }
    };
}

impl_dynadvalue_scalar_binop!(Add, add, f32);
impl_dynadvalue_scalar_binop!(Add, add, f64);
impl_dynadvalue_scalar_binop!(Add, add, Complex32);
impl_dynadvalue_scalar_binop!(Add, add, Complex64);
impl_dynadvalue_scalar_binop!(Sub, sub, f32);
impl_dynadvalue_scalar_binop!(Sub, sub, f64);
impl_dynadvalue_scalar_binop!(Sub, sub, Complex32);
impl_dynadvalue_scalar_binop!(Sub, sub, Complex64);
impl_dynadvalue_scalar_binop!(Mul, mul, f32);
impl_dynadvalue_scalar_binop!(Mul, mul, f64);
impl_dynadvalue_scalar_binop!(Mul, mul, Complex32);
impl_dynadvalue_scalar_binop!(Mul, mul, Complex64);
impl_dynadvalue_scalar_binop!(Div, div, f32);
impl_dynadvalue_scalar_binop!(Div, div, f64);
impl_dynadvalue_scalar_binop!(Div, div, Complex32);
impl_dynadvalue_scalar_binop!(Div, div, Complex64);

impl TryFrom<DynAdScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: DynAdScalar) -> core::result::Result<Self, Self::Error> {
        match value {
            DynAdScalar::F32(v) => Ok(*v.primal_ref() as f64),
            DynAdScalar::F64(v) => Ok(*v.primal_ref()),
            DynAdScalar::C32(_) | DynAdScalar::C64(_) => {
                Err("Cannot convert complex DynAdScalar to f64")
            }
        }
    }
}

impl From<DynAdScalar> for Complex64 {
    fn from(value: DynAdScalar) -> Self {
        match value {
            DynAdScalar::F32(v) => Complex64::new(*v.primal_ref() as f64, 0.0),
            DynAdScalar::F64(v) => Complex64::new(*v.primal_ref(), 0.0),
            DynAdScalar::C32(v) => {
                let z = v.primal_ref();
                Complex64::new(z.re as f64, z.im as f64)
            }
            DynAdScalar::C64(v) => *v.primal_ref(),
        }
    }
}

impl Default for DynAdScalar {
    fn default() -> Self {
        Self::new_real(0.0)
    }
}

impl PartialOrd for DynAdScalar {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        match (self, other) {
            (DynAdScalar::F32(a), DynAdScalar::F32(b)) => {
                a.primal_ref().partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F64(a), DynAdScalar::F64(b)) => {
                a.primal_ref().partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F32(a), DynAdScalar::F64(b)) => {
                (*a.primal_ref() as f64).partial_cmp(b.primal_ref())
            }
            (DynAdScalar::F64(a), DynAdScalar::F32(b)) => {
                a.primal_ref().partial_cmp(&(*b.primal_ref() as f64))
            }
            _ => None,
        }
    }
}

impl fmt::Display for DynAdScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DynAdScalar::F32(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::F64(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::C32(v) => write!(f, "{}", v.primal_ref()),
            DynAdScalar::C64(v) => write!(f, "{}", v.primal_ref()),
        }
    }
}

impl Mul<&DynAdTensor> for &DynAdScalar {
    type Output = crate::Result<DynAdTensor>;

    fn mul(self, rhs: &DynAdTensor) -> Self::Output {
        rhs.scale(self)
    }
}

impl Div<&DynAdScalar> for &DynAdTensor {
    type Output = crate::Result<DynAdTensor>;

    fn div(self, rhs: &DynAdScalar) -> Self::Output {
        self.div_scalar(rhs)
    }
}
