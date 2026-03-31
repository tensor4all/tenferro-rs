use crate::Tensor;
use tenferro_internal_ad_linalg::{
    DynEigValues, DynEigenValues, DynLstsqValues, DynLuValues, DynQrValues, DynSlogdetValues,
    DynSvdValues,
};

#[derive(Debug)]
pub struct QrResult {
    pub q: Tensor,
    pub r: Tensor,
}

#[derive(Debug)]
pub struct LstsqResult {
    pub x: Tensor,
    pub residual: Tensor,
}

#[derive(Debug)]
pub struct LuResult {
    pub p: Tensor,
    pub l: Tensor,
    pub u: Tensor,
}

#[derive(Debug)]
pub struct EigResult {
    pub values: Tensor,
    pub vectors: Tensor,
}

#[derive(Debug)]
pub struct EigenResult {
    pub values: Tensor,
    pub vectors: Tensor,
}

#[derive(Debug)]
pub struct SvdResult {
    pub u: Tensor,
    pub s: Tensor,
    pub vt: Tensor,
}

#[derive(Debug)]
pub struct SlogdetResult {
    pub sign: Tensor,
    pub logabsdet: Tensor,
}

impl From<DynQrValues> for QrResult {
    fn from(value: DynQrValues) -> Self {
        Self {
            q: Tensor::from_value(value.q),
            r: Tensor::from_value(value.r),
        }
    }
}

impl From<DynLstsqValues> for LstsqResult {
    fn from(value: DynLstsqValues) -> Self {
        Self {
            x: Tensor::from_value(value.x),
            residual: Tensor::from_value(value.residual),
        }
    }
}

impl From<DynLuValues> for LuResult {
    fn from(value: DynLuValues) -> Self {
        Self {
            p: Tensor::from_value(value.p),
            l: Tensor::from_value(value.l),
            u: Tensor::from_value(value.u),
        }
    }
}

impl From<DynEigValues> for EigResult {
    fn from(value: DynEigValues) -> Self {
        Self {
            values: Tensor::from_value(value.values),
            vectors: Tensor::from_value(value.vectors),
        }
    }
}

impl From<DynEigenValues> for EigenResult {
    fn from(value: DynEigenValues) -> Self {
        Self {
            values: Tensor::from_value(value.values),
            vectors: Tensor::from_value(value.vectors),
        }
    }
}

impl From<DynSvdValues> for SvdResult {
    fn from(value: DynSvdValues) -> Self {
        Self {
            u: Tensor::from_value(value.u),
            s: Tensor::from_value(value.s),
            vt: Tensor::from_value(value.vt),
        }
    }
}

impl From<DynSlogdetValues> for SlogdetResult {
    fn from(value: DynSlogdetValues) -> Self {
        Self {
            sign: Tensor::from_value(value.sign),
            logabsdet: Tensor::from_value(value.logabsdet),
        }
    }
}
