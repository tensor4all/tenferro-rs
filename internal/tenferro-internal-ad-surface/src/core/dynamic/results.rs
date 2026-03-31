use crate::Tensor;
use tenferro_internal_ad_linalg::{DynQrValues, DynSvdValues};

#[derive(Debug)]
pub struct QrResult {
    pub q: Tensor,
    pub r: Tensor,
}

#[derive(Debug)]
pub struct SvdResult {
    pub u: Tensor,
    pub s: Tensor,
    pub vt: Tensor,
}

impl From<DynQrValues> for QrResult {
    fn from(value: DynQrValues) -> Self {
        Self {
            q: Tensor::from_value(value.q),
            r: Tensor::from_value(value.r),
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
