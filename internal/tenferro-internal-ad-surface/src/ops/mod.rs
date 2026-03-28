use tenferro_algebra::Scalar;
use tenferro_linalg::{
    CholeskyExResult, EigenResult, InvExResult, LinalgScalar, LuFactorExResult, LuFactorResult,
    LuPivot, LuResult, NormKind, QrResult, SolveExResult,
};
use tenferro_tensor::Tensor;

use crate::runtime::contracts::*;
use crate::runtime::dispatch::*;
use crate::{AdTensor, Error, Result};

pub mod ad {
    pub use tenferro_internal_ad_linalg::*;
    pub use tenferro_internal_ad_ops::ad::*;
}

mod linalg;

pub use linalg::*;
