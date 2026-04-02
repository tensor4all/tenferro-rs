use tenferro_algebra::Scalar;
use tenferro_internal_ad_core as tape;
use tenferro_internal_ad_core::ops::*;
use tenferro_internal_ad_core::{AdTensor, DynAdTensorTyped};
use tenferro_internal_error::{Error, Result};
use tenferro_internal_runtime::contracts::*;
use tenferro_internal_runtime::dispatch::*;
use tenferro_linalg::{LuPivot, NormKind, SvdOptions};
use tenferro_tensor::Tensor;

pub(crate) mod common {
    #[allow(unused_imports)]
    pub use tenferro_internal_ad_core::ops::*;
}

pub mod linalg;

pub use linalg::*;
