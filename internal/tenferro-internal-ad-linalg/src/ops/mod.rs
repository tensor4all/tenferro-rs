use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use tenferro_algebra::{Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_internal_ad_core as tape;
use tenferro_internal_ad_core::{ops::*, AdTensor, DynAdTensorTyped};
use tenferro_internal_error::{Error, Result};
use tenferro_internal_frontend_core::DynTensorTyped;
use tenferro_internal_runtime::contracts::*;
use tenferro_internal_runtime::dispatch::*;
use tenferro_linalg::{
    CholeskyExResult, EigenResult, InvExResult, LinalgScalar, LuFactorExResult, LuFactorResult,
    LuPivot, LuResult, NormKind, QrResult, SolveExResult, SvdOptions,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::structured::StructuredTensor;

pub(crate) mod common {
    pub use tenferro_internal_ad_core::ops::*;
}

pub mod linalg;

pub use linalg::*;
