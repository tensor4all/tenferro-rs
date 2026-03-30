use std::collections::HashMap;

use chainrules_core::Differentiable as _;
use tenferro_algebra::{Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_linalg::{
    CholeskyExResult, EigenResult, InvExResult, LinalgScalar, LuFactorExResult, LuFactorResult,
    LuPivot, LuResult, NormKind, QrResult, SolveExResult, SvdOptions,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::runtime::contracts::*;
use crate::runtime::dispatch::*;
use crate::structured::{
    accumulate_structured_tangent, compress_dense_to_layout_in_ctx, einsum_with_subscripts_in_ctx,
    reverse_subscripts, to_dense_in_ctx, StructuredTensor,
};
use crate::tape;
use crate::{Error, Result};

#[path = "ad/mod.rs"]
#[doc(hidden)]
pub(crate) mod __typed_ad;
#[path = "einsum/mod.rs"]
#[doc(hidden)]
mod __typed_einsum;
#[path = "reduction/mod.rs"]
#[doc(hidden)]
mod __typed_reduction;
#[path = "scalar/mod.rs"]
#[doc(hidden)]
mod __typed_scalar;
mod linalg;

mod common;

pub use __typed_einsum::*;
pub use __typed_reduction::*;
pub use __typed_scalar::*;
#[allow(unused_imports)]
pub(crate) use common::*;
pub use linalg::*;
