use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chainrules_core::Differentiable as _;
use num_complex::Complex;
use num_traits::Float;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_linalg::backend::CpuLinalgScalar;
use tenferro_linalg::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LinalgScalar, LstsqResult,
    LuFactorExResult, LuFactorResult, LuPivot, LuResult, NormKind, QrResult, SlogdetResult,
    SolveExResult, SvdOptions, SvdResult,
};
use tenferro_prims::{CpuBackend, CpuContext, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ad_value::{AdValue, NodeId};
use crate::reverse_tape;
use crate::runtime::with_default_runtime;
use crate::structured::{
    accumulate_structured_tangent, compress_dense_to_layout_in_ctx, einsum_with_subscripts_in_ctx,
    reverse_subscripts, to_dense_in_ctx, StructuredTensor,
};
use crate::{AdTensor, Error, Result, TapeId};

pub mod ad;
mod ad_results;
pub mod chainrules_api;

pub use ad_results::{
    AdEigResult, AdEigenResult, AdLstsqResult, AdLuResult, AdQrResult, AdSlogdetResult, AdSvdResult,
};

mod ad_builders;
mod linalg_builders;
mod primal_builders;
mod runtime;
mod runtime_dispatch;
mod scalar_ad_builders;
mod scalar_contracts;
mod scalar_runtime;

pub use ad_builders::*;
pub use linalg_builders::*;
pub use primal_builders::*;
#[allow(unused_imports)]
pub(crate) use runtime::*;
pub(crate) use runtime_dispatch::*;
pub use scalar_ad_builders::*;

#[cfg(test)]
pub(crate) mod tests;
