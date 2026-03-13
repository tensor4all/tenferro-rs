use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use chainrules_core::Differentiable as _;
use num_complex::Complex;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_linalg::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LinalgScalar, LstsqResult,
    LuFactorExResult, LuFactorResult, LuPivot, LuResult, NormKind, QrResult, SlogdetResult,
    SolveExResult, SvdOptions, SvdResult,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::core::value::{AdValue, NodeId};
use crate::runtime::contracts::*;
use crate::runtime::dispatch::*;
use crate::structured::{
    accumulate_structured_tangent, compress_dense_to_layout_in_ctx, einsum_with_subscripts_in_ctx,
    reverse_subscripts, to_dense_in_ctx, StructuredTensor,
};
use crate::tape;
use crate::{AdTensor, Error, Result, TapeId};

pub mod ad;
pub mod einsum;
mod linalg;
pub mod reduction;
pub mod scalar;

mod common;

#[allow(unused_imports)]
pub(crate) use common::*;
pub use einsum::chainrules as chainrules_api;
pub use einsum::*;
pub use linalg::*;
pub use reduction::*;
pub use scalar::*;

#[cfg(test)]
pub(crate) mod tests;
