use tenferro_algebra::{Scalar, Standard};
use tenferro_einsum::{self as tf_einsum, Subscripts};
use tenferro_internal_ad_core::AdTensor;
use tenferro_tensor::Tensor;

use crate::core::NodeId;
use crate::runtime::contracts::*;
use crate::runtime::dispatch::*;
use crate::structured::{einsum_with_subscripts_in_ctx, reverse_subscripts, StructuredTensor};
use crate::{tape, Error, Result};

pub mod ad;
pub mod einsum;
pub(crate) mod linalg;
pub mod reduction;
pub mod scalar;

mod common;

#[allow(unused_imports)]
pub(crate) use common::*;
pub use einsum::*;
pub use reduction::*;
pub use scalar::*;
