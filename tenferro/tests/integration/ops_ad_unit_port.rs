pub(crate) use num_complex::Complex64;

pub(crate) use crate::ops::ad::*;
pub(crate) use crate::ops::tests::with_cpu_runtime;
pub(crate) use crate::ops::{einsum_ad, sum_ad};
pub(crate) use crate::runtime::contracts::{EinsumRuntimeValue, ScalarRuntimeValue};
pub(crate) use crate::structured::StructuredTensor;
pub(crate) use crate::{AdTensor, Error, RuntimeContext};
pub(crate) use tenferro_linalg::NormKind;
pub(crate) use tenferro_prims::CpuContext;
pub(crate) use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
pub(crate) use tidu::expert::Tape;

#[path = "ported_ops_ad_tests/builder_pullbacks.rs"]
mod builder_pullbacks;
#[path = "ported_ops_ad_tests/eager_surface.rs"]
mod eager_surface;
#[path = "ported_ops_ad_tests/einsum_one_stage_complex.rs"]
mod einsum_one_stage_complex;
#[path = "ported_ops_ad_tests/einsum_one_stage_real.rs"]
mod einsum_one_stage_real;
#[path = "ported_ops_ad_tests/einsum_two_stage.rs"]
mod einsum_two_stage;
#[path = "ported_ops_ad_tests/linalg_finite_difference.rs"]
mod linalg_finite_difference;
#[path = "ported_ops_ad_tests/reduction_edge_cases.rs"]
mod reduction_edge_cases;
#[path = "ported_ops_ad_tests/scalar_generic.rs"]
mod scalar_generic;
#[path = "ported_ops_ad_tests/structured_pullbacks.rs"]
mod structured_pullbacks;
#[path = "ported_ops_ad_tests/support.rs"]
mod support;

use support::*;
