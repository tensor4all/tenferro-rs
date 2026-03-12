mod builder_pullbacks;
mod eager_surface;
mod einsum_one_stage_complex;
mod einsum_one_stage_real;
mod einsum_two_stage;
mod linalg_finite_difference;
mod scalar_generic;
mod structured_pullbacks;
mod support;

use num_complex::Complex64;

use super::*;
use crate::api::tests::with_cpu_runtime;
use crate::{
    AdScalar, AdValue, DynAdScalar, DynAdTensor, NodeId, RuntimeContext, StructuredTensor, TapeId,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use support::*;
