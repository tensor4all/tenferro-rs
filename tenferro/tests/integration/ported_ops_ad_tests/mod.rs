mod builder_pullbacks;
mod eager_surface;
mod einsum_one_stage_complex;
mod einsum_one_stage_real;
mod einsum_two_stage;
mod linalg_finite_difference;
mod reduction_edge_cases;
mod scalar_generic;
mod structured_pullbacks;
mod support;

use num_complex::Complex64;
use tenferro_linalg::NormKind;
use tidu::expert::Tape;

use super::*;
use crate::ops::tests::with_cpu_runtime;
use crate::{Error, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use support::*;
