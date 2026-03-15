use std::collections::HashMap;

use super::*;
use crate::RuntimeContext;
use tenferro_algebra::Standard;
use tenferro_linalg::SvdOptions;
use tenferro_prims::{CpuBackend, CpuContext, CudaContext, RocmContext};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

mod builder_coverage;
mod organization;
mod runtime_dispatch;
mod runtime_helpers;
mod runtime_surface;
mod support;

pub(crate) use support::{as_slice, assert_primal_mode, reverse_leaf_f64, with_cpu_runtime};
