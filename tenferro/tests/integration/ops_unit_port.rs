pub(crate) use std::collections::HashMap;

pub(crate) use crate::ops::*;
pub(crate) use crate::structured::StructuredTensor;
pub(crate) use crate::{AdTensor, Error, Result, RuntimeContext, Tensor};
pub(crate) use tenferro_algebra::Standard;
pub(crate) use tenferro_internal_ad_linalg::__typed_ad::*;
pub(crate) use tenferro_linalg::{LuPivot, NormKind, SvdOptions};
pub(crate) use tenferro_prims::{CpuBackend, CpuContext, CudaContext, RocmContext};
pub(crate) use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
pub(crate) use tidu::expert::Tape;

#[path = "ported_ops_tests/builder_coverage.rs"]
mod builder_coverage;
#[path = "ported_ops_tests/organization.rs"]
mod organization;
#[path = "ported_ops_tests/runtime_dispatch.rs"]
mod runtime_dispatch;
#[path = "ported_ops_tests/runtime_helpers.rs"]
mod runtime_helpers;
#[path = "ported_ops_tests/runtime_surface.rs"]
mod runtime_surface;
#[path = "ported_ops_tests/support.rs"]
pub(crate) mod support;

pub(crate) use support::*;

fn tensor_data<T: tenferro_algebra::Scalar + Copy>(tensor: &DenseTensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}
