//! The shared allocation domain: the one allocator for tensors the GPU and the paired CPU
//! engine both address.
//!
//! `tenferro_tensor::SharedTensorAllocationDomain` is the backend-neutral boundary the CPU
//! operation crates already allocate results through. This is the Metal provider: every
//! allocation is a `MTLResourceStorageModeShared` buffer, so a CPU result lands where a
//! Metal launch can read it next, with no upload (design §4.3).
//!
//! Every `DType` is allocated, not only the two the GPU computes. An f64 reduction over a
//! shared tensor runs on the paired [`tenferro_cpu::CpuEngine`] and its output must stay in
//! this domain — otherwise the next Metal operation would need a transfer, which is the
//! thing this design removes. It is the Metal *engine* that refuses f64 and Complex64, not
//! the allocator.

use std::sync::Arc;

use tenferro_tensor::{AllocationDomainId, DType, Result, SharedTensorAllocationDomain, Tensor};

use super::context::RuntimeCell;

/// The shared allocation domain of one [`super::MetalContext`].
///
/// Construct it through `MetalContext::open`; a domain belongs to exactly one context.
#[derive(Debug, Clone)]
pub struct MetalAllocationDomain {
    cell: Arc<RuntimeCell>,
}

impl MetalAllocationDomain {
    pub(crate) fn new(cell: Arc<RuntimeCell>) -> Self {
        Self { cell }
    }
}

impl SharedTensorAllocationDomain for MetalAllocationDomain {
    fn id(&self) -> AllocationDomainId {
        self.cell.domain()
    }

    fn allocate(&self, dtype: DType, shape: &[usize]) -> Result<Tensor> {
        self.cell
            .allocate("MetalAllocationDomain::allocate", dtype, shape)
    }
}
