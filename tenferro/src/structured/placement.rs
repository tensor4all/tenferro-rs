use tenferro_algebra::Scalar;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};

use super::StructuredTensor;
use crate::{Error, Result};

impl<T: Scalar> StructuredTensor<T> {
    pub fn memory_space(&self) -> LogicalMemorySpace {
        self.payload().logical_memory_space()
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.payload().preferred_compute_device()
    }

    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        let mut payload = self.payload().clone();
        payload.set_preferred_compute_device(device);
        *self = self.with_payload_like(payload).unwrap_or_else(|err| {
            unreachable!(
                "StructuredTensor::set_preferred_compute_device should preserve layout: {err}"
            )
        });
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        let payload = self
            .payload()
            .to_memory_space_async(target)
            .map_err(Error::from)?;
        self.with_payload_like(payload)
    }

    pub fn wait(&self) {
        self.payload().wait();
    }

    pub fn is_ready(&self) -> bool {
        self.payload().is_ready()
    }
}
