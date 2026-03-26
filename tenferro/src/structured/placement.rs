use tenferro_algebra::Scalar;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};

use super::StructuredTensor;
use crate::Result;

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
        *self = Self(tenferro_tensor::StructuredTensor::from_validated_parts(
            self.logical_dims().to_vec(),
            self.axis_classes().to_vec(),
            payload,
        ));
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        let payload = self.payload().to_memory_space_async(target)?;
        Ok(Self(self.0.with_payload_like(payload)?))
    }

    pub fn wait(&self) {
        self.payload().wait();
    }

    pub fn is_ready(&self) -> bool {
        self.payload().is_ready()
    }
}
