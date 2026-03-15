use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};

use super::{AdTensor, Result, TensorAdState};

impl<T: Scalar> AdTensor<T> {
    pub fn memory_space(&self) -> LogicalMemorySpace {
        self.structured_primal().memory_space()
    }

    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.structured_primal().preferred_compute_device()
    }

    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        match &mut self.0 {
            TensorAdState::Primal(primal) => primal.set_preferred_compute_device(device),
            TensorAdState::Forward { primal, tangent } => {
                primal.set_preferred_compute_device(device);
                tangent.set_preferred_compute_device(device);
            }
            TensorAdState::Reverse {
                primal, tangent, ..
            } => {
                primal.set_preferred_compute_device(device);
                if let Some(tangent) = tangent {
                    tangent.set_preferred_compute_device(device);
                }
            }
        }
    }

    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        match &self.0 {
            TensorAdState::Primal(primal) => {
                Ok(Self::new_primal(primal.to_memory_space_async(target)?))
            }
            TensorAdState::Forward { primal, tangent } => Self::new_forward(
                primal.to_memory_space_async(target)?,
                tangent.to_memory_space_async(target)?,
            ),
            TensorAdState::Reverse {
                primal,
                tangent,
                state,
            } => Ok(Self(TensorAdState::Reverse {
                primal: primal.to_memory_space_async(target)?,
                tangent: tangent
                    .as_ref()
                    .map(|value| value.to_memory_space_async(target))
                    .transpose()?,
                state: Arc::clone(state),
            })),
        }
    }

    pub fn to_memory_space(&self, target: LogicalMemorySpace) -> Result<Self> {
        let moved = self.to_memory_space_async(target)?;
        moved.wait();
        Ok(moved)
    }

    pub fn wait(&self) {
        self.structured_primal().wait();
        if let Some(tangent) = self.structured_tangent() {
            tangent.wait();
        }
    }

    pub fn is_ready(&self) -> bool {
        self.structured_primal().is_ready()
            && self
                .structured_tangent()
                .map(|tangent| tangent.is_ready())
                .unwrap_or(true)
    }
}
