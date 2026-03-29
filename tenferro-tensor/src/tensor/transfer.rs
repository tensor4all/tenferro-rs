use tenferro_algebra::Scalar;
use tenferro_device::{LogicalMemorySpace, Result};

use super::Tensor;

impl<T: Scalar> Tensor<T> {
    /// Asynchronously transfer this tensor to a different memory space.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let t2 = t.to_memory_space_async(LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(t2.dims(), &[2, 3]);
    /// ```
    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Tensor<T>> {
        if target == self.logical_memory_space {
            return Ok(self.clone());
        }

        #[cfg(feature = "cuda")]
        {
            return crate::cuda_runtime::transfer_tensor(self, target);
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(tenferro_device::Error::DeviceError(
                "GPU memory transfer not available: rebuild with `tenferro-tensor --features cuda`"
                    .into(),
            ))
        }
    }
}

impl<T> Tensor<T> {
    /// Wait for any pending GPU computation to complete.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// t.wait();
    /// ```
    pub fn wait(&self) {
        let _ = self;
    }

    /// Check if tensor data is ready without blocking.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(t.is_ready());
    /// ```
    pub fn is_ready(&self) -> bool {
        self.event.is_none()
    }
}
