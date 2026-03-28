use tenferro_device::{ComputeDevice, LogicalMemorySpace};

use super::Tensor;
use crate::Result;

impl Tensor {
    /// Returns the logical memory space holding the primal payload.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{LogicalMemorySpace, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// assert_eq!(x.memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn memory_space(&self) -> LogicalMemorySpace {
        match self {
            Self::F32(value) => value.memory_space(),
            Self::F64(value) => value.memory_space(),
            Self::C32(value) => value.memory_space(),
            Self::C64(value) => value.memory_space(),
        }
    }

    /// Returns the preferred compute-device override for this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{ComputeDevice, Tensor};
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    /// assert_eq!(x.preferred_compute_device(), Some(ComputeDevice::Cpu { device_id: 0 }));
    /// ```
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        match self {
            Self::F32(value) => value.preferred_compute_device(),
            Self::F64(value) => value.preferred_compute_device(),
            Self::C32(value) => value.preferred_compute_device(),
            Self::C64(value) => value.preferred_compute_device(),
        }
    }

    /// Sets the preferred compute-device override for this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{ComputeDevice, Tensor};
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    /// assert_eq!(x.preferred_compute_device(), Some(ComputeDevice::Cpu { device_id: 0 }));
    /// ```
    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        match self {
            Self::F32(value) => value.set_preferred_compute_device(device),
            Self::F64(value) => value.set_preferred_compute_device(device),
            Self::C32(value) => value.set_preferred_compute_device(device),
            Self::C64(value) => value.set_preferred_compute_device(device),
        }
    }

    /// Asynchronously transfers this tensor to a target memory space.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{LogicalMemorySpace, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// let y = x.to_memory_space_async(LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(y.memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn to_memory_space_async(&self, target: LogicalMemorySpace) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(value.to_memory_space_async(target)?)),
            Self::F64(value) => Ok(Self::F64(value.to_memory_space_async(target)?)),
            Self::C32(value) => Ok(Self::C32(value.to_memory_space_async(target)?)),
            Self::C64(value) => Ok(Self::C64(value.to_memory_space_async(target)?)),
        }
    }

    /// Transfers this tensor to a target memory space and waits for readiness.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{LogicalMemorySpace, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// let y = x.to_memory_space(LogicalMemorySpace::MainMemory).unwrap();
    /// assert_eq!(y.memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn to_memory_space(&self, target: LogicalMemorySpace) -> Result<Self> {
        let moved = self.to_memory_space_async(target)?;
        moved.wait();
        Ok(moved)
    }

    /// Waits for pending transfers or device work associated with this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.wait();
    /// ```
    pub fn wait(&self) {
        match self {
            Self::F32(value) => value.wait(),
            Self::F64(value) => value.wait(),
            Self::C32(value) => value.wait(),
            Self::C64(value) => value.wait(),
        }
    }

    /// Returns `true` when tensor data is ready without blocking.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// assert!(x.is_ready());
    /// ```
    pub fn is_ready(&self) -> bool {
        match self {
            Self::F32(value) => value.is_ready(),
            Self::F64(value) => value.is_ready(),
            Self::C32(value) => value.is_ready(),
            Self::C64(value) => value.is_ready(),
        }
    }

    /// Convenience wrapper for synchronous transfer to CPU-visible main memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{LogicalMemorySpace, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// let y = x.to_cpu().unwrap();
    /// assert_eq!(y.memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn to_cpu(&self) -> Result<Self> {
        self.to_memory_space(LogicalMemorySpace::MainMemory)
    }

    /// Convenience wrapper for asynchronous transfer to CPU-visible main memory.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{LogicalMemorySpace, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// let y = x.to_cpu_async().unwrap();
    /// assert_eq!(y.memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn to_cpu_async(&self) -> Result<Self> {
        self.to_memory_space_async(LogicalMemorySpace::MainMemory)
    }

    /// Convenience wrapper for synchronous transfer to the default GPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let gpu = x.to_gpu()?;
    /// assert_eq!(gpu.memory_space(), tenferro::LogicalMemorySpace::GpuMemory { device_id: 0 });
    /// # Ok::<(), tenferro::Error>(())
    /// ```
    pub fn to_gpu(&self) -> Result<Self> {
        self.to_gpu_on(0)
    }

    /// Convenience wrapper for asynchronous transfer to the default GPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let gpu = x.to_gpu_async()?;
    /// assert_eq!(gpu.memory_space(), tenferro::LogicalMemorySpace::GpuMemory { device_id: 0 });
    /// # Ok::<(), tenferro::Error>(())
    /// ```
    pub fn to_gpu_async(&self) -> Result<Self> {
        self.to_gpu_async_on(0)
    }

    /// Convenience wrapper for synchronous transfer to a specific GPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let gpu = x.to_gpu_on(1)?;
    /// assert_eq!(gpu.memory_space(), tenferro::LogicalMemorySpace::GpuMemory { device_id: 1 });
    /// # Ok::<(), tenferro::Error>(())
    /// ```
    pub fn to_gpu_on(&self, device_id: usize) -> Result<Self> {
        self.to_memory_space(LogicalMemorySpace::GpuMemory { device_id })
    }

    /// Convenience wrapper for asynchronous transfer to a specific GPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let gpu = x.to_gpu_async_on(1)?;
    /// assert_eq!(gpu.memory_space(), tenferro::LogicalMemorySpace::GpuMemory { device_id: 1 });
    /// # Ok::<(), tenferro::Error>(())
    /// ```
    pub fn to_gpu_async_on(&self, device_id: usize) -> Result<Self> {
        self.to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id })
    }
}
