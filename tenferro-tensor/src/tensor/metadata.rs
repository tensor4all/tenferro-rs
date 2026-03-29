use tenferro_device::{
    preferred_compute_devices, ComputeDevice, Error, LogicalMemorySpace, OpKind, Result,
};

use super::Tensor;

impl<T> Tensor<T> {
    /// Returns the shape (size of each dimension).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert_eq!(t.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strides (in units of `T`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let _strides = t.strides();
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns the element offset into the data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert_eq!(t.offset(), 0);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns a reference to the underlying data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let _buf = t.buffer();
    /// ```
    pub fn buffer(&self) -> &crate::DataBuffer<T> {
        &self.buffer
    }

    /// Returns a mutable reference to the underlying data buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let _buf = t.buffer_mut();
    /// ```
    pub fn buffer_mut(&mut self) -> &mut crate::DataBuffer<T> {
        &mut self.buffer
    }

    /// Returns the number of dimensions (rank).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert_eq!(t.ndim(), 2);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert_eq!(t.len(), 6);
    /// ```
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns `true` if the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[0, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(t.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the logical memory space where this tensor's data resides.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert_eq!(t.logical_memory_space(), LogicalMemorySpace::MainMemory);
    /// ```
    pub fn logical_memory_space(&self) -> LogicalMemorySpace {
        self.logical_memory_space
    }

    /// Returns the preferred compute device override, if set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(t.preferred_compute_device().is_none());
    /// ```
    pub fn preferred_compute_device(&self) -> Option<ComputeDevice> {
        self.preferred_compute_device
    }

    /// Set the preferred compute device override.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// t.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    /// ```
    pub fn set_preferred_compute_device(&mut self, device: Option<ComputeDevice>) {
        self.preferred_compute_device = device;
    }

    /// Returns `true` if this tensor is logically conjugated.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(!t.is_conjugated());
    /// ```
    pub fn is_conjugated(&self) -> bool {
        self.conjugated
    }

    /// Returns a reference to the forward-mode tangent, if set.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(t.fw_grad().is_none());
    /// ```
    pub fn fw_grad(&self) -> Option<&Tensor<T>> {
        self.fw_grad.as_deref()
    }

    /// Returns `true` if this tensor carries a forward-mode tangent.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// assert!(!t.has_fw_grad());
    /// ```
    pub fn has_fw_grad(&self) -> bool {
        self.fw_grad.is_some()
    }

    /// Attach a forward-mode tangent to this tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let grad = Tensor::<f64>::ones(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// t.set_fw_grad(grad);
    /// ```
    pub fn set_fw_grad(&mut self, grad: Tensor<T>) {
        self.fw_grad = Some(Box::new(grad));
    }

    /// Detach and return the forward-mode tangent, leaving `None`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// t.set_fw_grad(Tensor::<f64>::ones(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap());
    /// let _grad = t.detach_fw_grad().unwrap();
    /// ```
    pub fn detach_fw_grad(&mut self) -> Option<Tensor<T>> {
        self.fw_grad.take().map(|boxed| *boxed)
    }

    /// Return the effective compute devices for a given operation kind.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor).unwrap();
    /// let _devices = t.effective_compute_devices(OpKind::BatchedGemm).unwrap();
    /// ```
    pub fn effective_compute_devices(&self, op_kind: OpKind) -> Result<Vec<ComputeDevice>> {
        let compatible = preferred_compute_devices(self.logical_memory_space, op_kind)?;
        if let Some(device) = self.preferred_compute_device {
            if compatible.contains(&device) {
                Ok(vec![device])
            } else {
                Err(Error::NoCompatibleComputeDevice {
                    space: self.logical_memory_space,
                    op: op_kind,
                })
            }
        } else {
            Ok(compatible)
        }
    }
}
