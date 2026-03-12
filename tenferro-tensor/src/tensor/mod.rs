mod autodiff;
mod constructors;
mod data_ops;
mod metadata;
mod transfer;
mod views;

use std::sync::Arc;

use tenferro_algebra::Scalar;
use tenferro_device::{ComputeDevice, LogicalMemorySpace};

use crate::{layout::compute_contiguous_strides, CompletionEvent, DataBuffer, MemoryOrder};

/// Multi-dimensional dense tensor.
///
/// `Tensor<T>` owns or shares a [`DataBuffer`] together with shape, strides,
/// and memory-space metadata.
///
/// ## Zero-copy views
///
/// Operations like [`permute`](Tensor::permute), [`broadcast`](Tensor::broadcast),
/// and [`diagonal`](Tensor::diagonal) return new tensors that share the
/// underlying buffer and only adjust metadata.
///
/// ## Accessing raw data
///
/// Use [`DataBuffer::as_slice`] via [`Tensor::buffer`] together with
/// [`Tensor::dims`], [`Tensor::strides`], and [`Tensor::offset`] to build
/// backend-specific views.
///
/// ## GPU async support
///
/// The optional [`CompletionEvent`] tracks pending GPU computation so future
/// backends can chain asynchronous work without forcing CPU synchronization.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::zeros(
///     &[2, 3],
///     LogicalMemorySpace::MainMemory,
///     MemoryOrder::ColumnMajor,
/// );
/// assert_eq!(t.dims(), &[2, 3]);
/// assert_eq!(t.len(), 6);
/// ```
pub struct Tensor<T: Scalar> {
    buffer: DataBuffer<T>,
    dims: Arc<[usize]>,
    strides: Arc<[isize]>,
    offset: isize,
    logical_memory_space: LogicalMemorySpace,
    preferred_compute_device: Option<ComputeDevice>,
    event: Option<CompletionEvent>,
    conjugated: bool,
    fw_grad: Option<Box<Tensor<T>>>,
}

impl<T: Scalar> Clone for Tensor<T> {
    /// Shallow clone: shares the underlying data buffer.
    ///
    /// For a deep copy, materialize into a new allocation with
    /// [`Tensor::contiguous`] or another explicit data-producing operation.
    fn clone(&self) -> Self {
        Self::from_parts(
            self.buffer.clone(),
            self.dims.clone(),
            self.strides.clone(),
            self.offset,
            self.logical_memory_space,
            self.preferred_compute_device,
            self.event.clone(),
            self.conjugated,
            self.fw_grad.clone(),
        )
    }
}

impl<T: Scalar> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_pending_event = self.event.is_some();
        let has_fw_grad = self.fw_grad.is_some();
        f.debug_struct("Tensor")
            .field("dtype", &std::any::type_name::<T>())
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .field("len", &self.len())
            .field("logical_memory_space", &self.logical_memory_space)
            .field("preferred_compute_device", &self.preferred_compute_device)
            .field("is_contiguous", &self.is_contiguous())
            .field("conjugated", &self.conjugated)
            .field("has_pending_event", &has_pending_event)
            .field("has_fw_grad", &has_fw_grad)
            .finish()
    }
}

impl<T: Scalar> Tensor<T> {
    pub(crate) fn from_parts(
        buffer: DataBuffer<T>,
        dims: Arc<[usize]>,
        strides: Arc<[isize]>,
        offset: isize,
        logical_memory_space: LogicalMemorySpace,
        preferred_compute_device: Option<ComputeDevice>,
        event: Option<CompletionEvent>,
        conjugated: bool,
        fw_grad: Option<Box<Tensor<T>>>,
    ) -> Self {
        Self {
            buffer,
            dims,
            strides,
            offset,
            logical_memory_space,
            preferred_compute_device,
            event,
            conjugated,
            fw_grad,
        }
    }

    pub(crate) fn from_owned_contiguous_data(
        data: Vec<T>,
        dims: Arc<[usize]>,
        order: MemoryOrder,
        logical_memory_space: LogicalMemorySpace,
        preferred_compute_device: Option<ComputeDevice>,
        conjugated: bool,
    ) -> Self {
        let strides = Arc::from(compute_contiguous_strides(dims.as_ref(), order));
        Self::from_parts(
            DataBuffer::from_vec(data),
            dims,
            strides,
            0,
            logical_memory_space,
            preferred_compute_device,
            None,
            conjugated,
            None,
        )
    }

    pub(crate) fn shared_view_with(
        &self,
        dims: Arc<[usize]>,
        strides: Arc<[isize]>,
        offset: isize,
    ) -> Self {
        Self::from_parts(
            self.buffer.clone(),
            dims,
            strides,
            offset,
            self.logical_memory_space,
            self.preferred_compute_device,
            None,
            self.conjugated,
            None,
        )
    }

    pub(crate) fn materialized_from_vec(&self, data: Vec<T>, order: MemoryOrder) -> Self {
        Self::from_owned_contiguous_data(
            data,
            self.dims.clone(),
            order,
            self.logical_memory_space,
            self.preferred_compute_device,
            self.conjugated,
        )
    }

    pub(crate) fn cpu_backed_slice_or_panic(&self, operation: &str) -> &[T] {
        self.buffer.as_slice().unwrap_or_else(|| {
            panic!("{operation}: CPU-only operation; GPU tensors are not supported")
        })
    }
}
