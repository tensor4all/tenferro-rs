use num_complex::{Complex, Complex32, Complex64};
use num_traits::{One, Zero};
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use tenferro_tensor_core::SliceSpec as CoreSliceSpec;
pub use tenferro_tensor_core::{DynRank, Rank, TensorLayout, TensorRank};

mod accessors;
mod shape_packing;
mod strided_view;

pub use strided_view::{StridedSliceSpec, StridedTensorView, StridedTensorViewMut};

/// Memory location for tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::MemoryKind;
///
/// let kind = MemoryKind::UnpinnedHost;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Managed,
    Other(String),
}

/// Compute device family.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DeviceKind;
///
/// let kind = DeviceKind::Cpu;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

/// GPU backend family used by placement metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::GpuBackendKind;
///
/// let kind = GpuBackendKind::Cuda;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Cuda,
    Rocm,
    Other(String),
}

/// Concrete compute device identifier.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind};
///
/// let device = DeviceId {
///     kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///     ordinal: 0,
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub kind: DeviceKind,
    pub ordinal: usize,
}

/// Backwards-compatible internal alias during the crate-boundary migration.
pub type ComputeDevice = DeviceId;

/// Placement metadata for a tensor buffer.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement};
///
/// let placement = Placement {
///     memory_kind: MemoryKind::Device,
///     device: Some(DeviceId {
///         kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///         ordinal: 0,
///     }),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
}

/// Backend-owned buffer handle.
///
/// `BufferHandle::new` creates an empty opaque handle. Use
/// [`BufferHandle::new_with_len`] when test or adapter code needs to model a
/// non-empty backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BufferHandle;
///
/// let handle = BufferHandle::<f64>::new(7);
/// ```
#[derive(Clone)]
pub struct BufferHandle<T> {
    pub id: u64,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Debug for BufferHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<T> BufferHandle<T> {
    /// Create a new backend buffer handle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::BufferHandle;
    ///
    /// let handle = BufferHandle::<f64>::new(1);
    /// assert_eq!(handle.id, 1);
    /// assert_eq!(tenferro_tensor::BackendBuffer::len(&handle), 0);
    /// ```
    pub fn new(id: u64) -> Self {
        Self::new_with_len(id, 0)
    }

    /// Create a new backend buffer handle with a logical element count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{BackendBuffer, BufferHandle};
    ///
    /// let handle = BufferHandle::<f64>::new_with_len(1, 4);
    /// assert_eq!(handle.id, 1);
    /// assert_eq!(BackendBuffer::len(&handle), 4);
    /// ```
    pub fn new_with_len(id: u64, len: usize) -> Self {
        Self {
            id,
            len,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Opaque backend-owned tensor buffer.
///
/// Tensor core never inspects backend-native allocations directly. Backend
/// crates store their own concrete handle types behind this trait and
/// downcast inside the owning backend only.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use tenferro_tensor::{BackendBuffer, BufferHandle};
///
/// let buffer: Arc<dyn BackendBuffer<f64>> = Arc::new(BufferHandle::<f64>::new_with_len(7, 2));
/// assert_eq!(buffer.backend_family(), "opaque");
/// assert_eq!(buffer.len(), 2);
/// ```
pub trait BackendBuffer<T>: Debug + Send + Sync + 'static {
    /// Stable backend family identifier.
    fn backend_family(&self) -> &'static str;

    /// Number of logical elements in the backend allocation.
    fn len(&self) -> usize;

    /// Returns `true` when the backend allocation is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Type-erased access for the backend crate that owns the concrete handle.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> BackendBuffer<T> for BufferHandle<T> {
    fn backend_family(&self) -> &'static str {
        "opaque"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::Buffer;
///
/// let host = Buffer::Host(vec![1.0_f64, 2.0]);
/// ```
#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(Arc<dyn BackendBuffer<T>>),
}

/// Contiguous column-major typed tensor storage.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Rank, TypedTensor};
///
/// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(t.shape(), &[2, 2]);
///
/// let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]);
/// assert_eq!(static_rank.rank(), 2);
/// ```
///
/// The `R` parameter stores rank metadata. It defaults to dynamic rank
/// (`DynRank`); use [`Rank<N>`](Rank) for compile-time rank validation.
#[derive(Clone, Debug)]
pub struct TypedTensor<T, R: TensorRank = DynRank> {
    pub buffer: Buffer<T>,
    layout: TensorLayout<R>,
    pub placement: Placement,
}

/// Borrowed tensor buffer reference used by read-only typed views.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBufferRef;
///
/// let data = [1_i32, 2];
/// let buffer = TensorBufferRef::Host(&data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorBufferRef<'a, T> {
    Host(&'a [T]),
    Backend(Arc<dyn BackendBuffer<T>>),
}

impl<T> Clone for TensorBufferRef<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Host(data) => Self::Host(data),
            Self::Backend(buffer) => Self::Backend(Arc::clone(buffer)),
        }
    }
}

impl<T: 'static> TensorBufferRef<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRef;
    ///
    /// let data = [1_i32, 2, 3];
    /// assert_eq!(TensorBufferRef::Host(&data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRef;
    ///
    /// let data: [f64; 0] = [];
    /// assert!(TensorBufferRef::Host(&data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed tensor buffer reference used by mutable typed views.
///
/// Backend buffers can be represented for residency metadata, but this crate
/// does not expose host mutation for backend-native allocations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBufferRefMut;
///
/// let mut data = [1_i32, 2];
/// let buffer = TensorBufferRefMut::Host(&mut data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorBufferRefMut<'a, T> {
    Host(&'a mut [T]),
    Backend(Arc<dyn BackendBuffer<T>>),
}

impl<T: 'static> TensorBufferRefMut<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRefMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// assert_eq!(TensorBufferRefMut::Host(&mut data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRefMut;
    ///
    /// let mut data: [f64; 0] = [];
    /// assert!(TensorBufferRefMut::Host(&mut data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read-only borrowed view of typed tensor storage with arbitrary strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Rank, TypedTensorView};
///
/// let data = [1_i32, 2, 3, 4];
/// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
/// assert_eq!(view.get(&[1, 1]), Some(&4));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct TypedTensorView<'a, T, R: TensorRank = DynRank> {
    buffer: TensorBufferRef<'a, T>,
    layout: TensorLayout<R>,
    placement: Placement,
}

impl<'a, T: 'static> TypedTensorView<'a, T, DynRank> {
    /// Create a borrowed compact column-major host view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::new(&[2], &data)?;
    /// assert_eq!(view.as_slice()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn new(shape: &'a [usize], data: &'a [T]) -> crate::Result<Self> {
        Self::from_col_major(shape, data)
    }

    /// Create a borrowed dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_col_major(&[2, 2], &data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape.to_vec().into())
            .map_err(|err| tensor_layout_error("TypedTensorView::from_col_major", err))?;
        Self::from_buffer_ref(
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            layout.offset(),
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_col_major",
        )
    }

    /// Create a borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.get(&[2]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape.as_ref().to_vec(),
            strides.as_ref().to_vec(),
            offset,
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorView<'a, T, R> {
    /// Create a rank-generic borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
    /// assert_eq!(view.get(&[1, 1]), Some(&4));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape,
            strides,
            offset,
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice_ranked",
        )
    }

    fn from_buffer_ref(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorBufferRef<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            layout,
            placement,
        })
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![-1], 1, &data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 1, &data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed physical host slice backing this view.
    ///
    /// Panics when called on a backend buffer; use [`TypedTensorView::as_slice`]
    /// when a fallible host-inspection API is needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.as_physical_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_physical_slice(&self) -> &'a [T] {
        match &self.buffer {
            TensorBufferRef::Host(data) => data,
            TensorBufferRef::Backend(_) => panic!("as_physical_slice called on backend buffer"),
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 6];
    /// let view = TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert!(view.layout().is_compact_col_major());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorView};
    ///
    /// let data = [1_i32];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 0, &data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.try_linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Borrow one host element by logical index.
    ///
    /// Returns `None` for out-of-bounds indices and backend buffers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.try_linear_offset(indices)?;
        match &self.buffer {
            TensorBufferRef::Host(data) => data.get(offset),
            TensorBufferRef::Backend(_) => None,
        }
    }

    /// Explicit alias for [`TypedTensorView::get`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 0, &data)?;
    /// assert_eq!(view.try_get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        self.get(indices)
    }

    /// Borrow the contiguous host slice covered by this view.
    ///
    /// Returns an explicit error for backend buffers and for non-contiguous
    /// layouts. This method never downloads or materializes backend data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?;
    /// assert_eq!(view.as_slice()?, &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice(&self) -> crate::Result<&'a [T]> {
        let data =
            match &self.buffer {
                TensorBufferRef::Host(data) => data,
                TensorBufferRef::Backend(_) => return Err(crate::Error::backend_failure(
                    "TypedTensorView::as_slice",
                    "backend buffers cannot be inspected as host slices; download explicitly first",
                )),
            };
        contiguous_layout_slice(self.layout(), data, "TypedTensorView::as_slice")
    }

    /// Materialize this view as compact column-major host tensor storage.
    ///
    /// This is an explicit copy boundary. Backend buffers return an error
    /// instead of being downloaded implicitly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4]);
    /// let transposed = tensor.as_view().transpose_view([1, 0])?;
    /// let compact = transposed.to_contiguous()?;
    /// assert_eq!(compact.as_slice(), &[1, 3, 2, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn to_contiguous(&self) -> crate::Result<TypedTensor<T, R>>
    where
        T: Clone,
    {
        let op = "TypedTensorView::to_contiguous";
        let data = materialize_view_buffer_col_major(
            self.shape(),
            self.strides(),
            self.offset(),
            &self.buffer,
            op,
        )?;
        let shape = R::shape_from_vec(self.shape().to_vec().into())
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensor::from_vec_col_major(shape, data))
    }

    /// Return a metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4, 5, 6];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorView::transpose_view", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Explicit alias for [`TypedTensorView::transpose_view`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
    /// assert_eq!(view.try_permute_axes(&[1, 0])?.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_permute_axes(&self, axes: &[usize]) -> crate::Result<Self> {
        self.transpose_view(axes)
    }

    /// Return a metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![1], 0, &data)?;
    /// let reversed = view.try_slice(&[StridedSliceSpec::reverse()])?;
    /// assert_eq!(reversed.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice(&self, slices: &[StridedSliceSpec]) -> crate::Result<Self> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorView::try_slice")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorView::try_slice", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Return a metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.try_slice_axis(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice_axis(&self, axis: usize, slice: StridedSliceSpec) -> crate::Result<Self> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorView::try_slice_axis",
        )?;
        self.try_slice(&slices)
    }

    /// Return a metadata-only dynamic-rank reshape for contiguous column-major views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.try_reshape(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_reshape(&self, shape: &[usize]) -> crate::Result<TypedTensorView<'a, T, DynRank>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorView::try_reshape",
        )?;
        Ok(TypedTensorView {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }
}

/// Mutable borrowed view of typed tensor storage with arbitrary strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TypedTensorViewMut;
///
/// let mut data = [1_i32, 2, 3];
/// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
/// *view.get_mut(&[2]).unwrap() = 10;
/// assert_eq!(view.as_read_only().get(&[2]), Some(&10));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct TypedTensorViewMut<'a, T, R: TensorRank = DynRank> {
    buffer: TensorBufferRefMut<'a, T>,
    layout: TensorLayout<R>,
    placement: Placement,
}

impl<'a, T: 'static> TypedTensorViewMut<'a, T, DynRank> {
    /// Create a mutable dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::from_col_major(&[2, 2], &mut data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a mut [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape.to_vec().into())
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::from_col_major", err))?;
        Self::from_buffer_ref_mut(
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            layout.offset(),
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_col_major",
        )
    }

    /// Create a mutable host view from explicit layout metadata.
    ///
    /// Layouts where distinct logical elements can alias the same physical
    /// element are rejected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// assert!(TypedTensorViewMut::from_slice(vec![2], vec![0], 0, &mut data).is_err());
    /// ```
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape.as_ref().to_vec(),
            strides.as_ref().to_vec(),
            offset,
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorViewMut<'a, T, R> {
    /// Create a rank-generic mutable host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape,
            strides,
            offset,
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice_ranked",
        )
    }

    fn from_buffer_ref_mut(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorBufferRefMut<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            layout,
            placement,
        })
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![-1], 1, &mut data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 1, &mut data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed physical host slice backing this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.as_physical_slice(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_physical_slice(&self) -> &[T] {
        match &self.buffer {
            TensorBufferRefMut::Host(data) => data,
            TensorBufferRefMut::Backend(_) => panic!("as_physical_slice called on backend buffer"),
        }
    }

    /// Mutably borrow the physical host slice backing this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// view.as_physical_slice_mut()[0] = 3;
    /// assert_eq!(view.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_physical_slice_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => data,
            TensorBufferRefMut::Backend(_) => {
                panic!("as_physical_slice_mut called on backend buffer")
            }
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 6];
    /// let view = TypedTensorViewMut::from_slice(vec![2, 3], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert!(view.layout().is_compact_col_major());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
    /// assert_eq!(view.try_linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.try_linear_offset(indices)?;
        match &self.buffer {
            TensorBufferRefMut::Host(data) => data.get(offset),
            TensorBufferRefMut::Backend(_) => None,
        }
    }

    /// Mutably borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// *view.get_mut(&[1]).unwrap() = 20;
    /// assert_eq!(view.get(&[1]), Some(&20));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.try_linear_offset(indices)?;
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => data.get_mut(offset),
            TensorBufferRefMut::Backend(_) => None,
        }
    }

    /// Explicit alias for [`TypedTensorViewMut::get`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.try_get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_get(&self, indices: &[usize]) -> Option<&T> {
        self.get(indices)
    }

    /// Explicit alias for [`TypedTensorViewMut::get_mut`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let mut view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// *view.try_get_mut(&[0]).unwrap() = 2;
    /// assert_eq!(view.get(&[0]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.get_mut(indices)
    }

    /// Copy compact column-major host tensor values into this mutable view.
    ///
    /// This is an explicit copy-back boundary. Backend source or destination
    /// buffers return an error instead of transferring data implicitly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let mut tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![0, 0, 0, 0]);
    /// let src = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4]);
    /// tensor.as_view_mut().transpose_view([1, 0])?.copy_from_contiguous(&src)?;
    /// assert_eq!(tensor.as_slice(), &[1, 3, 2, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn copy_from_contiguous(&mut self, src: &TypedTensor<T, R>) -> crate::Result<()>
    where
        T: Clone,
    {
        let op = "TypedTensorViewMut::copy_from_contiguous";
        if self.shape() != src.shape() {
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!(
                    "shape mismatch: destination {:?} does not match source {:?}",
                    self.shape(),
                    src.shape()
                ),
            });
        }

        let src_data = match &src.buffer {
            Buffer::Host(data) => contiguous_layout_slice(src.layout(), data, op)?,
            Buffer::Backend(_) => {
                return Err(crate::Error::backend_failure(
                    op,
                    "source backend buffer cannot be copied through host memory; download explicitly first",
                ))
            }
        };

        let shape = self.shape().to_vec();
        let strides = self.strides().to_vec();
        let offset = self.offset();
        let dst_data = match &mut self.buffer {
            TensorBufferRefMut::Host(data) => data,
            TensorBufferRefMut::Backend(_) => {
                return Err(crate::Error::backend_failure(
                    op,
                    "destination backend buffer cannot be updated through host memory; download explicitly first",
                ))
            }
        };

        let mut src_iter = src_data.iter();
        for_each_layout_offset_col_major(&shape, &strides, offset, op, |offset| {
            let value = src_iter.next().ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "source tensor ended before destination view".to_string(),
            })?;
            let dst = dst_data
                .get_mut(offset)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op,
                    message: "destination view offset is outside host buffer".to_string(),
                })?;
            *dst = value.clone();
            Ok(())
        })?;
        if src_iter.next().is_some() {
            return Err(crate::Error::InvalidConfig {
                op,
                message: "source tensor has elements remaining after destination copy".to_string(),
            });
        }
        Ok(())
    }

    /// Borrow this mutable view as a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.as_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read_only(&self) -> TypedTensorView<'_, T, R> {
        let buffer = match &self.buffer {
            TensorBufferRefMut::Host(data) => TensorBufferRef::Host(data),
            TensorBufferRefMut::Backend(buffer) => TensorBufferRef::Backend(Arc::clone(buffer)),
        };
        TypedTensorView {
            buffer,
            layout: self.layout.clone(),
            placement: self.placement.clone(),
        }
    }

    /// Convert this mutable view into a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.into_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_read_only(self) -> TypedTensorView<'a, T, R> {
        let buffer = match self.buffer {
            TensorBufferRefMut::Host(data) => TensorBufferRef::Host(data),
            TensorBufferRefMut::Backend(buffer) => TensorBufferRef::Backend(buffer),
        };
        TypedTensorView {
            buffer,
            layout: self.layout,
            placement: self.placement,
        }
    }

    /// Return a mutable metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn transpose_view(
        self,
        axes: impl AsRef<[usize]>,
    ) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
        let Self {
            buffer,
            layout,
            placement,
        } = self;
        let layout = layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        match buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(buffer),
                layout,
                placement,
            }),
        }
    }

    /// Explicit alias for [`TypedTensorViewMut::transpose_view`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// assert_eq!(view.try_permute_axes(&[1, 0])?.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_permute_axes(
        &mut self,
        axes: &[usize],
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(&mut *data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
                layout,
                placement,
            }),
        }
    }

    /// Return a mutable metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![1], 0, &mut data)?;
    /// *view.try_slice(&[StridedSliceSpec::reverse()])?.get_mut(&[0]).unwrap() = 30;
    /// assert_eq!(view.get(&[2]), Some(&30));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice(
        &mut self,
        slices: &[StridedSliceSpec],
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorViewMut::try_slice")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_slice", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_slice", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(&mut *data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
                layout,
                placement,
            }),
        }
    }

    /// Return a mutable metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.try_slice_axis(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice_axis(
        &mut self,
        axis: usize,
        slice: StridedSliceSpec,
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorViewMut::try_slice_axis",
        )?;
        self.try_slice(&slices)
    }

    /// Return two mutable metadata-only slices when their physical ranges are disjoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![4], vec![1], 0, &mut data)?;
    /// let (left, right) = view
    ///     .try_multi_slice_mut(
    ///         &[StridedSliceSpec::new(0, Some(2), 1)],
    ///         &[StridedSliceSpec::new(2, Some(4), 1)],
    ///     )
    ///     .unwrap();
    /// assert_eq!(left.shape(), &[2]);
    /// assert_eq!(right.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_multi_slice_mut(
        &mut self,
        first: &[StridedSliceSpec],
        second: &[StridedSliceSpec],
    ) -> Option<(TypedTensorViewMut<'_, T, R>, TypedTensorViewMut<'_, T, R>)> {
        let first_specs = core_slice_specs(
            first,
            self.shape(),
            "TypedTensorViewMut::try_multi_slice_mut",
        )
        .ok()?;
        let second_specs = core_slice_specs(
            second,
            self.shape(),
            "TypedTensorViewMut::try_multi_slice_mut",
        )
        .ok()?;
        let buffer_len = self.buffer.len();
        let first_layout = self.layout.slice_view(first_specs, buffer_len).ok()?;
        let second_layout = self.layout.slice_view(second_specs, buffer_len).ok()?;
        first_layout.validate_mutable_no_overlap().ok()?;
        second_layout.validate_mutable_no_overlap().ok()?;

        match (
            reachable_layout_span(
                first_layout.shape(),
                first_layout.strides(),
                first_layout.offset(),
            )
            .ok()?,
            reachable_layout_span(
                second_layout.shape(),
                second_layout.strides(),
                second_layout.offset(),
            )
            .ok()?,
        ) {
            (Some(first_span), Some(second_span)) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (first_data, second_data) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => {
                        split_two_mut_ranges(data, first_span, second_span)?
                    }
                    TensorBufferRefMut::Backend(_) => return None,
                };
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )
                .ok()?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )
                .ok()?;
                Some((first_view, second_view))
            }
            (None, Some(second_span)) => {
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => data.split_at_mut(second_span.0),
                    TensorBufferRefMut::Backend(_) => return None,
                };
                let (second_data, _) = after_start.split_at_mut(second_span.1 - second_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )
                .ok()?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )
                .ok()?;
                Some((first_view, second_view))
            }
            (Some(first_span), None) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => data.split_at_mut(first_span.0),
                    TensorBufferRefMut::Backend(_) => return None,
                };
                let (first_data, _) = after_start.split_at_mut(first_span.1 - first_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )
                .ok()?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )
                .ok()?;
                Some((first_view, second_view))
            }
            (None, None) => {
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )
                .ok()?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )
                .ok()?;
                Some((first_view, second_view))
            }
        }
    }

    /// Return a mutable metadata-only dynamic-rank reshape for contiguous views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.try_reshape(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_reshape(
        &mut self,
        shape: &[usize],
    ) -> crate::Result<TypedTensorViewMut<'_, T, DynRank>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorViewMut::try_reshape",
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_reshape", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(&mut *data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
                layout,
                placement,
            }),
        }
    }
}

/// Runtime scalar dtype tag.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DType;
///
/// assert_eq!(DType::F64 as u8, DType::F64 as u8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

/// Sealed trait for scalar types that can be stored in a [`Tensor`].
///
/// This trait is implemented for `f64`, `f32`, `i32`, `i64`, `bool`,
/// [`Complex64`], and [`Complex32`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TensorScalar;
///
/// let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]);
/// assert_eq!(tensor.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
/// ```
pub trait TensorScalar: Copy + Clone + Send + Sync + 'static + private::Sealed {
    /// Real-valued counterpart of this scalar type.
    type Real: TensorScalar;

    /// The [`DType`] tag corresponding to this scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar};
    ///
    /// assert_eq!(f64::dtype(), DType::F64);
    /// assert_eq!(f32::dtype(), DType::F32);
    /// ```
    fn dtype() -> DType;

    /// Wrap typed column-major data into a [`Tensor`] enum variant.
    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor;

    /// Try to borrow the host data from a [`Tensor`].
    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]>;

    /// Try to mutably borrow the host data from a [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    /// <f64 as TensorScalar>::try_as_slice_mut(&mut tensor).unwrap()[0] = 3.0;
    ///
    /// assert_eq!(tensor.as_slice::<f64>().unwrap(), &[3.0]);
    /// ```
    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]>;

    /// Try to extract a [`TypedTensor<Self>`] from a dynamic [`Tensor`].
    ///
    /// Returns `None` if the tensor dtype does not match `Self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    /// let typed = <f64 as TensorScalar>::try_into_typed(tensor).unwrap();
    ///
    /// assert_eq!(typed.as_slice(), &[1.0, 2.0]);
    /// ```
    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

impl TensorScalar for f64 {
    type Real = f64;

    fn dtype() -> DType {
        DType::F64
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::F64(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::F64(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::F64(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for f32 {
    type Real = f32;

    fn dtype() -> DType {
        DType::F32
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::F32(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::F32(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::F32(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::F32(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for i64 {
    type Real = i64;

    fn dtype() -> DType {
        DType::I64
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::I64(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::I64(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::I64(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::I64(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for i32 {
    type Real = i32;

    fn dtype() -> DType {
        DType::I32
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::I32(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::I32(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::I32(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::I32(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for bool {
    type Real = bool;

    fn dtype() -> DType {
        DType::Bool
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::Bool(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::Bool(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::Bool(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::Bool(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for Complex64 {
    type Real = f64;

    fn dtype() -> DType {
        DType::C64
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::C64(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::C64(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::C64(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::C64(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for Complex32 {
    type Real = f32;

    fn dtype() -> DType {
        DType::C32
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::C32(TypedTensor::from_vec_col_major(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::C32(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_as_slice_mut(tensor: &mut Tensor) -> Option<&mut [Self]> {
        match tensor {
            Tensor::C32(t) => Some(t.host_data_mut()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::C32(inner) => Some(inner),
            _ => None,
        }
    }
}

/// Dynamic tensor enum over the supported scalar types.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
/// assert_eq!(t.shape(), &[2]);
/// ```
#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    I32(TypedTensor<i32>),
    I64(TypedTensor<i64>),
    Bool(TypedTensor<bool>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}

/// Dynamic read-only borrowed tensor view.
#[derive(Clone, Debug)]
pub enum TensorView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex<f32>>),
    C64(TypedTensorView<'a, Complex<f64>>),
}

/// Read-only tensor input accepted by synchronous eager kernels.
#[derive(Clone, Debug)]
pub enum TensorRead<'a> {
    Tensor(&'a Tensor),
    View(TensorView<'a>),
}

/// Wrap an `f64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f64>> for Tensor {
    fn from(t: TypedTensor<f64>) -> Self {
        Tensor::F64(t)
    }
}

/// Wrap an `f32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]);
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f32>> for Tensor {
    fn from(t: TypedTensor<f32>) -> Self {
        Tensor::F32(t)
    }
}

/// Wrap an `i64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]);
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I64);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i64>> for Tensor {
    fn from(t: TypedTensor<i64>) -> Self {
        Tensor::I64(t)
    }
}

/// Wrap an `i32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i32, 2]);
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I32);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i32>> for Tensor {
    fn from(t: TypedTensor<i32>) -> Self {
        Tensor::I32(t)
    }
}

/// Wrap a `bool` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![true, false]);
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::Bool);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<bool>> for Tensor {
    fn from(t: TypedTensor<bool>) -> Self {
        Tensor::Bool(t)
    }
}

/// Wrap a [`Complex64`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex64::new(1.0, 2.0)],
/// );
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f64>>> for Tensor {
    fn from(t: TypedTensor<Complex<f64>>) -> Self {
        Tensor::C64(t)
    }
}

/// Wrap a [`Complex32`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex32;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex32::new(1.0, 2.0)],
/// );
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f32>>> for Tensor {
    fn from(t: TypedTensor<Complex<f32>>) -> Self {
        Tensor::C32(t)
    }
}

impl<'a> TensorView<'a> {
    pub fn f32(shape: &'a [usize], data: &'a [f32]) -> crate::Result<Self> {
        Ok(Self::F32(TypedTensorView::new(shape, data)?))
    }

    pub fn f64(shape: &'a [usize], data: &'a [f64]) -> crate::Result<Self> {
        Ok(Self::F64(TypedTensorView::new(shape, data)?))
    }

    pub fn i64(shape: &'a [usize], data: &'a [i64]) -> crate::Result<Self> {
        Ok(Self::I64(TypedTensorView::new(shape, data)?))
    }

    pub fn i32(shape: &'a [usize], data: &'a [i32]) -> crate::Result<Self> {
        Ok(Self::I32(TypedTensorView::new(shape, data)?))
    }

    pub fn bool(shape: &'a [usize], data: &'a [bool]) -> crate::Result<Self> {
        Ok(Self::Bool(TypedTensorView::new(shape, data)?))
    }

    pub fn c32(shape: &'a [usize], data: &'a [Complex32]) -> crate::Result<Self> {
        Ok(Self::C32(TypedTensorView::new(shape, data)?))
    }

    pub fn c64(shape: &'a [usize], data: &'a [Complex64]) -> crate::Result<Self> {
        Ok(Self::C64(TypedTensorView::new(shape, data)?))
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        match self {
            Self::F32(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::F32(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::F64(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::F64(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::I32(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::I32(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::I64(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::I64(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::Bool(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::Bool(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::C32(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::C32(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
            Self::C64(t) => match materialize_typed_view_col_major(t, "TensorView::to_tensor") {
                Ok(tensor) => Tensor::C64(tensor),
                Err(err) => panic!("TensorView::to_tensor failed: {err}"),
            },
        }
    }
}

impl<'a> TensorRead<'a> {
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self::Tensor(tensor)
    }

    pub fn from_view(view: TensorView<'a>) -> Self {
        Self::View(view)
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub fn as_tensor(&self) -> Option<&'a Tensor> {
        match self {
            Self::Tensor(tensor) => Some(*tensor),
            Self::View(_) => None,
        }
    }

    pub fn to_tensor(&self) -> Tensor {
        match self {
            Self::Tensor(tensor) => (*tensor).clone(),
            Self::View(view) => view.to_tensor(),
        }
    }
}

/// Column-major strides derived from a shape.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3]), vec![1, 2]);
/// ```
pub fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

fn linear_offset(shape: &[usize], indices: &[usize]) -> usize {
    assert_eq!(indices.len(), shape.len());
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&idx, &extent) in indices.iter().zip(shape) {
        assert!(idx < extent, "index out of bounds");
        offset += idx * stride;
        stride *= extent;
    }
    offset
}

fn checked_shape_len(shape: &[usize], data_len: usize, op: &str) {
    let n: usize = shape.iter().product();
    assert_eq!(
        data_len, n,
        "{op}: data length {} does not match shape product {}",
        data_len, n
    );
}

fn compact_layout<R: TensorRank>(shape: impl Into<R::Shape>, op: &str) -> TensorLayout<R> {
    match TensorLayout::compact(shape.into()) {
        Ok(layout) => layout,
        Err(err) => panic!("{op}: invalid compact tensor layout: {err}"),
    }
}

fn tensor_layout_error(op: &'static str, err: tenferro_tensor_core::Error) -> crate::Error {
    match err {
        tenferro_tensor_core::Error::RankMismatch { expected, actual } => {
            crate::Error::RankMismatch {
                op,
                expected,
                actual,
            }
        }
        tenferro_tensor_core::Error::AxisOutOfBounds { axis, rank } => {
            crate::Error::AxisOutOfBounds { op, axis, rank }
        }
        tenferro_tensor_core::Error::DuplicateAxis { axis } => crate::Error::DuplicateAxis {
            op,
            axis,
            role: "permutation",
        },
        tenferro_tensor_core::Error::InvalidPermutationLength { expected, actual } => {
            crate::Error::RankMismatch {
                op,
                expected,
                actual,
            }
        }
        other => crate::Error::InvalidConfig {
            op,
            message: other.to_string(),
        },
    }
}

fn checked_view_element_count(shape: &[usize], op: &'static str) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |product, &dim| {
        if dim == 0 {
            Ok(0)
        } else {
            product
                .checked_mul(dim)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op,
                    message: format!("shape product overflows for shape {shape:?}"),
                })
        }
    })
}

fn checked_view_offset(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
) -> Option<usize> {
    if indices.len() != shape.len() {
        return None;
    }

    let mut offset = base_offset;
    for ((&index, &extent), &stride) in indices.iter().zip(shape).zip(strides) {
        if index >= extent {
            return None;
        }
        let index = isize::try_from(index).ok()?;
        let delta = index.checked_mul(stride)?;
        offset = offset.checked_add(delta)?;
    }

    usize::try_from(offset).ok()
}

fn for_each_layout_offset_col_major(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    op: &'static str,
    mut f: impl FnMut(usize) -> crate::Result<()>,
) -> crate::Result<()> {
    if shape.len() != strides.len() {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!(
                "shape rank {} does not match stride rank {}",
                shape.len(),
                strides.len()
            ),
        });
    }

    if shape.iter().any(|&dim| dim == 0) {
        return Ok(());
    }

    let mut offset = base_offset;
    if shape.is_empty() {
        let offset = usize::try_from(offset).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: "view offset is negative".to_string(),
        })?;
        return f(offset);
    }

    let mut index = vec![0; shape.len()];
    loop {
        let physical = usize::try_from(offset).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: "view offset is negative".to_string(),
        })?;
        f(physical)?;

        let mut axis = 0;
        loop {
            index[axis] += 1;
            offset =
                offset
                    .checked_add(strides[axis])
                    .ok_or_else(|| crate::Error::InvalidConfig {
                        op,
                        message: "view offset overflows".to_string(),
                    })?;
            if index[axis] < shape[axis] {
                break;
            }

            let extent = isize::try_from(shape[axis]).map_err(|_| crate::Error::InvalidConfig {
                op,
                message: "shape extent does not fit in isize".to_string(),
            })?;
            let rewind =
                strides[axis]
                    .checked_mul(extent)
                    .ok_or_else(|| crate::Error::InvalidConfig {
                        op,
                        message: "stride rewind overflows".to_string(),
                    })?;
            offset = offset
                .checked_sub(rewind)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op,
                    message: "view offset rewind overflows".to_string(),
                })?;
            index[axis] = 0;
            axis += 1;
            if axis == shape.len() {
                return Ok(());
            }
        }
    }
}

fn reachable_layout_span(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> crate::Result<Option<(usize, usize)>> {
    if shape.iter().any(|&extent| extent == 0) {
        return Ok(None);
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps =
            isize::try_from(extent.saturating_sub(1)).map_err(|_| crate::Error::InvalidConfig {
                op: "TypedTensorViewMut::try_multi_slice_mut",
                message: "shape extent does not fit in isize".to_string(),
            })?;
        let end = stride
            .checked_mul(steps)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensorViewMut::try_multi_slice_mut",
                message: "stride span overflows".to_string(),
            })?;
        let (axis_min, axis_max) = if end < 0 { (end, 0) } else { (0, end) };
        min_offset =
            min_offset
                .checked_add(axis_min)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "TypedTensorViewMut::try_multi_slice_mut",
                    message: "minimum reachable offset overflows".to_string(),
                })?;
        max_offset =
            max_offset
                .checked_add(axis_max)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "TypedTensorViewMut::try_multi_slice_mut",
                    message: "maximum reachable offset overflows".to_string(),
                })?;
    }

    let min_offset = usize::try_from(min_offset).map_err(|_| crate::Error::InvalidConfig {
        op: "TypedTensorViewMut::try_multi_slice_mut",
        message: "minimum reachable offset is negative".to_string(),
    })?;
    let max_offset = usize::try_from(max_offset).map_err(|_| crate::Error::InvalidConfig {
        op: "TypedTensorViewMut::try_multi_slice_mut",
        message: "maximum reachable offset is negative".to_string(),
    })?;
    Ok(Some((min_offset, max_offset)))
}

fn split_two_mut_ranges<T>(
    data: &mut [T],
    first: (usize, usize),
    second: (usize, usize),
) -> Option<(&mut [T], &mut [T])> {
    if first.1 < second.0 {
        let (_, after_first_start) = data.split_at_mut(first.0);
        let (first_slice, after_first) = after_first_start.split_at_mut(first.1 - first.0 + 1);
        let (_, after_gap) = after_first.split_at_mut(second.0 - first.1 - 1);
        let (second_slice, _) = after_gap.split_at_mut(second.1 - second.0 + 1);
        Some((first_slice, second_slice))
    } else if second.1 < first.0 {
        let (_, after_second_start) = data.split_at_mut(second.0);
        let (second_slice, after_second) = after_second_start.split_at_mut(second.1 - second.0 + 1);
        let (_, after_gap) = after_second.split_at_mut(first.0 - second.1 - 1);
        let (first_slice, _) = after_gap.split_at_mut(first.1 - first.0 + 1);
        Some((first_slice, second_slice))
    } else {
        None
    }
}

fn adjusted_view_offset(offset: isize, span_start: usize) -> Option<isize> {
    let span_start = isize::try_from(span_start).ok()?;
    offset.checked_sub(span_start)
}

fn view_mut_from_layout_and_slice<'a, T: 'static, R: TensorRank>(
    layout: &TensorLayout<R>,
    offset: isize,
    data: &'a mut [T],
    placement: Placement,
) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
    let shape = R::shape_from_vec(layout.shape().to_vec().into())
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    let strides = R::strides_from_vec(layout.strides().to_vec().into())
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    TypedTensorViewMut::from_buffer_ref_mut(
        shape,
        strides,
        offset,
        TensorBufferRefMut::Host(data),
        placement,
        "TypedTensorViewMut::try_multi_slice_mut",
    )
}

fn contiguous_layout_slice<'a, T, R: TensorRank>(
    layout: &TensorLayout<R>,
    data: &'a [T],
    op: &'static str,
) -> crate::Result<&'a [T]> {
    if !layout.is_compact_col_major() {
        return Err(crate::Error::InvalidConfig {
            op,
            message: "view is not contiguous column-major".to_string(),
        });
    }
    let len = checked_view_element_count(layout.shape(), op)?;
    let start = usize::try_from(layout.offset()).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: "view offset is negative".to_string(),
    })?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: "contiguous view range overflows".to_string(),
        })?;
    data.get(start..end)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: "contiguous view range is outside host buffer".to_string(),
        })
}

fn materialize_view_buffer_col_major<T: Clone>(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    buffer: &TensorBufferRef<'_, T>,
    op: &'static str,
) -> crate::Result<Vec<T>> {
    let source = match buffer {
        TensorBufferRef::Host(data) => *data,
        TensorBufferRef::Backend(_) => return Err(crate::Error::backend_failure(
            op,
            "backend buffers cannot be materialized through host memory; download explicitly first",
        )),
    };

    let n_elements = checked_view_element_count(shape, op)?;
    let mut out = Vec::with_capacity(n_elements);
    for_each_layout_offset_col_major(shape, strides, offset, op, |physical| {
        let value = source
            .get(physical)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "view offset is outside host buffer".to_string(),
            })?;
        out.push(value.clone());
        Ok(())
    })?;
    Ok(out)
}

fn relaxed_col_major_contiguous(
    shape: &[usize],
    strides: &[isize],
    op: &'static str,
) -> crate::Result<bool> {
    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        if stride != expected {
            return Ok(false);
        }
        let extent = isize::try_from(extent).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: "shape extent does not fit in isize".to_string(),
        })?;
        expected = expected
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "contiguous stride overflows".to_string(),
            })?;
    }
    Ok(true)
}

fn reshape_layout_dyn<R: TensorRank>(
    layout: &TensorLayout<R>,
    shape: &[usize],
    buffer_len: usize,
    op: &'static str,
) -> crate::Result<TensorLayout<DynRank>> {
    match layout.reshape_view_as::<DynRank>(shape.to_vec().into(), buffer_len) {
        Ok(layout) => Ok(layout),
        Err(err) => {
            if !relaxed_col_major_contiguous(layout.shape(), layout.strides(), op)? {
                return Err(tensor_layout_error(op, err));
            }
            let from = checked_view_element_count(layout.shape(), op)?;
            let to = checked_view_element_count(shape, op)?;
            if from != to {
                return Err(tensor_layout_error(
                    op,
                    tenferro_tensor_core::Error::ReshapeElementCountMismatch { from, to },
                ));
            }
            TensorLayout::<DynRank>::compact(shape.to_vec().into())
                .and_then(|compact| {
                    TensorLayout::from_parts(
                        compact.shape().to_vec().into(),
                        compact.strides().to_vec().into(),
                        layout.offset(),
                        buffer_len,
                    )
                })
                .map_err(|err| tensor_layout_error(op, err))
        }
    }
}

fn core_slice_specs(
    slices: &[StridedSliceSpec],
    shape: &[usize],
    op: &'static str,
) -> crate::Result<Vec<CoreSliceSpec>> {
    if slices.len() != shape.len() {
        return Err(crate::Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: slices.len(),
        });
    }

    let mut specs = Vec::with_capacity(slices.len());
    for (slice, &axis_len) in slices.iter().zip(shape) {
        specs.push(core_slice_spec(*slice, axis_len, op)?);
    }
    Ok(specs)
}

fn core_slice_spec(
    slice: StridedSliceSpec,
    axis_len: usize,
    op: &'static str,
) -> crate::Result<CoreSliceSpec> {
    if slice.step() == 0 {
        return Err(crate::Error::InvalidConfig {
            op,
            message: "slice step must not be zero".to_string(),
        });
    }

    let start = normalize_strided_bound(slice.start(), axis_len, op, "slice start")?;
    let end = match slice.end() {
        Some(end) => normalize_strided_bound(end, axis_len, op, "slice end")?,
        None => isize::try_from(axis_len).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: format!("axis length {axis_len} does not fit in isize"),
        })?,
    };

    if slice.step() > 0 {
        return Ok(CoreSliceSpec {
            start,
            end,
            step: slice.step(),
        });
    }

    if start >= end {
        return Ok(CoreSliceSpec {
            start,
            end: start,
            step: slice.step(),
        });
    }

    Ok(CoreSliceSpec {
        start: end
            .checked_sub(1)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "negative-step slice start overflows".to_string(),
            })?,
        end: start
            .checked_sub(1)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "negative-step slice end overflows".to_string(),
            })?,
        step: slice.step(),
    })
}

fn normalize_strided_bound(
    bound: isize,
    axis_len: usize,
    op: &'static str,
    role: &'static str,
) -> crate::Result<isize> {
    let axis_len = isize::try_from(axis_len).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: format!("axis length {axis_len} does not fit in isize"),
    })?;
    let bound = if bound < 0 {
        axis_len
            .checked_add(bound)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("{role} {bound} overflows"),
            })?
    } else {
        bound
    };
    if !(0..=axis_len).contains(&bound) {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!("{role} {bound} is outside 0..={axis_len}"),
        });
    }
    Ok(bound)
}

fn slice_axis_specs(
    rank: usize,
    axis: usize,
    slice: StridedSliceSpec,
    op: &'static str,
) -> crate::Result<Vec<StridedSliceSpec>> {
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
    }

    let mut slices = vec![StridedSliceSpec::all(); rank];
    slices[axis] = slice;
    Ok(slices)
}

fn row_major_offset(shape: &[usize], indices: &[usize]) -> usize {
    let mut stride = 1;
    let mut offset = 0;
    for (&dim, &index) in shape.iter().rev().zip(indices.iter().rev()) {
        offset += index * stride;
        stride *= dim;
    }
    offset
}

fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.iter().any(|&dim| dim == 0) {
        return;
    }

    let mut index = vec![0; shape.len()];
    loop {
        f(&index);
        let mut axis = 0;
        loop {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
            if axis == shape.len() {
                return;
            }
        }
    }
}

fn for_each_row_major_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    if shape.is_empty() {
        f(&[]);
        return;
    }
    if shape.iter().any(|&dim| dim == 0) {
        return;
    }

    let mut index = vec![0; shape.len()];
    loop {
        f(&index);
        let mut axis = shape.len();
        loop {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return;
            }
        }
    }
}

fn row_major_to_col_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Vec<T> {
    checked_shape_len(shape, data.len(), "from_vec_row_major");
    let mut out = Vec::with_capacity(data.len());
    for_each_index(shape, |index| {
        out.push(data[row_major_offset(shape, index)].clone());
    });
    out
}

fn col_major_to_row_major<T: Clone>(shape: &[usize], data: Vec<T>) -> Vec<T> {
    checked_shape_len(shape, data.len(), "try_into_vec_row_major");
    if shape.is_empty() {
        return data;
    }

    let mut out = Vec::with_capacity(data.len());
    for_each_row_major_index(shape, |index| {
        out.push(data[linear_offset(shape, index)].clone());
    });
    out
}

pub(crate) fn materialize_typed_view_col_major<T: Clone + 'static, R: TensorRank>(
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<TypedTensor<T>> {
    let data = materialize_view_buffer_col_major(
        view.shape(),
        view.strides(),
        view.offset(),
        &view.buffer,
        op,
    )?;
    Ok(TypedTensor::from_vec_col_major(view.shape().to_vec(), data))
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
    }
}

fn typed_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl Into<R::Shape>,
    data: Vec<T>,
    op: &str,
) -> TypedTensor<T, R> {
    let layout = compact_layout(shape, op);
    checked_shape_len(layout.shape(), data.len(), op);
    TypedTensor {
        buffer: Buffer::Host(data),
        layout,
        placement: default_placement(),
    }
}

fn typed_tensor_from_vec_row_major<T: Clone, R: TensorRank>(
    shape: impl Into<R::Shape>,
    data: Vec<T>,
) -> TypedTensor<T, R> {
    let layout = compact_layout(shape, "from_vec_row_major");
    let data = row_major_to_col_major(layout.shape(), data);
    TypedTensor {
        buffer: Buffer::Host(data),
        layout,
        placement: default_placement(),
    }
}

fn typed_tensor_zeros<T: Clone + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> TypedTensor<T, R> {
    let layout = compact_layout(shape, "zeros");
    let n: usize = layout.shape().iter().product();
    TypedTensor {
        buffer: Buffer::Host(vec![T::zero(); n]),
        layout,
        placement: default_placement(),
    }
}

fn typed_tensor_ones<T: Clone + One + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> TypedTensor<T, R> {
    let layout = compact_layout(shape, "ones");
    let n: usize = layout.shape().iter().product();
    TypedTensor {
        buffer: Buffer::Host(vec![T::one(); n]),
        layout,
        placement: default_placement(),
    }
}

impl<T: Clone + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a zero-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]);
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn zeros(shape: impl Into<R::Shape>) -> Self {
        typed_tensor_zeros(shape)
    }
}

impl<T: Clone + One + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a one-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::ones(vec![2]);
    /// assert_eq!(t.host_data(), &[1.0, 1.0]);
    /// ```
    pub fn ones(shape: impl Into<R::Shape>) -> Self {
        typed_tensor_ones(shape)
    }
}

impl<T, R: TensorRank> TypedTensor<T, R> {
    /// Create a tensor from an existing buffer and compact column-major layout.
    ///
    /// This preserves the owned tensor invariant that layout metadata is
    /// compact column-major, including for backend-owned buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Buffer, Placement, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_buffer_col_major(
    ///     vec![2],
    ///     Buffer::Host(vec![1.0, 2.0]),
    ///     Placement {
    ///         memory_kind: tenferro_tensor::MemoryKind::UnpinnedHost,
    ///         device: None,
    ///     },
    /// );
    /// assert_eq!(tensor.shape(), &[2]);
    /// ```
    pub fn from_buffer_col_major(
        shape: impl Into<R::Shape>,
        buffer: Buffer<T>,
        placement: Placement,
    ) -> Self
    where
        T: 'static,
    {
        let layout = compact_layout(shape, "from_buffer_col_major");
        let len = match &buffer {
            Buffer::Host(data) => data.len(),
            Buffer::Backend(data) => data.len(),
        };
        checked_shape_len(layout.shape(), len, "from_buffer_col_major");
        Self {
            buffer,
            layout,
            placement,
        }
    }

    /// Number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Tensor rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Tensor layout metadata.
    ///
    /// Owned typed tensors are always compact column-major layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.layout().strides(), &[1, 2]);
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]);
    /// let view = tensor.as_view();
    /// assert_eq!(view.strides(), &[1, 2]);
    /// ```
    pub fn as_view(&self) -> TypedTensorView<'_, T, R>
    where
        T: 'static,
    {
        let buffer = match &self.buffer {
            Buffer::Host(data) => TensorBufferRef::Host(data),
            Buffer::Backend(buffer) => TensorBufferRef::Backend(Arc::clone(buffer)),
        };
        TypedTensorView {
            buffer,
            layout: self.layout.clone(),
            placement: self.placement.clone(),
        }
    }

    /// Mutably borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![1]);
    /// *tensor.as_view_mut().get_mut(&[0]).unwrap() = 2;
    /// assert_eq!(tensor.as_slice(), &[2]);
    /// ```
    pub fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R>
    where
        T: 'static,
    {
        let layout = self.layout.clone();
        let placement = self.placement.clone();
        let buffer = match &mut self.buffer {
            Buffer::Host(data) => TensorBufferRefMut::Host(data),
            Buffer::Backend(buffer) => TensorBufferRefMut::Backend(Arc::clone(buffer)),
        };
        TypedTensorViewMut {
            buffer,
            layout,
            placement,
        }
    }

    /// Consume this tensor and return its layout metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert!(t.into_layout().is_compact_col_major());
    /// ```
    pub fn into_layout(self) -> TensorLayout<R> {
        self.layout
    }
}

impl<T: Clone, R: TensorRank> TypedTensor<T, R> {
    /// Create a tensor from a column-major buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.get(&[1, 0]), &2.0);
    /// ```
    pub fn from_vec_col_major(shape: impl Into<R::Shape>, data: Vec<T>) -> Self {
        typed_tensor_from_vec_col_major(shape, data, "from_vec_col_major")
    }

    /// Create a tensor from a row-major buffer.
    ///
    /// The data is converted into tenferro's column-major physical storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_row_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec_row_major(shape: impl Into<R::Shape>, data: Vec<T>) -> Self {
        typed_tensor_from_vec_row_major(shape, data)
    }

    /// Consume this tensor and return its owned column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// let (shape, data) = t.try_into_vec_col_major().unwrap();
    /// assert_eq!(shape, vec![2]);
    /// assert_eq!(data, vec![1.0, 2.0]);
    /// ```
    pub fn try_into_vec_col_major(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let shape = self.shape().to_vec();
        match self.buffer {
            Buffer::Host(data) => Ok((shape, data)),
            Buffer::Backend(_) => Err(crate::Error::backend_failure(
                "try_into_vec_col_major",
                "backend buffers cannot be exported as host Vec",
            )),
        }
    }

    /// Consume this tensor and return an owned row-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]);
    /// let (_, data) = t.try_into_vec_row_major().unwrap();
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn try_into_vec_row_major(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let shape = self.shape().to_vec();
        match self.buffer {
            Buffer::Host(data) => {
                let data = col_major_to_row_major(&shape, data);
                Ok((shape, data))
            }
            Buffer::Backend(_) => Err(crate::Error::backend_failure(
                "try_into_vec_row_major",
                "backend buffers cannot be exported as host Vec",
            )),
        }
    }

    /// Borrow the host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.host_data(), &[1.0, 2.0]);
    /// ```
    pub fn host_data(&self) -> &[T] {
        match &self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data called on backend buffer"),
        }
    }

    /// View the tensor data as a flat slice.
    ///
    /// This is an alias for `host_data()` for API consistency with
    /// `Tensor::as_slice`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.as_slice(), &[1.0, 2.0]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.host_data()
    }

    /// Mutably borrow the host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![2]);
    /// t.host_data_mut()[0] = 3.0;
    /// assert_eq!(t.host_data(), &[3.0, 0.0]);
    /// ```
    pub fn host_data_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data_mut called on backend buffer"),
        }
    }

    /// Compute the linear physical-buffer offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]);
    /// assert_eq!(t.linear_offset(&[1, 2]), 5);
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> usize {
        linear_offset(self.shape(), indices)
    }

    /// Borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.get(&[1]), &2.0);
    /// ```
    pub fn get(&self, indices: &[usize]) -> &T {
        let off = self.linear_offset(indices);
        &self.host_data()[off]
    }

    /// Mutably borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![1]);
    /// *t.get_mut(&[0]) = 7.0;
    /// assert_eq!(t.host_data(), &[7.0]);
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let off = self.linear_offset(indices);
        &mut self.host_data_mut()[off]
    }
}

/// Element-wise conjugation helper.
pub trait ConjElem {
    fn conj_elem(self) -> Self;
}

impl ConjElem for f32 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for f64 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for Complex<f32> {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

impl ConjElem for Complex<f64> {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

impl Tensor {
    /// Create a tensor from a shape and column-major flat data.
    ///
    /// This is the `Tensor`-level equivalent of
    /// `TypedTensor::<T>::from_vec_col_major`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec_col_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        T::into_tensor(shape, data)
    }

    /// Create a tensor from a shape and row-major flat data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec_row_major<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        let data = row_major_to_col_major(&shape, data);
        Self::from_vec_col_major(shape, data)
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(t) => t.shape(),
            Tensor::F64(t) => t.shape(),
            Tensor::I32(t) => t.shape(),
            Tensor::I64(t) => t.shape(),
            Tensor::Bool(t) => t.shape(),
            Tensor::C32(t) => t.shape(),
            Tensor::C64(t) => t.shape(),
        }
    }

    /// Tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![1.0]));
    /// assert_eq!(t.dtype(), DType::F64);
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::F32(_) => DType::F32,
            Tensor::F64(_) => DType::F64,
            Tensor::I32(_) => DType::I32,
            Tensor::I64(_) => DType::I64,
            Tensor::Bool(_) => DType::Bool,
            Tensor::C32(_) => DType::C32,
            Tensor::C64(_) => DType::C64,
        }
    }

    /// Try to borrow the host data as a typed slice.
    ///
    /// Returns `None` if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]));
    /// assert_eq!(t.as_slice::<f64>(), Some([1.0, 2.0, 3.0].as_slice()));
    /// assert_eq!(t.as_slice::<f32>(), None);
    /// ```
    pub fn as_slice<T: TensorScalar>(&self) -> Option<&[T]> {
        T::try_as_slice(self)
    }

    /// Consume this tensor and return its owned column-major buffer when the
    /// dtype matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]);
    /// assert_eq!(t.try_into_vec_col_major::<f64>().unwrap().1, vec![2.0]);
    /// ```
    pub fn try_into_vec_col_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let actual = self.dtype();
        let typed = T::try_into_typed(self).ok_or(crate::Error::DTypeMismatch {
            op: "try_into_vec_col_major",
            lhs: T::dtype(),
            rhs: actual,
        })?;
        typed.try_into_vec_col_major()
    }

    /// Consume this tensor and return a row-major buffer when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// assert_eq!(t.try_into_vec_row_major::<f64>().unwrap().1, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn try_into_vec_row_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let actual = self.dtype();
        let typed = T::try_into_typed(self).ok_or(crate::Error::DTypeMismatch {
            op: "try_into_vec_row_major",
            lhs: T::dtype(),
            rhs: actual,
        })?;
        typed.try_into_vec_row_major()
    }
}

pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}
